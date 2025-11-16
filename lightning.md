# Lightning Workflow in TINYNAC

This document summarizes how the `Exploiting-NACs/TINYNAC` project organises its PyTorch Lightning training pipeline. The reference commit was cloned into `/tmp/tinynac_repo` with repository layout:

```
TINYNAC/
├── config/
├── experiment_phi.py
├── model.py
├── modules.py
├── discriminators.py
├── losses.py
├── phi.py
└── …
```

## Configuration Layer
- Hydra is used to drive experiment configuration (`experiment_phi.py:227` decorator) with YAML files under `config/`.
- `config/base.yaml` defines dataset paths, model hyperparameters, optimizer settings, loss toggles, logging backend, and Lightning trainer arguments (`config/base.yaml:1`).
- Alternative configs (e.g. `config/desktop.yaml`) can override the base values for different environments.
- Training is launched via `python experiment_phi.py --config-name=<yaml>`; Hydra injects the merged config as the `args` DictConfig.

## LightningModule: `ExperimentPhi`
- Defined in `experiment_phi.py:16`; inherits from `lightning.LightningModule`.
- Constructor wires together:
  - The core model (`SoundPhi` from `model.py`) using config-driven latent dimensions (`experiment_phi.py:30`).
  - Optional discriminator stacks (`WaveDiscriminator`, `STFTDiscriminator`) depending on `args.losses.discriminators`.
  - Reconstruction losses and TorchMetrics instances (SI-SDR/SI-SNR) controlled by config flags (`experiment_phi.py:41`).
  - Manual optimization is enabled by setting `self.automatic_optimization = False`, allowing explicit GAN-style training loops.

### Optimizers & Schedulers
- `configure_optimizers` returns one or two Adam optimizers (`experiment_phi.py:62`):
  - Generator optimizer always created for the main model parameters.
  - Discriminator optimizer instantiated only when GAN losses are enabled.
- No learning rate schedulers are used; the method returns `[optimizers], []`.

### Training Logic
- `training_step` dispatches between GAN training and standard reconstruction based on config (`experiment_phi.py:103`).
- In GAN mode:
  - Retrieves generator/discriminator optimizers via `self.optimizers()` and toggles them manually (`experiment_phi.py:113`).
  - Runs a forward pass, computes adversarial + feature matching + reconstruction losses (`train_generator` helper) and optional SI metrics.
  - Calls `self.manual_backward`, optimizer `.step()`, `.zero_grad()`, and uses `self.toggle_optimizer` / `self.untoggle_optimizer` guards.
  - Repeats analogous steps for discriminator updates using `train_discriminator`.
- In reconstruction-only mode the loop reduces to a single optimizer with manual backward/step.

### Validation Loop
- `validation_step` accumulates PESQ (via `pesq` package), SI-SDR, SI-SNR, and stores sample audio tensors (`experiment_phi.py:156`).
- `on_validation_epoch_end` aggregates lists into mean tensors, logs metrics, and, when using the WandB logger, uploads example audio clips (`experiment_phi.py:169`).
- A helper `reset_valid_outputs` keeps validation state tidy between epochs.

### Data Pipeline
- `train_dataloader` / `val_dataloader` call a shared `_make_dataloader` helper that wraps `torchaudio.datasets.LIBRITTS` (`experiment_phi.py:196`).
- Custom `VoiceDataset` class performs resampling, amplitude normalization, random segment cropping, and padding (`experiment_phi.py:203`).
- DataLoader uses config-driven batch size, 20 workers, pinned memory, and persistent workers for throughput.

### Auxiliary Hooks
- `generate` and `forward` expose inference utilities (`experiment_phi.py:89`).
- `configure_callbacks` is currently a stub, leaving room for future Lightning callbacks (`experiment_phi.py:222`).

## Trainer & Logging Orchestration
- Hydra `train` function instantiates the logger and model, then launches `Trainer.fit` (`experiment_phi.py:227`).
- Logger selection:
  - WandB logger when `args.logger == "wandb"` with project/name hard-coded (`experiment_phi.py:231`).
  - Otherwise a CSVLogger writing to `logs/exp_1`.
- Optional warm start: if `args.wandb.artifact_url` is provided, WANDB artifacts are fetched and `ExperimentPhi.load_from_checkpoint` restores weights before training (`experiment_phi.py:235`).
- Lightning `Trainer` receives config-driven `devices`, `accelerator`, and `max_steps`; no callbacks or strategies configured by default (`experiment_phi.py:241`).

## Key Takeaways for Porting
- Hydra-driven configuration cleanly separates dataset, model, loss, optimizer, and Trainer parameters.
- Manual optimization pattern (`automatic_optimization=False`) is central to mixing generator/discriminator updates.
- Validation metrics are batched and logged explicitly in `on_validation_epoch_end`, enabling audio artifact logging with WandB.
- Data loading keeps augmentation logic inside the LightningModule to avoid external data modules.
- Logger choice and artifact resume logic are handled outside the module in the Hydra entry point.

## Birds Lightning Implementation (current status)
- `birds_distillation_edge/experiment.py` instantiates `Improved_Phi_GRU_ATT`, blends supervised/focal/distillation losses based on batch masks, and logs validation/test classification reports with confusion matrices.
- `birds_distillation_edge/datamodule.py` now produces supervised or hybrid soft-label batches (with optional mixing ratio), tracks dataset metadata, and feeds it to the CLI for auto-adjusting `num_classes`.
- `birds_distillation_edge/cli/train.py` loads Hydra configs (from `config/base.yaml` and overrides under `config/experiments/`), seeds runs, aligns model params with dataset metadata, and runs `trainer.fit` (followed by optional `trainer.test`).
- Legacy directories (`datasets/`, `distillation/*`, `losses/*`) have been mirrored into the package; shims remain only for backward compatibility.
- TODOs: migrate benchmarking/soft-label scripts into CLI commands, publish updated docs/README, and add automated smoke tests for supervised/distillation runs.
