Project directory structure (current snapshot):

```
bird_classification_edge/
├── Dockerfile
├── Dockerfile.benchmark
├── README.md
├── benchmark/
│   ├── README.md
│   ├── __init__.py
│   ├── benchmark_results/
│   ├── compare_predictions.py
│   ├── config/
│   ├── predict_birdnet.py
│   ├── predict_student.py
│   ├── reanalyze_results.py
│   ├── requirements.txt
│   ├── results/
│   ├── run_benchmark.py
│   ├── segment_audio.py
│   ├── segment_audio_structured.py
│   └── statistical_analysis.py
├── best_distillation_model BACKUP 24.06.pt
├── best_distillation_model.pt
├── config/
│   ├── base.yaml
│   ├── debug.yaml
│   ├── distillation.yaml
│   └── experiments/
│       └── four_birds_combined.yaml
├── birds_distillation_edge/
│   ├── __init__.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── benchmark.py
│   │   ├── distill.py
│   │   ├── extract_soft_labels.py
│   │   └── train.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── audio_utils.py
│   │   ├── bird_dataset.py
│   │   ├── dataset_factory.py
│   │   ├── empty_segment_dataset.py
│   │   ├── esc50_dataset.py
│   │   └── preprocessed_dataset.py
│   ├── datamodule.py
│   ├── distillation/
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   ├── distillation_dataset.py
│   │   └── hybrid_dataset.py
│   ├── experiment.py
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── distillation_loss.py
│   │   └── focal_loss.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── classification.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── core.py
│   │   ├── legacy_models.py
│   │   └── legacy_modules.py
│   ├── optim/
│   │   ├── __init__.py
│   │   ├── combined.py
│   │   └── combined_optimizer.py
│   ├── spectral/
│   │   ├── __init__.py
│   │   └── differentiable_spec.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       ├── paths.py
│       └── reporting.py
├── dataset_transfer/
│   ├── dataset_analyzer_realistic.py
│   ├── simple_preprocess.py
│   ├── statistiche.md
│   ├── to_process_species.txt
│   └── upload_dataset.sh
├── datasets/
│   ├── __init__.py
│   ├── audio_utils.py
│   ├── bird_dataset.py
│   ├── dataset_factory.py
│   ├── empty_segment_dataset.py
│   ├── esc50_dataset.py
│   ├── preprocessed_dataset.py
│   └── test_datasets.py
├── differentiable_spec_torch.py
├── distillation/
│   ├── __init__.py
│   ├── configs/
│   ├── datasets/
│   ├── losses/
│   ├── optimizer/
│   ├── scripts/
│   └── species.txt
├── docker-compose.yml
├── extract_soft_labels.py
├── generate_no_birds_samples.py
├── logs/
│   ├── 4birds_CF_combined_log_linear_gru64/
│   ├── 4birds_CF_combined_log_linear_gru64_10epochs/
│   ├── 4birds_CF_combined_log_linear_gru64_30epochs/
│   ├── 4birds_CF_combined_log_linear_gru64_50epochs/
│   ├── 4birds_CF_combined_log_linear_gru64_50epochs_try2/
│   ├── 4birds_CF_linear_triangular_10epochs/
│   ├── 4birds_CF_linear_triangular_30epochs/
│   ├── 4birds_CF_linear_triangular_run1/
├── logs_backup/
│   ├── 4birds_CF_combined_log_linear_gru64_10epochs/
│   ├── 4birds_CF_combined_log_linear_gru64_50epochs/
│   └── 4birds_CF_combined_log_linear_gru64_50epochs_try2/
├── models/
│   ├── distillation/
│   │   └── best_distillation_model.pt
│   └── img/
│       └── improved_phi_gru_att_architecture/
├── models.py
├── modules.py
├── present.pdf
├── requirements.txt
├── run_docker_benchmark.sh
├── run_docker_distillation.sh
├── run_docker_soft_labels.sh
├── run_docker_training.sh
├── soft_labels_complete/
│   └── soft_labels_metadata.json
├── test_soft_labels/
│   ├── soft_labels.json
│   └── soft_labels_metadata.json
├── tests/
│   ├── test_fully_learnable.py
│   ├── test_gpu_docker.sh
│   └── test_no_birds_split.py
├── train.py
└── train_distillation.py
```

Note: subdirectories listed without full expansion contain further files and folders as seen in the repository.

External repository `gemelo-ai/vocos` structure (current snapshot):

```
vocos/
├── LICENSE
├── README.md
├── configs/
│   ├── vocos-encodec.yaml
│   ├── vocos-imdct.yaml
│   ├── vocos-resnet.yaml
│   └── vocos.yaml
├── metrics/
│   ├── UTMOS.py
│   └── periodicity.py
├── notebooks/
│   └── Bark+Vocos.ipynb
├── requirements-train.txt
├── requirements.txt
├── setup.py
├── train.py
├── vocos/
│   ├── __init__.py
│   ├── dataset.py
│   ├── discriminators.py
│   ├── experiment.py
│   ├── feature_extractors.py
│   ├── heads.py
│   ├── helpers.py
│   ├── loss.py
│   ├── models.py
│   ├── modules.py
│   └── spectral_ops.py
└── .github/
    └── workflows/
        └── pypi-release.yml
```

Refactor task list for `bird_classification_edge`:

- Kickoff & scaffolding
  - Audit current imports/dependencies in `train.py`, `train_distillation.py`, `datasets/`, `distillation/` to identify relocation blockers.
  - Create top-level package: add `birds_distillation_edge/__init__.py`, `birds_distillation_edge/birds_distillation_edge/__init__.py`.
  - Within the nested package, scaffold subpackages (`data`, `models`, `losses`, `optim`, `distillation`, `spectral`, `cli`, `utils`, `metrics`) each with `__init__.py`.
  - Decide final naming for relocated modules (e.g., `differentiable_spec_torch.py` → `spectral/differentiable_spec.py`) and note required refactor steps.

- Code relocation
  - Status: dataset and distillation modules fully migrated into `birds_distillation_edge.data`/`birds_distillation_edge.distillation`; model/module files relocated under `birds_distillation_edge.models` with compatibility shims preserved for legacy imports.
  - Move dataset modules (`datasets/*.py`) into `birds_distillation_edge/data/` consolidating factory logic; update intra-package imports.
  - Place model definitions (`models.py`, `modules.py`, distillation architectures) into `birds_distillation_edge/models/` and `birds_distillation_edge/distillation/`.
  - Relocate optimizer/loss utilities (`distillation/losses`, `distillation/optimizer`) into dedicated `losses/` and `optim/` subpackages.
  - Convert root-level helper scripts (`generate_no_birds_samples.py`, `extract_soft_labels.py`, `benchmark/*.py`) into package modules under `cli/` or `tools/`.

- Lightning core implementation
  - Status: DataModule now builds supervised or distillation datasets (soft labels + metadata logging); `BirdsExperiment` blends supervised, focal, and KD losses when configured. CLI adjusts `num_classes`, seeds, and runs `trainer.fit`. Remaining TODOs: hybrid dataset mixing, validation/confusion reporting, benchmarking integration.
  - Design common data module factory (Lightning `DataModule` or helper) consolidating dataset setup for standard and distillation modes.
  - Implement `birds_distillation_edge/experiment.py` LightningModule inspired by TinyNAC:
    - Encapsulate model instantiation, training/validation/test steps, loss composition (standard vs distillation), metric logging.
    - Support manual optimization if discriminators or multi-optimizer logic required.
  - Extract optimizer configuration into `configure_optimizers` using shared helpers; ensure compatibility with multiple optimizers (e.g., generator/discriminator).
  - Add CLI entrypoint (`birds_distillation_edge/cli/train.py`) that loads config, instantiates module, sets up callbacks/loggers, and runs `Trainer`.

- Configuration system transition
  - Status: base/default configs (`config/base.yaml`, `config/debug.yaml`, `config/distillation.yaml`) aligned with Lightning; new overrides live under `config/experiments/` (e.g., `four_birds_combined.yaml`). Legacy YAMLs removed.
  - Introduce TinyNAC-style config files: `config/base.yaml` (core defaults), plus override files (`distillation.yaml`, `benchmark.yaml`, `debug.yaml`).
  - Map existing Hydra structure into new schema; document each key’s new location (`dataset`, `model`, `losses`, `trainer`, `logging`).
  - Update CLI to parse configs via Hydra or OmegaConf, validating required fields and providing defaults.
  - Status: distillation overrides consolidated under `config/distillation.yaml`; experiment-specific tweaks live in `config/experiments/`.

- Tooling & benchmarking alignment
  - Wrap benchmarking scripts into the new package (e.g., `birds_distillation_edge/cli/benchmark.py`) reusing Lightning checkpoints.
  - Ensure soft-label generation and dataset transfer utilities call into package APIs rather than standalone scripts.
  - Update Docker entry scripts to invoke `python -m birds_distillation_edge.cli.train --config-name ...` or equivalent.

- Documentation & communication
  - Rewrite `README.md` to explain the Lightning-based workflow, new commands, and package layout; reference `lightning.md` insights.
  - Create `docs/` directory for focused guides (`docs/lightning_setup.md`, `docs/configuration.md`, `docs/benchmarking.md`).
  - Provide migration notes (`docs/migration-guide.md`) detailing old vs new command/config mapping.

- Testing & validation
  - Refactor existing tests to use new import paths and instantiate Lightning components; add tests for new DataModule and LightningModule.
  - Introduce smoke tests for CLI commands (e.g., run training for 1 epoch with synthetic data via pytest).
  - Validate benchmark pipeline and distillation flow end-to-end using updated commands; compare metrics with previous runs for regression detection.

- Cleanup & packaging
  - Remove obsolete directories (`distillation/`, legacy `train*.py`, redundant scripts) once replacements verified; store critical artifacts in `archive/` if needed.
  - Update `.gitignore`, logging paths, and ensure Hydra outputs map to new structure (`logs/` inside `outputs/` or `lightning_logs/`).
  - Evaluate packaging (`pyproject.toml` or `setup.cfg`) for editable installs; ensure `requirements.txt` aligns with new dependencies.
  - Final pass: run formatters/lint, regenerate `directory.md`, and confirm repository passes CI tests.

Target directory structure (desired outcome):

```
bird_classification_edge/
├── README.md
├── LICENSE? (if introduced)
├── pyproject.toml / setup.cfg (optional packaging)
├── requirements.txt
├── config/
│   ├── base.yaml
│   ├── distillation.yaml
│   ├── benchmark.yaml
│   ├── debug.yaml
│   └── overrides/
│       └── *.yaml
├── birds_distillation_edge/
│   ├── __init__.py
│   ├── birds_distillation_edge/
│   │   ├── __init__.py
│   │   ├── cli/
│   │   │   ├── __init__.py
│   │   │   ├── train.py
│   │   │   ├── benchmark.py
│   │   │   ├── extract_soft_labels.py
│   │   │   └── generate_no_birds_samples.py
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── audio_utils.py
│   │   │   ├── datasets.py
│   │   │   ├── dataset_factory.py
│   │   │   └── preprocessing.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── core.py
│   │   │   ├── modules.py
│   │   │   └── distillation.py
│   │   ├── losses/
│   │   │   ├── __init__.py
│   │   │   ├── classification.py
│   │   │   └── distillation.py
│   │   ├── optim/
│   │   │   ├── __init__.py
│   │   │   └── schedulers.py
│   │   ├── distillation/
│   │   │   ├── __init__.py
│   │   │   ├── datasets.py
│   │   │   └── workflows.py
│   │   ├── spectral/
│   │   │   ├── __init__.py
│   │   │   └── differentiable_spec.py
│   │   ├── metrics/
│   │   │   ├── __init__.py
│   │   │   └── evaluation.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── logging.py
│   │   │   └── paths.py
│   │   ├── experiment.py
│   │   ├── datamodule.py
│   │   └── version.py
├── scripts/
│   ├── run_docker_training.sh
│   ├── run_docker_distillation.sh
│   ├── run_docker_benchmark.sh
│   └── docker-compose.yml
├── benchmarks/
│   ├── README.md
│   └── configs/
├── docs/
│   ├── lightning_setup.md
│   ├── configuration.md
│   └── migration-guide.md
├── tests/
│   ├── __init__.py
│   ├── test_training.py
│   ├── test_distillation.py
│   └── test_cli.py
├── models/
│   └── checkpoints/
├── data/ (optional placeholders or symlinks)
└── outputs/ (generated logs, gitignored)
```
