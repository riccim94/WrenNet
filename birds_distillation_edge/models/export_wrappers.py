"""Export-time model wrappers.

These helpers build lightweight modules that reuse parts of the trained
Lightning models but accept pre-computed inputs (e.g. log-mel spectrograms)
so that ONNX export does not need to include torchaudio's ``stft`` graph.
"""

from __future__ import annotations

import copy
from typing import Tuple

import torch
from torch import nn


class PostMelExportWrapper(nn.Module):
    """Reuse the feature extractor/classifier while skipping torchaudio ops."""

    def __init__(self, source_model: nn.Module) -> None:
        super().__init__()
        self._base = copy.deepcopy(source_model)
        self.phi = self._base.phi
        self.gru = self._base.gru
        self.projection = getattr(self._base, "projection", None)
        self.attention = (
            getattr(self._base, "keyword_attention", None)
            or getattr(self._base, "attention", None)
        )
        self.dropout = getattr(self._base, "dropout", None)
        self.classifier = getattr(self._base, "fc", None) or getattr(self._base, "fc2", None)
        if self.classifier is None:
            raise AttributeError("Could not locate classifier layer (fc/fc2) on model")

    @staticmethod
    def expected_input_shape(log_mel: torch.Tensor) -> Tuple[int, int]:
        if log_mel.dim() != 4:
            raise ValueError(
                "PostMelExportWrapper expects input shaped (batch, 1, n_mels, time)."
            )
        return log_mel.shape[2], log_mel.shape[3]

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        if log_mel.dim() not in (3, 4):
            raise RuntimeError(
                "Expected log-mel input shaped (batch, 1, n_mels, time) or (batch, n_mels, time)."
            )

        x = log_mel
        if x.dim() == 4 and x.size(1) == 1:
            mean = x.mean(dim=(2, 3), keepdim=True)
            std = x.std(dim=(2, 3), keepdim=True) + 1e-5
            x = (x - mean) / std
            x = x.squeeze(1)
        elif x.dim() == 3:
            mean = x.mean(dim=(1, 2), keepdim=True)
            std = x.std(dim=(1, 2), keepdim=True) + 1e-5
            x = (x - mean) / std
        else:
            raise RuntimeError("Unexpected log-mel tensor shape: %s" % (x.shape,))

        if hasattr(self._base, "_align_feature_dimension"):
            x = self._base._align_feature_dimension(x)

        x = self.phi(x)
        x = x.permute(0, 2, 1).contiguous()
        x, _ = self.gru(x)
        if self.projection is not None:
            x = self.projection(x)
        if self.attention is not None:
            attn_out = self.attention(x)
            if isinstance(attn_out, tuple):
                x = attn_out[0]
            else:
                x = attn_out
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.classifier(x)
        return x


__all__ = ["PostMelExportWrapper"]
