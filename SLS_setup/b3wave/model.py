"""
ModelSLSB3: model.sls_model.ModelSLS with two additions and an identical state_dict.

    forward(x, return_emb=True)  -> (logits, h) with h the 1024-d fc1 output (the pooled
                                    embedding used by the consistency loss)
    --specaug                    SpecAugment-style time / frequency masking on the Qwen3-ASR
                                    log-mel input, training mode only, drawn with the torch RNG

The parent's forward body is reproduced line for line (model/sls_model.py: ModelSLS.forward), so
with --specaug off the outputs are identical and B2e checkpoints load with no missing or
unexpected keys (SpecAugmentMel has no parameters or buffers).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.sls_model import ModelSLS


class SpecAugmentMel(nn.Module):
    """Mask `n_time` time spans (<= t_max frames) and `n_freq` mel spans (<= f_max bins) per clip."""

    def __init__(self, n_time=2, t_max=40, n_freq=2, f_max=16):
        super().__init__()
        self.n_time, self.t_max, self.n_freq, self.f_max = int(n_time), int(t_max), int(n_freq), int(f_max)

    def extra_repr(self):
        return f"time {self.n_time}x<={self.t_max} frames, freq {self.n_freq}x<={self.f_max} bins"

    @staticmethod
    def _span_mask(length, n_spans, max_width, bs, device):
        """(bs, length) bool: union of n_spans random spans of width U{0..max_width} per row."""
        mask = torch.zeros(bs, length, dtype=torch.bool, device=device)
        if n_spans <= 0 or max_width <= 0 or length <= 0:
            return mask
        pos = torch.arange(length, device=device).view(1, length)
        for _ in range(n_spans):
            width = torch.randint(0, min(max_width, length) + 1, (bs, 1), device=device)
            start = (torch.rand(bs, 1, device=device) * (length - width + 1).clamp(min=1)).long()
            mask |= (pos >= start) & (pos < start + width)
        return mask

    def forward(self, x):
        """x: (bs, n_mel, frames). Masked entries take the clip's mean log-mel value."""
        if x.dim() != 3:
            return x
        bs, n_mel, frames = x.shape
        fill = x.mean(dim=(1, 2), keepdim=True)
        t_mask = self._span_mask(frames, self.n_time, self.t_max, bs, x.device).view(bs, 1, frames)
        f_mask = self._span_mask(n_mel, self.n_freq, self.f_max, bs, x.device).view(bs, n_mel, 1)
        return torch.where(t_mask | f_mask, fill.expand_as(x), x)


class ModelSLSB3(ModelSLS):

    def __init__(self, args, device):
        super().__init__(args, device)
        self.specaug = None
        if getattr(args, "specaug", False):
            n_t, t_max = getattr(args, "specaug_time", (2, 40))
            n_f, f_max = getattr(args, "specaug_freq", (2, 16))
            self.specaug = SpecAugmentMel(n_t, t_max, n_f, f_max)

    def forward(self, x, return_emb=False):
        if x.dim() == 3 and self.ssl_model.kind == "wav2vec2":
            x = x.squeeze(-1)
        if self.specaug is not None and self.training and x.dim() == 3:
            x = self.specaug(x)

        if self.freeze_ssl:
            with torch.no_grad():
                H = self.ssl_model(x)           # (bs, layers, frames, out_dim)
        else:
            H = self.ssl_model(x)
        x = self.sls(H)                         # (bs, frames, out_dim)

        x = x.unsqueeze(dim=1)
        x = self.first_bn(x)
        x = self.selu(x)
        x = F.max_pool2d(x, (3, 3))
        x = torch.flatten(x, 1)
        h = self.selu(self.fc1(x))
        output = self.fc2(h)
        if return_emb:
            return output, h
        return output
