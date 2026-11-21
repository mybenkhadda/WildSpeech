"""Valid-frame and speech-frame masked pooling of encoder hidden states (padding-invariant)."""
from __future__ import annotations

import numpy as np
import torch

POOLINGS = ("mean", "std", "speech_mean")


def lengths_to_mask(n_frames, T: int, device) -> torch.Tensor:
    n = torch.as_tensor(np.asarray(n_frames), device=device)
    return (torch.arange(T, device=device)[None, :] < n[:, None]).float()


def masked_mean(h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """h (B,T,D), m (B,T) -> (B,D)."""
    return (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp_min(1.0)


def masked_std(h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    mu = masked_mean(h, m)
    var = (((h - mu.unsqueeze(1)) ** 2) * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp_min(1.0)
    return var.clamp_min(0).sqrt()


def speech_mask_from_waves(waves: list[np.ndarray], n_frames, T: int, sr: int, device, frame_rate: float = 50.0,
                           offset_db: float = 9.0, floor_pct: float = 10.0) -> torch.Tensor:
    """Energy VAD at the encoder frame rate: frame energy > (per-clip 10th percentile + offset_db)."""
    hop = int(round(sr / frame_rate))
    out = np.zeros((len(waves), T), np.float32)
    for i, w in enumerate(waves):
        n = int(min(n_frames[i], T))
        if n <= 0:
            continue
        L = min(len(w), n * hop)
        e = np.zeros(n)
        seg = np.asarray(w[: (L // hop) * hop], np.float64).reshape(-1, hop)
        e[: len(seg)] = 10 * np.log10((seg ** 2).mean(1) + 1e-12)
        if len(seg) < n:
            e[len(seg):] = e[: len(seg)].min() if len(seg) else -120.0
        thr = np.percentile(e, floor_pct) + offset_db
        sp = e > thr
        out[i, :n] = sp if sp.any() else 1.0
    return torch.from_numpy(out).to(device)


def pool_layers(hidden_states, layers: list[int], valid: torch.Tensor, speech: torch.Tensor | None,
                poolings=POOLINGS) -> torch.Tensor:
    """-> (B, L, P, D) float32 for the requested layers and poolings."""
    outs = []
    for l in layers:
        h = hidden_states[l].float()
        per = []
        for p in poolings:
            if p == "mean":
                per.append(masked_mean(h, valid))
            elif p == "std":
                per.append(masked_std(h, valid))
            elif p == "speech_mean":
                per.append(masked_mean(h, speech * valid if speech is not None else valid))
            else:
                raise ValueError(f"unknown pooling {p}")
        outs.append(torch.stack(per, 1))
    return torch.stack(outs, 1)
