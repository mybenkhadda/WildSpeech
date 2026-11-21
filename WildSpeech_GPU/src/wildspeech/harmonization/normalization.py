"""Noise- and channel-robust representations computed on the GPU with valid-length and VAD masks.

Family `robust` (per clip, variant-specific band limits):
  cmvn_delta_std / cmvn_ddelta_std   dynamics of utterance-level CMVN-normalised MFCCs (channel-invariant)
  cmvn_lag1_autocorr, cmvn_frame_dist temporal smoothness / frame-to-frame spectral change after CMVN
  pncc{c}_mean, pncc{c}_std          PNCC-like cepstra: per-band noise-floor subtraction (10th percentile over
                                     frames) + power-law compression (1/15) instead of log
  speech_band_{low,mid,high}, bg_band_{low,mid,high}, band_snr_{low,mid,high}
                                     speech-active vs background-only band energies (dB) and their difference
  tilt_in_band                       LTAS slope (dB/kHz) over speech frames inside the variant band
All statistics are invariant to padding (masks) — see tests/test_v2_padding_batch.py.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..acoustic.dsp import EPS, AcousticParams, dct_matrix
from ..acoustic.gpu_features import GPUAcousticExtractor, _dilate, _erode, masked_mean, masked_quantiles, masked_std

N_PNCC = 13


def cmvn_numpy(frames: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Utterance-level cepstral mean and variance normalisation of (N, C) frames over the selected rows."""
    m = np.ones(len(frames), bool) if mask is None else np.asarray(mask, bool)
    sel = frames[m] if m.any() else frames
    mu, sd = sel.mean(0), sel.std(0)
    return (frames - mu) / np.where(sd > 1e-9, sd, 1.0)


class RobustFeatureExtractor(GPUAcousticExtractor):
    def __init__(self, params: AcousticParams, device: str = "cuda", dtype: torch.dtype = torch.float32):
        super().__init__(params, device, dtype)
        self.dct_pncc = torch.from_numpy(dct_matrix(params.n_mels, N_PNCC)).to(self.device, dtype)
        self.band_bins = self.freqs <= float(params.fmax)
        f = self.freqs[self.band_bins] / 1000.0
        self.fb_c = f - f.mean()
        self.fb_den = float((self.fb_c ** 2).sum().clamp_min(1e-9).item())

    def _masks(self, P: torch.Tensor, lengths: torch.Tensor):
        """Identical VAD to GPUAcousticExtractor.extract_batch (adaptive threshold + closing/opening)."""
        p = self.p
        N = P.shape[1]
        n_valid = (1 + torch.div(lengths, p.hop, rounding_mode="floor")).clamp(max=N)
        valid = torch.arange(N, device=self.device)[None, :] < n_valid[:, None]
        e_db = 10.0 * torch.log10(2.0 * P.sum(-1) / (p.n_fft * self.win_energy) + EPS)
        q = masked_quantiles(e_db, valid, [p.vad_noise_percentile / 100.0, 0.5, p.vad_peak_percentile / 100.0])
        thr = torch.maximum(q[:, 0] + p.vad_offset_db, q[:, 2] - p.vad_max_below_peak_db)
        raw = (e_db > thr[:, None]) & valid
        sp = _erode(_dilate(raw.to(self.dtype), self.k_close), self.k_close)
        sp = _dilate(_erode(sp, self.k_open), self.k_open)
        speech = (sp > 0.5) & valid
        use_speech = speech.sum(1) >= p.spectral_min_frames
        frame_sel = torch.where(use_speech[:, None], speech, valid)
        return valid, speech, frame_sel

    def _delta(self, v: torch.Tensor) -> torch.Tensor:
        B, N, C = v.shape
        seq = v.transpose(1, 2).reshape(B * C, 1, N)
        d = F.conv1d(F.pad(seq, (2, 2), mode="replicate"), self.delta_kernel)
        return d.reshape(B, C, N).transpose(1, 2)

    @torch.no_grad()
    def extract_batch(self, waves: list[np.ndarray], families=None) -> list[dict]:  # noqa: ARG002 - one family
        B = len(waves)
        x, lengths = self._pad(waves)
        P = self._stft_power(x)
        valid, speech, sel = self._masks(P, lengths)
        w = sel.to(self.dtype)
        mel = P @ self.melfb
        logmel = 10 * torch.log10(mel + EPS)
        mf = logmel @ self.dct                                            # (B,N,C)
        mu, sd = masked_mean(mf, w), masked_std(mf, w).clamp_min(1e-6)
        cm = (mf - mu.unsqueeze(1)) / sd.unsqueeze(1)
        d1 = self._delta(cm)
        d2 = self._delta(d1)
        vals = {"cmvn_delta_std": masked_std(d1, w)[:, 1:].mean(-1), "cmvn_ddelta_std": masked_std(d2, w)[:, 1:].mean(-1)}
        pair = (sel[:, 1:] & sel[:, :-1]).to(self.dtype)
        vals["cmvn_lag1_autocorr"] = masked_mean((cm[:, 1:, 1:] * cm[:, :-1, 1:]).mean(-1), pair)
        vals["cmvn_frame_dist"] = masked_mean((cm[:, 1:] - cm[:, :-1]).norm(dim=-1) / np.sqrt(cm.shape[-1]), pair)
        # PNCC-like: noise floor per band (10th percentile over valid frames), power-law compression
        floor = torch.nanquantile(mel.masked_fill(~valid.unsqueeze(-1), float("nan")), 0.10, dim=1)    # (B,M)
        clean = torch.maximum(mel - floor.unsqueeze(1), 0.01 * mel)
        pw = clean.clamp_min(EPS) ** (1.0 / 15.0)
        pw = pw / masked_mean(pw.mean(-1), w).clamp_min(EPS)[:, None, None]
        pc = pw @ self.dct_pncc
        pm, ps = masked_mean(pc, w), masked_std(pc, w)
        for c in range(N_PNCC):
            vals[f"pncc{c}_mean"] = pm[:, c]
            if c < 6:
                vals[f"pncc{c}_std"] = ps[:, c]
        # speech vs background band energies (three equal mel-index groups inside the band)
        nb = self.melfb.shape[1]
        edges = np.linspace(0, nb, 4).astype(int)
        bg = (valid & ~speech).to(self.dtype)
        sp_w = speech.to(self.dtype)
        for j, name in enumerate(("low", "mid", "high")):
            band = 10 * torch.log10(mel[..., edges[j]:edges[j + 1]].sum(-1) + EPS)
            s_mean, b_mean = masked_mean(band, sp_w), masked_mean(band, bg)
            has_s, has_b = sp_w.sum(1) > 0, bg.sum(1) > 0
            nan = torch.full_like(s_mean, float("nan"))
            s_mean, b_mean = torch.where(has_s, s_mean, nan), torch.where(has_b, b_mean, nan)
            vals[f"speech_band_{name}"], vals[f"bg_band_{name}"], vals[f"band_snr_{name}"] = s_mean, b_mean, s_mean - b_mean
        ltas = (P * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True).clamp_min(1)
        ldb = 10 * torch.log10(ltas[:, self.band_bins] + EPS)
        vals["tilt_in_band"] = ((ldb - ldb.mean(-1, keepdim=True)) * self.fb_c).sum(-1) / self.fb_den
        rows = [dict() for _ in range(B)]
        self._scatter(rows, {k: v.detach().float().cpu().numpy() for k, v in vals.items()})
        return rows
