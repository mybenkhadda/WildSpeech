"""Batched PyTorch acoustic features with valid-length masks.

One padded batch -> one STFT -> shared power spectrum / mel / MFCC -> per-family statistics.
Frame-level operations run on the GPU; run-length/peak bookkeeping (cheap, branchy) runs on CPU numpy.

Families
  quality  : snr_db, noise_floor_db, peak_db, dynamic_range_db, rms_db, clipping_ratio, dc_offset,
             bandlimit_hz, hf_ratio, low_dynamic_range, no_speech
  temporal : speech_ratio, onset_latency_s, trailing_silence_s, speech_dur_s, speech_span_s, n_segments,
             mean_segment_s, n_pauses, pause_total_s, pause_ratio, mean_pause_s, max_pause_s, long_pauses,
             nucleus_count, nucleus_rate, energy_db_std
  spectral : {centroid,bandwidth,rolloff,flatness,entropy,flux,slope,zcr}_{mean,std,p10,p50,p90}
  mfcc     : mfcc{i}_mean, mfcc{i}_std, dmfcc{i}_std
  mel      : melband{j}_mean, melband{j}_std, logmel_mean
  pitch    : f0_median_st, f0_mean_st, f0_std_st, f0_range_st, voiced_ratio, jitter_proxy, f0_slope_st_s,
             voicing_strength
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .dsp import EPS, AcousticParams, dct_matrix, mel_filterbank

GPU_FAMILIES = ("quality", "temporal", "spectral", "mfcc", "mel", "pitch")
SPECTRAL_BASE = ("centroid", "bandwidth", "rolloff", "flatness", "entropy", "flux", "slope", "zcr")


def _odd(k: float) -> int:
    k = max(1, int(round(k)))
    return k if k % 2 == 1 else k + 1


def masked_mean(v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """v (B,N[,...]), w (B,N) float weights -> (B[,...])."""
    while w.dim() < v.dim():
        w = w.unsqueeze(-1)
    return (v * w).sum(1) / w.sum(1).clamp_min(1e-9)


def masked_std(v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    mu = masked_mean(v, w)
    ww = w
    while ww.dim() < v.dim():
        ww = ww.unsqueeze(-1)
    var = (((v - mu.unsqueeze(1)) ** 2) * ww).sum(1) / ww.sum(1).clamp_min(1e-9)
    return var.clamp_min(0).sqrt()


def masked_quantiles(v: torch.Tensor, mask: torch.Tensor, qs: list[float]) -> torch.Tensor:
    """(B,N) values + bool mask -> (B, len(qs)); rows without valid frames -> NaN."""
    vv = v.float().masked_fill(~mask, float("nan"))
    q = torch.tensor(qs, device=v.device, dtype=vv.dtype)
    return torch.nanquantile(vv, q, dim=1).transpose(0, 1)


def _dilate(x: torch.Tensor, k: int) -> torch.Tensor:
    return F.max_pool1d(x.unsqueeze(1), kernel_size=k, stride=1, padding=k // 2).squeeze(1)


def _erode(x: torch.Tensor, k: int) -> torch.Tensor:
    return 1.0 - _dilate(1.0 - x, k)


class GPUAcousticExtractor:
    def __init__(self, params: AcousticParams, device: str = "cuda", dtype: torch.dtype = torch.float32):
        self.p = params
        self.device = torch.device(device)
        self.dtype = dtype
        p = params
        self.window = torch.hann_window(p.win, periodic=True, device=self.device, dtype=dtype)
        self.freqs = torch.linspace(0, p.sr / 2, p.n_fft // 2 + 1, device=self.device, dtype=dtype)
        self.melfb = torch.from_numpy(mel_filterbank(p.sr, p.n_fft, p.n_mels, p.fmin, p.fmax)).to(self.device, dtype)
        self.dct = torch.from_numpy(dct_matrix(p.n_mels, p.n_mfcc)).to(self.device, dtype)
        self.win_energy = float((self.window ** 2).sum().item())
        k = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=self.device, dtype=dtype) / 10.0
        self.delta_kernel = k.view(1, 1, 5)
        self.k_close = _odd(p.min_silence_s / p.hop_s)
        self.k_open = _odd(p.min_speech_s / p.hop_s)
        fk = self.freqs / 1000.0
        self.fk_c = fk - fk.mean()
        self.fk_den = float((self.fk_c ** 2).sum().item())
        self.hf_bins = self.freqs >= p.hf_split_hz

    # ---------------------------------------------------------------- helpers
    def _pad(self, waves: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = np.array([len(w) for w in waves], dtype=np.int64)
        T = int(max(lengths.max(), self.p.n_fft // 2 + 1))
        buf = np.zeros((len(waves), T), dtype=np.float32)
        for i, w in enumerate(waves):
            buf[i, :len(w)] = w
        x = torch.from_numpy(buf)
        if self.device.type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.dtype), torch.from_numpy(lengths).to(self.device)

    def _stft_power(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p
        S = torch.stft(x, n_fft=p.n_fft, hop_length=p.hop, win_length=p.win, window=self.window,
                       center=True, pad_mode="constant", return_complex=True)
        return (S.real ** 2 + S.imag ** 2).transpose(1, 2).contiguous()  # (B, N, F)

    # ------------------------------------------------------------------- main
    @torch.no_grad()
    def extract_batch(self, waves: list[np.ndarray], families: tuple[str, ...] | list[str]) -> dict[str, list[dict]]:
        fams = set(families)
        p = self.p
        B = len(waves)
        x, lengths = self._pad(waves)
        P = self._stft_power(x)                                           # (B,N,F)
        N = P.shape[1]
        n_valid = (1 + torch.div(lengths, p.hop, rounding_mode="floor")).clamp(max=N)
        t_idx = torch.arange(N, device=self.device)
        valid = t_idx[None, :] < n_valid[:, None]                          # (B,N) bool
        e = P.sum(-1)
        e_db = 10.0 * torch.log10(2.0 * e / (p.n_fft * self.win_energy) + EPS)

        # ---------------- VAD (adaptive per-clip threshold + morphological cleanup)
        q = masked_quantiles(e_db, valid, [p.vad_noise_percentile / 100.0, 0.5, p.vad_peak_percentile / 100.0])
        noise_db, peak_db = q[:, 0], q[:, 2]
        thr = torch.maximum(noise_db + p.vad_offset_db, peak_db - p.vad_max_below_peak_db)
        raw = (e_db > thr[:, None]) & valid
        sp = _erode(_dilate(raw.to(self.dtype), self.k_close), self.k_close)    # closing: fill short silences
        sp = _dilate(_erode(sp, self.k_open), self.k_open)                      # opening: drop short blips
        speech = (sp > 0.5) & valid
        n_speech = speech.sum(1)
        use_speech = n_speech >= p.spectral_min_frames
        frame_sel = torch.where(use_speech[:, None], speech, valid)
        w = frame_sel.to(self.dtype)

        out: dict[str, list[dict]] = {f: [dict() for _ in range(B)] for f in fams}
        to_np = lambda t: t.detach().float().cpu().numpy()  # noqa: E731

        # ---------------- quality
        if "quality" in fams:
            nonspeech = valid & ~speech
            med_s = masked_quantiles(e_db, speech, [0.5])[:, 0]
            med_n = masked_quantiles(e_db, nonspeech, [0.5])[:, 0]
            sample_valid = torch.arange(x.shape[1], device=self.device)[None, :] < lengths[:, None]
            sv = sample_valid.to(self.dtype)
            nsv = sv.sum(1).clamp_min(1)
            clip_ratio = ((x.abs() >= 0.999).to(self.dtype) * sv).sum(1) / nsv
            dc = (x * sv).sum(1) / nsv
            rms_db = 10 * torch.log10((x ** 2 * sv).sum(1) / nsv + EPS)
            ltas = (P * valid.unsqueeze(-1).to(self.dtype)).sum(1) / n_valid[:, None].clamp_min(1)
            ltas_db = 10 * torch.log10(ltas + EPS)
            above = ltas_db > (ltas_db.max(1, keepdim=True).values - p.bandlimit_drop_db)
            last = (above.shape[1] - 1) - torch.argmax(above.flip(1).to(self.dtype), dim=1)
            bandlimit = self.freqs[last]
            Psel = P * w.unsqueeze(-1)
            hf = Psel[..., self.hf_bins].sum((1, 2)) / Psel.sum((1, 2)).clamp_min(EPS)
            vals = {"snr_db": med_s - med_n, "noise_floor_db": noise_db, "peak_db": peak_db,
                    "dynamic_range_db": peak_db - noise_db, "rms_db": rms_db, "clipping_ratio": clip_ratio,
                    "dc_offset": dc, "bandlimit_hz": bandlimit, "hf_ratio": hf,
                    "low_dynamic_range": ((peak_db - noise_db) < p.vad_min_dynamic_range_db).to(self.dtype),
                    "no_speech": (n_speech == 0).to(self.dtype)}
            self._scatter(out["quality"], {k: to_np(v) for k, v in vals.items()})

        # ---------------- temporal (masks on CPU)
        if "temporal" in fams:
            sm, ev, nv = to_np(speech.to(torch.uint8)).astype(bool), to_np(e_db), to_np(n_valid).astype(int)
            for i in range(B):
                out["temporal"][i].update(self._temporal_from_mask(sm[i, :nv[i]], ev[i, :nv[i]]))

        # ---------------- spectral
        if "spectral" in fams:
            Psum = P.sum(-1, keepdim=True) + EPS
            pn = P / Psum
            cen = (pn * self.freqs).sum(-1)
            bw = ((pn * (self.freqs - cen.unsqueeze(-1)) ** 2).sum(-1)).clamp_min(0).sqrt()
            cs = torch.cumsum(pn, -1)
            ridx = (cs < 0.85).sum(-1).clamp(max=P.shape[-1] - 1)
            roll = self.freqs[ridx]
            flat = torch.exp(torch.log(P + EPS).mean(-1)) / (P.mean(-1) + EPS)
            ent = -(pn * torch.log(pn + EPS)).sum(-1) / float(np.log(P.shape[-1]))
            mag = P.sqrt()
            mn = mag / (mag.norm(dim=-1, keepdim=True) + EPS)
            flux = torch.zeros_like(cen)
            flux[:, 1:] = (mn[:, 1:] - mn[:, :-1]).norm(dim=-1)
            ldb = 10 * torch.log10(P + EPS)
            slope = ((ldb - ldb.mean(-1, keepdim=True)) * self.fk_c).sum(-1) / self.fk_den
            zc = (x[:, 1:] * x[:, :-1] < 0).to(self.dtype)
            zc = F.pad(zc, (0, 1))
            zcr = F.avg_pool1d(zc.unsqueeze(1), kernel_size=p.n_fft, stride=p.hop, padding=p.n_fft // 2,
                               count_include_pad=False).squeeze(1)[:, :N]
            if zcr.shape[1] < N:
                zcr = F.pad(zcr, (0, N - zcr.shape[1]))
            flux_sel = frame_sel.clone()
            flux_sel[:, 0] = False
            flux_sel[:, 1:] &= frame_sel[:, :-1]
            feats = {"centroid": (cen, frame_sel), "bandwidth": (bw, frame_sel), "rolloff": (roll, frame_sel),
                     "flatness": (flat, frame_sel), "entropy": (ent, frame_sel), "flux": (flux, flux_sel),
                     "slope": (slope, frame_sel), "zcr": (zcr, frame_sel)}
            vals = {}
            for name, (v, m) in feats.items():
                mw = m.to(self.dtype)
                vals[f"{name}_mean"] = masked_mean(v, mw)
                vals[f"{name}_std"] = masked_std(v, mw)
                qq = masked_quantiles(v, m, [0.1, 0.5, 0.9])
                vals[f"{name}_p10"], vals[f"{name}_p50"], vals[f"{name}_p90"] = qq[:, 0], qq[:, 1], qq[:, 2]
            self._scatter(out["spectral"], {k: to_np(v) for k, v in vals.items()})

        # ---------------- mel / mfcc
        if "mfcc" in fams or "mel" in fams:
            mel = P @ self.melfb                                            # (B,N,M)
            logmel = 10 * torch.log10(mel + EPS)
            if "mfcc" in fams:
                mf = logmel @ self.dct                                       # (B,N,C)
                C = mf.shape[-1]
                seq = mf.transpose(1, 2).reshape(B * C, 1, N)
                d = F.conv1d(F.pad(seq, (2, 2), mode="replicate"), self.delta_kernel).reshape(B, C, N).transpose(1, 2)
                mu, sd, dsd = to_np(masked_mean(mf, w)), to_np(masked_std(mf, w)), to_np(masked_std(d, w))
                cols = {}
                for c in range(C):
                    cols[f"mfcc{c}_mean"], cols[f"mfcc{c}_std"], cols[f"dmfcc{c}_std"] = mu[:, c], sd[:, c], dsd[:, c]
                self._scatter(out["mfcc"], cols)
            if "mel" in fams:
                nb = p.mel_summary_bands
                edges = np.linspace(0, p.n_mels, nb + 1).astype(int)
                bands = torch.stack([10 * torch.log10(mel[..., edges[j]:edges[j + 1]].sum(-1) + EPS) for j in range(nb)], -1)
                mu, sd = to_np(masked_mean(bands, w)), to_np(masked_std(bands, w))
                cols = {"logmel_mean": to_np(masked_mean(logmel.mean(-1), w))}
                for j in range(nb):
                    cols[f"melband{j}_mean"], cols[f"melband{j}_std"] = mu[:, j], sd[:, j]
                self._scatter(out["mel"], cols)

        # ---------------- pitch (YIN on GPU, summaries on CPU)
        if "pitch" in fams:
            f0, voiced, strength = self._yin(x, N)
            f0n, vn, stn = to_np(f0), to_np((voiced & speech).to(torch.uint8)).astype(bool), to_np(strength)
            spn, nv = to_np(speech.to(torch.uint8)).astype(bool), to_np(n_valid).astype(int)
            for i in range(B):
                out["pitch"][i].update(self._pitch_summary(f0n[i, :nv[i]], vn[i, :nv[i]], stn[i, :nv[i]], spn[i, :nv[i]]))
        return out

    @torch.no_grad()
    def parity_stats(self, waves: list[np.ndarray]) -> list[dict]:
        """Simple all-valid-frame statistics used for CPU/GPU numerical parity checks."""
        p = self.p
        x, lengths = self._pad(waves)
        P = self._stft_power(x)
        N = P.shape[1]
        n_valid = (1 + torch.div(lengths, p.hop, rounding_mode="floor")).clamp(max=N)
        valid = (torch.arange(N, device=self.device)[None, :] < n_valid[:, None]).to(self.dtype)
        pn = P / (P.sum(-1, keepdim=True) + EPS)
        cen = (pn * self.freqs).sum(-1)
        ridx = (torch.cumsum(pn, -1) < 0.85).sum(-1).clamp(max=P.shape[-1] - 1)
        flat = torch.exp(torch.log(P + EPS).mean(-1)) / (P.mean(-1) + EPS)
        e_db = 10.0 * torch.log10(2.0 * P.sum(-1) / (p.n_fft * self.win_energy) + EPS)
        mf = (10 * torch.log10(P @ self.melfb + EPS)) @ self.dct
        vals = {"centroid_mean_all": masked_mean(cen, valid), "rolloff_mean_all": masked_mean(self.freqs[ridx], valid),
                "flatness_mean_all": masked_mean(flat, valid), "energy_db_mean_all": masked_mean(e_db, valid)}
        mfm = masked_mean(mf, valid)
        for c in range(mf.shape[-1]):
            vals[f"mfcc{c}_mean_all"] = mfm[:, c]
        arr = {k: v.float().cpu().numpy() for k, v in vals.items()}
        return [{k: float(v[i]) for k, v in arr.items()} for i in range(len(waves))]

    # -------------------------------------------------------------- internals
    @staticmethod
    def _scatter(rows: list[dict], cols: dict[str, np.ndarray]) -> None:
        for k, arr in cols.items():
            for i in range(len(rows)):
                v = float(arr[i])
                rows[i][k] = v if np.isfinite(v) else np.nan

    def _temporal_from_mask(self, m: np.ndarray, e_db: np.ndarray) -> dict:
        p = self.p
        hs = p.hop_s
        n = len(m)
        r = {"speech_ratio": float(m.mean()) if n else np.nan,
             "energy_db_std": float(np.std(e_db[m])) if m.any() else np.nan}
        if not m.any():
            r.update({k: np.nan for k in ("onset_latency_s", "trailing_silence_s", "speech_span_s", "mean_segment_s",
                                          "pause_ratio", "mean_pause_s", "max_pause_s", "nucleus_rate")})
            r.update({"speech_dur_s": 0.0, "n_segments": 0.0, "n_pauses": 0.0, "pause_total_s": 0.0,
                      "long_pauses": 0.0, "nucleus_count": 0.0})
            return r
        d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        seg = (ends - starts) * hs
        gaps = (starts[1:] - ends[:-1]) * hs
        span = (ends[-1] - starts[0]) * hs
        speech_dur = float(m.sum() * hs)
        r.update({"onset_latency_s": starts[0] * hs, "trailing_silence_s": (n - ends[-1]) * hs,
                  "speech_dur_s": speech_dur, "speech_span_s": span, "n_segments": float(len(starts)),
                  "mean_segment_s": float(seg.mean()), "n_pauses": float(len(gaps)),
                  "pause_total_s": float(gaps.sum()), "pause_ratio": float(gaps.sum() / span) if span > 0 else 0.0,
                  "mean_pause_s": float(gaps.mean()) if len(gaps) else 0.0,
                  "max_pause_s": float(gaps.max()) if len(gaps) else 0.0,
                  "long_pauses": float((gaps >= p.long_pause_s).sum())})
        from scipy.signal import find_peaks

        env = np.convolve(e_db, np.ones(5) / 5.0, mode="same")
        dist = max(1, int(round(p.nucleus_min_gap_s / hs)))
        peaks, _ = find_peaks(env, prominence=p.nucleus_prominence_db, distance=dist)
        peaks = peaks[m[peaks]] if len(peaks) else peaks
        r["nucleus_count"] = float(len(peaks))
        r["nucleus_rate"] = float(len(peaks) / speech_dur) if speech_dur > 0 else np.nan
        return r

    def _yin(self, x: torch.Tensor, N: int, chunk: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.p
        W = p.yin_frame_length
        tmin = max(2, int(np.floor(p.sr / p.f0_max_hz)))
        tmax = min(W // 2, int(np.ceil(p.sr / p.f0_min_hz)))
        f0s, vs, sts = [], [], []
        for s in range(0, x.shape[0], chunk):
            xc = F.pad(x[s:s + chunk], (W // 2, W // 2))
            fr = xc.unfold(1, W, p.hop)[:, :N]                                  # (b,N,W)
            if fr.shape[1] < N:
                fr = F.pad(fr, (0, 0, 0, N - fr.shape[1]))
            Xf = torch.fft.rfft(fr, n=2 * W)
            r = torch.fft.irfft(Xf.real ** 2 + Xf.imag ** 2, n=2 * W)[..., :tmax + 2]
            csq = F.pad(torch.cumsum(fr ** 2, -1), (1, 0))                       # (b,N,W+1)
            tau = torch.arange(tmax + 2, device=self.device)
            e1 = csq[..., W - tau]
            e2 = csq[..., W:W + 1] - csq[..., tau]
            d = (e1 + e2 - 2 * r).clamp_min(0)
            d[..., 0] = 0
            cum = torch.cumsum(d[..., 1:], -1)
            cmn = torch.ones_like(d)
            cmn[..., 1:] = d[..., 1:] * tau[1:] / cum.clamp_min(EPS)
            seg = cmn[..., tmin:tmax + 1]
            below = seg < p.yin_threshold
            has = below.any(-1)
            first = torch.argmax(below.to(self.dtype), -1)
            amin = torch.argmin(seg, -1)
            idx = torch.where(has, first, amin)
            # refine to the local minimum within the next few lags
            win = torch.stack([seg.gather(-1, (idx + k).clamp(max=seg.shape[-1] - 1).unsqueeze(-1)).squeeze(-1)
                               for k in range(6)], -1)
            idx = (idx + torch.argmin(win, -1)).clamp(max=seg.shape[-1] - 1)
            t0 = idx + tmin
            c0 = cmn.gather(-1, t0.unsqueeze(-1)).squeeze(-1)
            cm1 = cmn.gather(-1, (t0 - 1).clamp_min(1).unsqueeze(-1)).squeeze(-1)
            cp1 = cmn.gather(-1, (t0 + 1).clamp(max=cmn.shape[-1] - 1).unsqueeze(-1)).squeeze(-1)
            den = cm1 - 2 * c0 + cp1
            shift = torch.where(den.abs() > 1e-9, 0.5 * (cm1 - cp1) / den, torch.zeros_like(den)).clamp(-1, 1)
            period = t0.to(self.dtype) + shift
            f0s.append(p.sr / period.clamp_min(1.0))
            vs.append(c0 < p.yin_voicing_max)
            sts.append((1 - c0).clamp(0, 1))
        return torch.cat(f0s), torch.cat(vs), torch.cat(sts)

    def _pitch_summary(self, f0: np.ndarray, voiced: np.ndarray, strength: np.ndarray, speech: np.ndarray) -> dict:
        hs = self.p.hop_s
        n_sp = int(speech.sum())
        keys = ("f0_median_st", "f0_mean_st", "f0_std_st", "f0_range_st", "jitter_proxy", "f0_slope_st_s", "voicing_strength")
        if voiced.sum() < 3:
            r = {k: np.nan for k in keys}
            r["voiced_ratio"] = float(voiced.sum() / n_sp) if n_sp else np.nan
            return r
        fv = f0[voiced]
        st = 12.0 * np.log2(fv / 100.0)
        t = np.flatnonzero(voiced) * hs
        per = 1.0 / fv
        consec = np.diff(np.flatnonzero(voiced)) == 1
        dper = np.abs(np.diff(per))[consec]
        slope = float(np.polyfit(t, st, 1)[0]) if np.ptp(t) > 0 else np.nan
        return {"f0_median_st": float(np.median(st)), "f0_mean_st": float(st.mean()), "f0_std_st": float(st.std()),
                "f0_range_st": float(np.percentile(st, 90) - np.percentile(st, 10)),
                "voiced_ratio": float(voiced.sum() / n_sp) if n_sp else np.nan,
                "jitter_proxy": float(dper.mean() / per.mean()) if len(dper) else np.nan,
                "f0_slope_st_s": slope, "voicing_strength": float(strength[voiced].mean())}
