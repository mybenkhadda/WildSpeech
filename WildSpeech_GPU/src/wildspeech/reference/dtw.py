"""Dynamic time warping: exact numpy reference + batched GPU implementation with vectorised backtracking.

Local cost = Euclidean distance between frames / sqrt(dim). Steps: diagonal, vertical, horizontal (unit weight).
Optional Sakoe-Chiba band |i/n - j/m| <= band_frac. Outputs per pair:
  cost               accumulated cost at (n, m)
  path_len           number of steps on the optimal path;   norm_cost = cost / path_len
  local_max          max over the path of the moving average (window w) of local costs  (localized deviation)
  local_max_pos      relative query position (i/n) of that maximum
  nondiag_frac       share of horizontal / vertical steps (tempo irregularity)
  diag_dev           mean |i/n - j/m| along the path (global warping)
The GPU version fills the DP matrix by anti-diagonal wavefronts for a whole batch of pairs and backtracks all
pairs simultaneously (one vectorised step per path position), so no Python loop runs per pair.
"""
from __future__ import annotations

import numpy as np

INF = 1e18


def local_costs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d2 = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T
    return np.sqrt(np.clip(d2, 0, None)) / np.sqrt(a.shape[1])


def dtw_numpy(a: np.ndarray, b: np.ndarray, band_frac: float | None = None, window: int = 10) -> dict:
    return dtw_with_path(a, b, band_frac, window)[0]


def dtw_with_path(a: np.ndarray, b: np.ndarray, band_frac: float | None = None, window: int = 10):
    """-> (stats dict, local cost matrix C, optimal path [(i, j), ...]) — used for alignment figures and tests."""
    n, m = len(a), len(b)
    C = local_costs(a, b)
    if band_frac is not None:
        ii = (np.arange(n)[:, None] + 0.5) / n
        jj = (np.arange(m)[None, :] + 0.5) / m
        C = np.where(np.abs(ii - jj) <= band_frac, C, INF)
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        cand = (D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
        k = int(np.argmin(cand))
        i, j = (i - 1, j - 1) if k == 0 else ((i - 1, j) if k == 1 else (i, j - 1))
    path = path[::-1]
    return _path_stats(C, path, D[n, m], n, m, window), C, path


def _path_stats(C, path, cost, n, m, window) -> dict:
    lc = np.array([C[i, j] for i, j in path])
    steps = np.diff(np.array(path), axis=0) if len(path) > 1 else np.zeros((0, 2))
    nondiag = float(np.mean(~((steps[:, 0] == 1) & (steps[:, 1] == 1)))) if len(steps) else 0.0
    w = max(1, min(window, len(lc)))
    mov = np.convolve(lc, np.ones(w) / w, mode="valid") if len(lc) else np.array([np.nan])
    k = int(np.argmax(mov)) if len(mov) else 0                       # first window with the largest mean
    ipos = path[k][0] / max(n - 1, 1) if path else np.nan            # query position where it starts
    dev = float(np.mean([abs(i / max(n - 1, 1) - j / max(m - 1, 1)) for i, j in path])) if path else np.nan
    return {"cost": float(cost), "path_len": float(len(path)), "norm_cost": float(cost / max(len(path), 1)),
            "local_max": float(mov.max()) if len(mov) else np.nan, "local_max_pos": float(ipos),
            "nondiag_frac": nondiag, "diag_dev": dev}


def dtw_batch_torch(A: list[np.ndarray], B: list[np.ndarray], device: str = "cuda", band_frac: float | None = None,
                    window: int = 10) -> list[dict]:
    """Batched DTW of pairs (A[k], B[k]); identical results to dtw_numpy (float32 arithmetic)."""
    import torch

    P = len(A)
    if P == 0:
        return []
    dev = torch.device(device)
    n = torch.tensor([len(a) for a in A], device=dev)
    m = torch.tensor([len(b) for b in B], device=dev)
    N, M = int(n.max()), int(m.max())
    d = A[0].shape[1]
    Ta = torch.zeros((P, N, d), device=dev)
    Tb = torch.zeros((P, M, d), device=dev)
    for k in range(P):
        Ta[k, : len(A[k])] = torch.from_numpy(np.asarray(A[k], np.float32))
        Tb[k, : len(B[k])] = torch.from_numpy(np.asarray(B[k], np.float32))
    C = torch.cdist(Ta, Tb) / float(np.sqrt(d))                            # (P,N,M)
    ii = (torch.arange(N, device=dev)[None, :, None] + 0.5) / n[:, None, None]
    jj = (torch.arange(M, device=dev)[None, None, :] + 0.5) / m[:, None, None]
    if band_frac is not None:
        C = torch.where((ii - jj).abs() <= band_frac, C, torch.full_like(C, INF))
    # float32 DP matrices (memory-bound min/add; consumer GPUs have little FP64 throughput); per-pair
    # accumulators below are float64.
    D = torch.full((P, N + 1, M + 1), INF, device=dev, dtype=torch.float32)
    D[:, 0, 0] = 0.0
    Cd = C.float()
    for k in range(2, N + M + 1):
        i = torch.arange(max(1, k - M), min(N, k - 1) + 1, device=dev)
        if len(i) == 0:
            continue
        j = k - i
        best = torch.minimum(torch.minimum(D[:, i - 1, j - 1], D[:, i - 1, j]), D[:, i, j - 1])
        D[:, i, j] = Cd[:, i - 1, j - 1] + best
    # ---- vectorised backtracking from (n_p, m_p)
    bi, bj = n.clone(), m.clone()
    ar = torch.arange(P, device=dev)
    w = max(1, int(window))
    ring = torch.zeros((P, w), device=dev, dtype=torch.float64)
    plen = torch.zeros(P, device=dev, dtype=torch.float64)
    ndiag = torch.zeros(P, device=dev, dtype=torch.float64)
    dsum = torch.zeros(P, device=dev, dtype=torch.float64)
    lmax = torch.full((P,), -1.0, device=dev, dtype=torch.float64)
    lpos = torch.zeros(P, device=dev, dtype=torch.float64)
    nf, mf = (n - 1).clamp_min(1).double(), (m - 1).clamp_min(1).double()
    # The path is visited backwards. Once >= w elements are visited, the ring holds the w consecutive elements
    # whose forward START is the current element, so the window position is the current query index; '>='
    # makes ties resolve to the earliest window, exactly as argmax in dtw_numpy.
    while True:
        active = (bi > 0) & (bj > 0)
        if not bool(active.any()):
            break
        ci, cj = (bi - 1).clamp_min(0), (bj - 1).clamp_min(0)
        lc = Cd[ar, ci, cj].double()
        a_ = active.double()
        slot = (plen % w).long()
        ring[ar, slot] = torch.where(active, lc, ring[ar, slot])
        plen = plen + a_
        mov = ring.sum(1) / w
        upd = active & (plen >= w) & (mov >= lmax)
        lmax = torch.where(upd, mov, lmax)
        lpos = torch.where(upd, ci.double() / nf, lpos)
        dsum = dsum + a_ * ((ci.double() / nf) - (cj.double() / mf)).abs()
        cand = torch.stack([D[ar, bi - 1, bj - 1], D[ar, bi - 1, bj], D[ar, bi, (bj - 1).clamp_min(0)]], 1)
        step = torch.argmin(cand, 1)
        ndiag = ndiag + a_ * (step != 0).double()
        di = torch.where(step == 2, torch.zeros_like(bi), torch.ones_like(bi))
        dj = torch.where(step == 1, torch.zeros_like(bj), torch.ones_like(bj))
        bi = torch.where(active, bi - di, bi)
        bj = torch.where(active, bj - dj, bj)
    short = plen < w                                   # path shorter than the window: one window = whole path
    lmax = torch.where(short, ring.sum(1) / plen.clamp_min(1), lmax)
    lpos = torch.where(short, torch.zeros_like(lpos), lpos)
    cost = D[ar, n, m].double()
    out = []
    plen_c, cost_c = plen.cpu().numpy(), cost.cpu().numpy()
    nd, ds, lm, lp = ndiag.cpu().numpy(), dsum.cpu().numpy(), lmax.cpu().numpy(), lpos.cpu().numpy()
    for k in range(P):
        L = max(plen_c[k], 1.0)
        out.append({"cost": float(cost_c[k]), "path_len": float(plen_c[k]), "norm_cost": float(cost_c[k] / L),
                    "local_max": float(lm[k]) if lm[k] >= 0 else np.nan, "local_max_pos": float(lp[k]),
                    "nondiag_frac": float(nd[k] / max(L - 1, 1.0)), "diag_dev": float(ds[k] / L)})
    return out
