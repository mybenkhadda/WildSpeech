"""Sequence alignment of a recognised response against the expected response.

align(ref, hyp, sub_cost)   weighted Levenshtein with backtrace -> list[Op(op, ref, hyp, i, j)]
                            op in M (match) | S (substitution) | D (deletion: ref token missing) | I (insertion)
                            i = ref position (for I: position of the next ref token), j = hyp position
phonological_align          the same with articulatory substitution costs (phonetics.g2p.phone_distance), so
                            near substitutions are preferred over deletion+insertion pairs
order_preservation          Kendall-tau of the hyp positions of matched ref tokens (1 = order preserved)
lcs_ratio                   longest common subsequence / len(ref)
parse_spans                 the CTC forced-alignment JSON written by P24 -> list[dict]
Deterministic tie-breaking (diagonal > deletion > insertion) keeps the alignment reproducible.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class Op:
    op: str
    ref: str | None
    hyp: str | None
    i: int
    j: int


def _unit(a, b) -> float:
    return 0.0 if a == b else 1.0


def align(ref: Sequence, hyp: Sequence, sub_cost: Callable[[object, object], float] = _unit,
          ins_cost: float = 1.0, del_cost: float = 1.0) -> list[Op]:
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1))
    D[:, 0] = np.arange(n + 1) * del_cost
    D[0, :] = np.arange(m + 1) * ins_cost
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j - 1] + sub_cost(ref[i - 1], hyp[j - 1]), D[i - 1, j] + del_cost, D[i, j - 1] + ins_cost)
    ops: list[Op] = []
    i, j = n, m
    eps = 1e-9
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(D[i, j] - (D[i - 1, j - 1] + sub_cost(ref[i - 1], hyp[j - 1]))) < eps:
            ops.append(Op("M" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1], i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and abs(D[i, j] - (D[i - 1, j] + del_cost)) < eps:
            ops.append(Op("D", ref[i - 1], None, i - 1, j))
            i -= 1
        else:
            ops.append(Op("I", None, hyp[j - 1], i, j - 1))
            j -= 1
    return ops[::-1]


def phonological_align(ref_phones: Sequence[str], hyp_phones: Sequence[str]) -> list[Op]:
    from .g2p import phone_distance

    # substitution cost < deletion+insertion (2) for every pair, scaled to (0, 1.5] for distinct phones
    return align(ref_phones, hyp_phones, lambda a, b: 1.5 * phone_distance(a, b))


def counts(ops: list[Op]) -> dict:
    return {k: sum(o.op == k for o in ops) for k in ("M", "S", "D", "I")}


def error_rate(ops: list[Op], n_ref: int) -> float:
    c = counts(ops)
    return (c["S"] + c["D"] + c["I"]) / max(n_ref, 1)


def lcs_ratio(ref: Sequence, hyp: Sequence) -> float:
    if not ref:
        return np.nan
    if not hyp:
        return 0.0
    prev = [0] * (len(hyp) + 1)
    for x in ref:
        cur = [0]
        for j, y in enumerate(hyp, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1] / len(ref)


def order_preservation(ref: Sequence, hyp: Sequence) -> float:
    """Kendall tau between ref order and the hyp positions of matched ref tokens (via a unit-cost alignment
    that ignores order, i.e. first unused occurrence). NaN when < 2 tokens match."""
    used, pos = set(), []
    for x in ref:
        k = next((j for j, y in enumerate(hyp) if y == x and j not in used), None)
        if k is not None:
            used.add(k)
            pos.append(k)
    if len(pos) < 2:
        return np.nan
    conc = disc = 0
    for a in range(len(pos)):
        for b in range(a + 1, len(pos)):
            if pos[b] > pos[a]:
                conc += 1
            elif pos[b] < pos[a]:
                disc += 1
    return (conc - disc) / max(conc + disc, 1)


def ops_to_string(ops: list[Op]) -> str:
    return "|".join(f"{o.op}:{o.ref or '-'}:{o.hyp or '-'}:{o.i}" for o in ops)


def ops_from_string(s: str) -> list[Op]:
    out = []
    for part in str(s or "").split("|"):
        if not part:
            continue
        k, r, h, p = part.split(":")
        out.append(Op(k, None if r == "-" else r, None if h == "-" else h, int(p), -1))
    return out


def parse_spans(s) -> list[dict]:
    if s is None or (isinstance(s, float) and np.isnan(s)) or s == "":
        return []
    try:
        v = json.loads(s) if isinstance(s, str) else list(s)
        return [d for d in v if isinstance(d, dict) and {"p", "s", "e"} <= set(d)]
    except Exception:  # noqa: BLE001 - corrupt or legacy span string
        return []
