"""CTC vectorization must preserve the original CPU scalar mathematics.

Both the forward log-sum and Viterbi paths are compared against an independent,
intentionally slow state-by-state reference. Use no pretrained models/audio.
"""
import numpy as np
from scipy.special import logsumexp
from wildspeech_m87.ctc_alignment import ctc_align


def _scalar_reference(x, target, blank):
    states = [blank]
    for token in target:
        states.extend((int(token), blank))
    tmax, nstates = len(x), len(states)
    alpha = np.full(nstates, -np.inf)
    alpha[0] = x[0, blank]
    alpha[1] = x[0, states[1]]
    dp = np.full((tmax, nstates), -np.inf)
    back = np.full((tmax, nstates), -1, dtype=int)
    dp[0, 0] = x[0, blank]
    dp[0, 1] = x[0, states[1]]
    for t in range(1, tmax):
        nxt = np.full(nstates, -np.inf)
        for j, token in enumerate(states):
            prev = [j]
            if j >= 1:
                prev.append(j - 1)
            if j >= 2 and token != blank and token != states[j - 2]:
                prev.append(j - 2)
            terms = alpha[prev]
            nxt[j] = x[t, token] + logsumexp(terms)
            scores = dp[t - 1, prev]
            best = int(np.argmax(scores))
            dp[t, j] = scores[best] + x[t, token]
            back[t, j] = prev[best]
        alpha = nxt
    total = float(logsumexp(alpha[-2:]))
    state = nstates - 2 + int(np.argmax(dp[-1, -2:]))
    path = []
    for t in range(tmax - 1, -1, -1):
        path.append(state)
        if t:
            state = back[t, state]
    return total, path[::-1]


def test_ctc_vectorized_matches_scalar_forward_and_viterbi():
    rng = np.random.default_rng(401)
    for length, tokens, blank, vocab in (
        (5, [1, 2], 0, 4),
        (28, [1, 1, 2, 3], 0, 5),
        (80, [1, 3, 4, 1, 1, 2], 0, 7),
        (180, [0, 1, 1, 3, 4], 2, 7),
        (250, list(rng.integers(1, 11, size=25)), 0, 13),
    ):
        logits = rng.normal(size=(length, vocab))
        logp = logits - logsumexp(logits, axis=-1, keepdims=True)
        expected, path = _scalar_reference(logp, tokens, blank)
        fast = ctc_align(logp, tokens, blank=blank)
        assert fast["valid"]
        np.testing.assert_allclose(
            fast["log_likelihood"], expected, atol=1e-9, rtol=1e-12
        )
        assert fast["path"] == path
        assert fast["phone_frames"] == [
            np.flatnonzero(np.asarray(path) == 2 * i + 1).tolist()
            for i in range(len(tokens))
        ]
        without_path = ctc_align(logp, tokens, blank=blank, with_path=False)
        np.testing.assert_allclose(
            without_path["log_likelihood"], expected, atol=1e-9, rtol=1e-12
        )
        assert "path" not in without_path


def test_ctc_vectorized_invalid_and_repeated_token_boundaries():
    p = np.log(np.full((3, 4), 0.25))
    assert ctc_align(p, [1, 1, 1])["reason"] == "INSUFFICIENT_FRAMES"
    assert ctc_align(p, [4])["reason"] == "OUT_OF_VOCAB"
    assert ctc_align(p, [])["reason"] == "INVALID_LOGITS_OR_TARGET"
    assert ctc_align(p[:0], [1])["reason"] == "INVALID_LOGITS_OR_TARGET"
