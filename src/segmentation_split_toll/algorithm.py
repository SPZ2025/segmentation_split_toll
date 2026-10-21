"""Frozen single-scale smoothed-depth evidence and automatic exact DP."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


MIN_TOKENS = 2000
MAX_TOKENS = 8000
CUT_PENALTY = .8


class InfeasibleSegmentationError(ValueError):
    """The available original-text boundaries cannot satisfy the token limits."""


def _signal(vectors):
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError('Expected a nonempty finite (paragraphs, dimensions) array')
    return values


def boundary_evidence(vectors):
    """w=2, centered mean width 3, nearest peaks, MAD calibration to [-1, 1]."""
    signal = _signal(vectors)
    signal = signal / np.maximum(np.linalg.norm(signal, axis=1, keepdims=True), 1e-12)
    n = len(signal)
    evidence = np.zeros(n + 1)
    if n < 4:
        return evidence
    similarity = np.full(n + 1, np.nan)
    for cut in range(2, n - 1):
        left = signal[cut - 2:cut].mean(axis=0)
        right = signal[cut:cut + 2].mean(axis=0)
        left /= max(float(np.linalg.norm(left)), 1e-12)
        right /= max(float(np.linalg.norm(right)), 1e-12)
        similarity[cut] = float(np.dot(left, right))
    smooth = np.full(n + 1, np.nan)
    for cut in range(2, n - 1):
        smooth[cut] = np.mean(similarity[max(2, cut - 1):min(n - 1, cut + 2)])
    depth = np.zeros(n - 3)
    for index, cut in enumerate(range(2, n - 1)):
        left = right = smooth[cut]
        for k in range(cut - 1, 1, -1):
            if smooth[k] < left:
                break
            left = smooth[k]
        for k in range(cut + 1, n - 1):
            if smooth[k] < right:
                break
            right = smooth[k]
        depth[index] = max(0., float(left + right - 2 * smooth[cut]))
    median = float(np.median(depth))
    mad = float(np.median(np.abs(depth - median)))
    evidence[2:n - 1] = np.clip((depth - median) / (1.4826 * mad + 1e-6), -3., 3.) / 3.
    return evidence


@dataclass(frozen=True)
class Partition:
    cuts: tuple[int, ...]
    objective: float
    internal_cost: float
    boundary_reward: float
    cut_penalty: float


def partition(vectors, counts_for_end: Callable, *, evidence=None):
    """All count decisions are automatic. Counts refer to actual emitted substrings.

    counts_for_end(end, starts) returns exact counts for each interval [start,end).
    No pruning assumes token counts are additive or monotone.
    """
    signal = _signal(vectors)
    n = len(signal)
    scores = boundary_evidence(signal) if evidence is None else np.asarray(evidence, dtype=float)
    if scores.shape != (n + 1,) or not np.isfinite(scores).all() or scores[0] != 0 or scores[n] != 0:
        raise ValueError('Evidence must be finite, shape (n+1,), with zero endpoints')
    prefix = np.vstack([np.zeros(signal.shape[1]), np.cumsum(signal, axis=0)])
    squares = np.r_[0., np.cumsum(np.einsum('ij,ij->i', signal, signal))]
    best = np.full(n + 1, np.inf)
    best[0] = 0.
    number = np.full(n + 1, n + 1, dtype=int)
    number[0] = 0
    previous = np.full(n + 1, -1, dtype=int)
    selected_sse = np.zeros(n + 1)
    for end in range(1, n + 1):
        starts = np.flatnonzero(np.isfinite(best[:end]))
        if not len(starts):
            continue
        lengths = np.asarray(counts_for_end(end, starts))
        if lengths.shape != starts.shape or not np.issubdtype(lengths.dtype, np.integer) or np.any(lengths < 0):
            raise ValueError('Exact interval token counts must be nonnegative integers')
        valid = (lengths >= MIN_TOKENS) & (lengths <= MAX_TOKENS)
        delta = prefix[end] - prefix[starts]
        internal = np.maximum(0., squares[end] - squares[starts] -
                              np.einsum('ij,ij->i', delta, delta) / (end - starts))
        charge = CUT_PENALTY - scores[end] if end < n else 0.
        for start, cost in zip(starts[valid], internal[valid]):
            value = float(best[start] + cost + charge)
            count = int(number[start] + (end < n))
            if value < best[end] or (value == best[end] and count < number[end]):
                best[end], number[end], previous[end], selected_sse[end] = value, count, start, cost
    if not np.isfinite(best[n]):
        raise InfeasibleSegmentationError('No complete partition satisfies 2000..8000 tokens')
    cuts, internal, end = [], 0., n
    while end:
        internal += float(selected_sse[end])
        start = int(previous[end])
        if start:
            cuts.append(start)
        end = start
    cuts.reverse()
    reward = float(sum(scores[c] for c in cuts))
    return Partition(tuple(cuts), float(best[n]), internal, reward, CUT_PENALTY * len(cuts))
