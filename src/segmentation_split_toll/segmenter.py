"""Continuous-text API. Headings carry no structural authority."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from importlib.resources import files
import re

import numpy as np

from .algorithm import MIN_TOKENS, MAX_TOKENS, InfeasibleSegmentationError, partition
from .backend import E5Backend


def profile():
    return json.loads(files('segmentation_split_toll').joinpath('profile.json').read_text(encoding='utf-8'))


@dataclass(frozen=True)
class Fragment:
    index: int
    start: int
    end: int
    text: str
    token_count: int
    within_limits: bool


@dataclass(frozen=True)
class SegmentationResult:
    status: str
    unit_id: str | None
    source_sha256: str
    input_tokens: int
    fragments: tuple[Fragment, ...]
    candidate_atoms: int = 0
    refined_candidates: bool = False
    objective: float | None = None

    @property
    def cut_count(self):
        return max(0, len(self.fragments) - 1)

    def to_dict(self):
        return dict(asdict(self), cut_count=self.cut_count, algorithm=profile())


def paragraph_offsets(text):
    """Split blank-line paragraphs only. Every original character belongs to an atom."""
    offsets = [0]
    # CRLF must remain one line ending even if the regex engine backtracks.
    newline = r'(?:\r\n|\r(?!\n)|(?<!\r)\n)'
    for match in re.finditer(newline + r'[ \t]*(?:' + newline + r'[ \t]*)+', text):
        end = match.end()
        if end < len(text) and text[offsets[-1]:end].strip() and text[end:].strip():
            offsets.append(end)
    return offsets + [len(text)]


def refine_offsets(text, offsets, backend):
    """Refine only when coarse paragraphs cannot form a legal complete partition.

    Prefer sentence/line ends; bisect a remaining overlong span at original
    character positions. No decode/re-encode operation constructs output text.
    """
    boundaries = [0]
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        preferred = [lo + m.end() for m in re.finditer(r'[。！？.!?；;]["”’\x27]*|\r\n|\r|\n', text[lo:hi])]
        pending = [(lo, hi)]
        while pending:
            start, end = pending.pop()
            if backend.count(text[start:end]) <= 512 or end - start < 2:
                boundaries.append(end)
                continue
            middle = (start + end) // 2
            options = [p for p in preferred if start < p < end]
            cut = min(options, key=lambda p: abs(p - middle)) if options else middle
            pending.extend([(cut, end), (start, cut)])
    clean = [0]
    for end in boundaries[1:-1]:
        if text[clean[-1]:end].strip() and text[end:].strip():
            clean.append(end)
    return clean + [len(text)]


class _Intervals:
    def __init__(self, text, offsets, backend):
        self.text, self.offsets, self.backend = text, offsets, backend
        self.cache = {}

    def __call__(self, end, starts):
        missing = [(int(start), int(end)) for start in starts if (int(start), int(end)) not in self.cache]
        for offset in range(0, len(missing), 32):
            batch = missing[offset:offset + 32]
            lengths = self.backend.count_many([self.text[self.offsets[a]:self.offsets[b]] for a, b in batch])
            if len(lengths) != len(batch):
                raise ValueError('Tokenizer returned an unexpected number of counts')
            for key, length in zip(batch, lengths):
                if isinstance(length, bool) or not isinstance(length, (int, np.integer)) or length < 0:
                    raise ValueError('Tokenizer must return nonnegative integer counts')
                self.cache[key] = int(length)
        return np.array([self.cache[(int(start), int(end))] for start in starts], dtype=np.int64)

    def feasible(self):
        reachable = [0]
        for end in range(1, len(self.offsets)):
            for offset in range(0, len(reachable), 32):
                lengths = self(end, reachable[offset:offset + 32])
                if np.any((lengths >= MIN_TOKENS) & (lengths <= MAX_TOKENS)):
                    reachable.append(end)
                    break
        return reachable[-1] == len(self.offsets) - 1


class LargeTextSegmenter:
    def __init__(self, *, model_path=None, local_files_only=True, device=None, batch_size=32, backend=None):
        self.backend = backend if backend is not None else E5Backend(model_path=model_path,
            local_files_only=local_files_only, device=device, batch_size=batch_size)

    def segment(self, text: str, *, unit_id: str | None = None) -> SegmentationResult:
        if not isinstance(text, str):
            raise TypeError('text must be str')
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if text == '':
            return SegmentationResult('empty', unit_id, digest, 0, ())
        total = self.backend.count(text)
        if total <= MAX_TOKENS:
            fragment = Fragment(1, 0, len(text), text, total, total >= MIN_TOKENS)
            return SegmentationResult('passthrough' if total >= MIN_TOKENS else 'below_minimum',
                                      unit_id, digest, total, (fragment,))
        offsets = paragraph_offsets(text)
        intervals = _Intervals(text, offsets, self.backend)
        refined = not intervals.feasible()
        if refined:
            offsets = refine_offsets(text, offsets, self.backend)
            intervals = _Intervals(text, offsets, self.backend)
            if not intervals.feasible():
                raise InfeasibleSegmentationError('Cannot partition this text into complete 2000..8000 token spans, even after refining original-text candidate positions')
        vectors = self.backend.encode([text[a:b] for a, b in zip(offsets[:-1], offsets[1:])])
        if len(vectors) != len(offsets) - 1:
            raise ValueError('Encoder returned an unexpected number of vectors')
        solution = partition(vectors, intervals)
        edges = [0, *solution.cuts, len(offsets) - 1]
        fragments = []
        for index, (a, b) in enumerate(zip(edges[:-1], edges[1:]), 1):
            start, end = offsets[a], offsets[b]
            token_count = int(intervals(b, [a])[0])
            fragments.append(Fragment(index, start, end, text[start:end], token_count, True))
        return SegmentationResult('segmented', unit_id, digest, total, tuple(fragments),
                                  len(offsets) - 1, refined, solution.objective)

    def segment_many(self, texts):
        """Process independent continuous units, without concatenating them."""
        return [self.segment(text, unit_id=str(index)) for index, text in enumerate(texts)]
