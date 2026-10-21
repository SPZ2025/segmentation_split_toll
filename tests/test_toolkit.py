import contextlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from segmentation_split_toll import LargeTextSegmenter, profile
from segmentation_split_toll.algorithm import boundary_evidence, partition, InfeasibleSegmentationError
from segmentation_split_toll.cli import main
from segmentation_split_toll.segmenter import paragraph_offsets


class CharacterBackend:
    """Deterministic synthetic tokenizer/encoder; no model or network required."""
    def __init__(self):
        self.encoding_calls = 0

    def count(self, text):
        return len(text)

    def count_many(self, texts):
        return [self.count(t) for t in texts]

    def encode(self, texts):
        self.encoding_calls += 1
        vectors = np.array([[t.count('a') + 1, t.count('b') + 1, t.count('c') + 1] for t in texts], dtype=float)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def assert_coverage(case, source, result):
    case.assertEqual(''.join(f.text for f in result.fragments), source)
    cursor = 0
    for f in result.fragments:
        case.assertEqual(f.start, cursor)
        case.assertEqual(source[f.start:f.end], f.text)
        cursor = f.end
    case.assertEqual(cursor, len(source))


class PublicApiTests(unittest.TestCase):
    def test_passthrough_inclusive_bounds_and_no_encoding(self):
        backend = CharacterBackend()
        segmenter = LargeTextSegmenter(backend=backend)
        for length in (2000, 5000, 8000):
            text = '\ufeff' + 'a' * (length - 9) + '\r\n\r\nEND!'
            self.assertEqual(len(text), length)
            r = segmenter.segment(text)
            self.assertEqual(r.status, 'passthrough')
            self.assertEqual(r.cut_count, 0)
            assert_coverage(self, text, r)
        self.assertEqual(backend.encoding_calls, 0)

    def test_short_and_empty_inputs_are_not_lost(self):
        backend = CharacterBackend()
        segmenter = LargeTextSegmenter(backend=backend)
        r = segmenter.segment('短文本\r\n')
        self.assertEqual(r.status, 'below_minimum')
        self.assertFalse(r.fragments[0].within_limits)
        assert_coverage(self, '短文本\r\n', r)
        empty = segmenter.segment('')
        self.assertEqual(empty.status, 'empty')
        self.assertEqual(empty.fragments, ())
        self.assertEqual(backend.encoding_calls, 0)

    def test_no_blank_lines_and_no_sentence_punctuation(self):
        text = 'a' * 19001
        r = LargeTextSegmenter(backend=CharacterBackend()).segment(text)
        self.assertEqual(r.status, 'segmented')
        self.assertTrue(r.refined_candidates)
        self.assertTrue(all(2000 <= f.token_count <= 8000 for f in r.fragments))
        assert_coverage(self, text, r)

    def test_sentence_fallback_preserves_crlf_and_unicode(self):
        text = '\ufeff' + ('中文论述。\r\nEnglish paragraph!\r\n' * 500) + ' END'
        r = LargeTextSegmenter(backend=CharacterBackend()).segment(text)
        self.assertTrue(r.refined_candidates)
        assert_coverage(self, text, r)

    def test_uneven_paragraphs_trigger_refinement(self):
        text = 'a' * 900 + '\n\n' + 'b' * 7900
        r = LargeTextSegmenter(backend=CharacterBackend()).segment(text)
        self.assertTrue(r.refined_candidates)
        self.assertTrue(all(f.within_limits for f in r.fragments))
        assert_coverage(self, text, r)

    def test_headings_are_ordinary_content(self):
        text = '# Heading\r\n' + 'a' * 3000 + '\r\n\r\n## Internal\r\n' + 'b' * 3000
        backend = CharacterBackend()
        r = LargeTextSegmenter(backend=backend).segment(text)
        self.assertEqual(r.status, 'passthrough')
        self.assertEqual(backend.encoding_calls, 0)
        assert_coverage(self, text, r)

    def test_paragraph_mode_preserves_all_whitespace(self):
        text = '  ' + ('a' * 2200 + '\r\n\r\n') * 5 + '\t\r\n'
        r = LargeTextSegmenter(backend=CharacterBackend()).segment(text)
        self.assertFalse(r.refined_candidates)
        assert_coverage(self, text, r)

    def test_batch_does_not_concatenate_units(self):
        segmenter = LargeTextSegmenter(backend=CharacterBackend())
        results = segmenter.segment_many(['a' * 5000, 'b' * 6000])
        self.assertEqual([r.status for r in results], ['passthrough', 'passthrough'])
        self.assertEqual([r.unit_id for r in results], ['0', '1'])

    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            LargeTextSegmenter(backend=CharacterBackend()).segment(None)

    def test_result_is_json_serializable_and_profile_frozen(self):
        r = LargeTextSegmenter(backend=CharacterBackend()).segment('a' * 2000)
        data = json.loads(json.dumps(r.to_dict()))
        self.assertEqual(data['algorithm']['cut_penalty'], .8)
        self.assertEqual(data['algorithm']['min_tokens'], 2000)
        self.assertEqual(data['algorithm']['max_tokens'], 8000)
        self.assertFalse(profile()['headings_force_boundaries'])


class SolverTests(unittest.TestCase):
    def test_exhaustive_automatic_oracle_nonadditive_counts(self):
        rng = np.random.default_rng(19)
        x = rng.normal(size=(7, 3))
        counts = np.full((8, 8), -1, dtype=np.int64)
        for a in range(7):
            for b in range(a + 1, 8):
                counts[a, b] = 1150 * (b - a) + (113 if (a + b) % 2 else -51)
        evidence = np.r_[0., rng.uniform(-1, 1, 6), 0.]
        found = partition(x, lambda b, starts: counts[starts, b], evidence=evidence)
        best = (float('inf'), 99)
        for bits in itertools.product((0, 1), repeat=6):
            cuts = [i + 1 for i, bit in enumerate(bits) if bit]
            edges = [0, *cuts, 7]
            if not all(2000 <= counts[a, b] <= 8000 for a, b in zip(edges[:-1], edges[1:])):
                continue
            sse = sum(float(np.sum((x[a:b] - x[a:b].mean(axis=0)) ** 2)) for a, b in zip(edges[:-1], edges[1:]))
            cost = sse - sum(evidence[c] for c in cuts) + .8 * len(cuts)
            best = min(best, (cost, len(cuts)))
        self.assertAlmostEqual(found.objective, best[0], places=10)
        self.assertEqual(len(found.cuts), best[1])

    def test_infeasible_and_unsupported_windows(self):
        x = np.ones((3, 2))
        np.testing.assert_array_equal(boundary_evidence(x), np.zeros(4))
        with self.assertRaises(InfeasibleSegmentationError):
            partition(x, lambda end, starts: np.full(len(starts), 1000, dtype=np.int64))

    def test_blank_line_offsets_do_not_force_headings(self):
        text = '# H\nbody\n## Subheading\nmore'
        self.assertEqual(paragraph_offsets(text), [0, len(text)])
        text = text.replace('\n', '\r\n')
        self.assertEqual(paragraph_offsets(text), [0, len(text)])

    def test_ties_prefer_fewer_cuts(self):
        x = np.ones((4, 2))
        evidence = np.array([0., .8, .8, .8, 0.])
        found = partition(x, lambda b, starts: np.array([2000 * (b - a) for a in starts]), evidence=evidence)
        self.assertEqual(found.cuts, ())


class CliTests(unittest.TestCase):
    def test_recursive_files_and_output_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, out = root / 'inputs', root / 'outputs'
            (source / 'nested').mkdir(parents=True)
            text = '\ufeff' + 'a' * 2500 + '\r\n'
            (source / 'nested/unit.md').write_bytes(text.encode('utf-8'))
            (source / 'ignore.json').write_text('{}', encoding='utf-8')
            with patch('segmentation_split_toll.cli.LargeTextSegmenter', return_value=LargeTextSegmenter(backend=CharacterBackend())):
                with contextlib.redirect_stderr(io.StringIO()):
                    status = main([str(source), '-o', str(out), '--recursive'])
            self.assertEqual(status, 0)
            result = json.loads((out / 'nested/unit.md.segments.json').read_text(encoding='utf-8'))
            self.assertEqual(result['fragments'][0]['text'], text)
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                main([str(source), '-o', str(out), '--recursive'])

    def test_profile_without_model(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(main(['--profile']), 0)
        self.assertEqual(json.loads(stream.getvalue())['model'], 'intfloat/multilingual-e5-small')


if __name__ == '__main__':
    unittest.main()
