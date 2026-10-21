"""File and directory entry points; source newline characters are preserved."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .algorithm import InfeasibleSegmentationError
from .backend import ModelUnavailableError
from .segmenter import LargeTextSegmenter, profile


def main(argv=None):
    parser = argparse.ArgumentParser(description='Automatically split continuous text into 2000–8000 token large fragments.')
    parser.add_argument('input', nargs='?', help='UTF-8 .txt/.md file, directory, or - for stdin')
    parser.add_argument('-o', '--out-dir', type=Path, help='Output directory; single file/stdin defaults to stdout')
    parser.add_argument('--recursive', action='store_true', help='Include subdirectories')
    parser.add_argument('--encoding', default='utf-8')
    parser.add_argument('--model-path', type=Path, help='Local copy of multilingual-e5-small')
    parser.add_argument('--download-model', action='store_true', help='Allow downloading the pinned E5 model; otherwise offline cache only')
    parser.add_argument('--device', help='cpu, cuda, etc.; default follows sentence-transformers')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--profile', action='store_true', help='Print the frozen algorithm profile')
    args = parser.parse_args(argv)
    if args.profile:
        print(json.dumps(profile(), ensure_ascii=False, indent=2))
        return 0
    if args.input is None:
        parser.error('input is required unless --profile is used')
    source = None if args.input == '-' else Path(args.input)
    if source is not None and not source.exists():
        parser.error(f'Input does not exist: {source}')
    if source is not None and source.is_dir() and args.out_dir is None:
        parser.error('Directory input requires --out-dir')
    if source is not None and source.is_dir() and source.resolve().is_relative_to(args.out_dir.resolve()):
        parser.error('Output directory must not equal or contain the input directory')
    if source is None:
        paths = [None]
    elif source.is_dir():
        candidates = source.rglob('*') if args.recursive else source.iterdir()
        output_root = args.out_dir.resolve()
        paths = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in ('.md', '.txt')
                       and not p.resolve().is_relative_to(output_root))
    else:
        paths = [source]
    if not paths:
        parser.error('No .txt or .md input files found')
    jobs = []
    for path in paths:
        relative = Path('stdin') if path is None else path.relative_to(source) if source.is_dir() else Path(path.name)
        destination = args.out_dir / relative.with_name(relative.name + '.segments.json') if args.out_dir else None
        if destination is not None and destination.exists() and not args.overwrite:
            parser.error(f'Refusing to overwrite: {destination}')
        jobs.append((path, relative, destination))
    segmenter = LargeTextSegmenter(model_path=args.model_path, local_files_only=not args.download_model,
                                   device=args.device, batch_size=args.batch_size)
    failed = False
    for path, relative, destination in jobs:
        try:
            if path is None:
                text = sys.stdin.buffer.read().decode(args.encoding)
            else:
                with path.open('r', encoding=args.encoding, newline='') as stream:
                    text = stream.read()
            result = segmenter.segment(text, unit_id=relative.as_posix())
            payload = json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + '\n'
            if destination is None:
                print(payload, end='')
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open('w' if args.overwrite else 'x', encoding='utf-8', newline='') as stream:
                    stream.write(payload)
                print(f'{relative}: {result.status}, {len(result.fragments)} fragments', file=sys.stderr)
        except (InfeasibleSegmentationError, ModelUnavailableError, OSError, ValueError) as exc:
            failed = True
            print(f'{relative}: {type(exc).__name__}: {exc}', file=sys.stderr)
    return 1 if failed else 0
