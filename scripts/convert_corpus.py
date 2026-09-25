"""Convert a Parquet-only corpus into the gzipped JSONL `trigon.evals.corpora` reads.

    pip install -e ".[convert]"    # pyarrow, here and only here
    python scripts/convert_corpus.py measuring_hate_speech

`trigon.evals.corpora` reads plain files with the standard library, because
it is imported by the gateway's budgeting path and by the drift tests, and a
Parquet reader there puts a compiled dependency in both. So a corpus published
only as Parquet is converted once, here, into the ignored cache, and the
loader never learns the format existed.

**This is a format conversion and nothing else.** No filtering, grouping,
relabelling or holdout happens here: all of that is in the loader, where the
tests exercise it without `pyarrow`. A decision made in a script that CI never
runs is a decision nobody checks. The only thing dropped is columns the corpus
spec never reads -- measuring_hate_speech carries 131, most of them annotator
demographics -- so the cache holds what the loader uses and no more.

The download is checked against the spec's pinned SHA-256 before it is read.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pathlib
import sys
import tempfile
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from trigon.evals.corpora import CorpusSpec, cache_root, corpus, verify  # noqa: E402


def columns(spec: CorpusSpec) -> list[str]:
    """Every field the loader reads for this spec, in a stable order."""
    wanted = [
        spec.group_field,
        spec.holdout_key,
        spec.abstain_field,
        spec.label_field if spec.label_separator else "",
        *spec.state_fields,
        *spec.score_fields,
        *(c for group in spec.noul_groups.values() for c in group),
        # Kept although the loader does not read it: which rating came from
        # whom is what an annotator-level analysis needs, and it is small.
        "annotator_id",
    ]
    return list(dict.fromkeys(c for c in wanted if c))


def _clean(value):
    # JSON has no NaN; a missing rating is null, which the loader drops.
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def convert(spec: CorpusSpec, *, root: pathlib.Path | None = None) -> dict[str, int]:
    import pyarrow.parquet as pq  # scripts only; never imported by trigon

    if spec.converted_from != "parquet":
        raise SystemExit(f"{spec.name} is not a Parquet corpus; the loader fetches it itself")
    out_dir = (root or cache_root()) / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for split, url in spec.files.items():
        target = out_dir / f"{split}.jsonl.gz"
        print(f"{spec.name}/{split}: downloading {url}", file=sys.stderr)
        with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310
            data = response.read()
        verify(spec, split, data)
        with tempfile.NamedTemporaryFile(suffix=".parquet") as handle:
            handle.write(data)
            handle.flush()
            table = pq.read_table(handle.name)
        keep = [c for c in columns(spec) if c in table.column_names]
        missing = sorted(set(columns(spec)) - set(keep) - {"annotator_id"})
        if missing:
            raise SystemExit(f"{spec.name}/{split} has no column(s) {missing}")
        rows = table.select(keep).to_pylist()
        partial = target.with_suffix(".partial")
        with gzip.open(partial, "wt", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps({k: _clean(v) for k, v in row.items()}) + "\n")
        partial.replace(target)
        written[split] = len(rows)
        print(f"{spec.name}/{split}: {len(rows):,} rows -> {target}", file=sys.stderr)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", help="a corpus with converted_from set")
    args = parser.parse_args(argv)
    convert(corpus(args.corpus))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
