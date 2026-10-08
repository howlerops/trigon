"""Zero-shot letter accuracy on the benchmark's knowledge slices, by backbone size (MLX, 4-bit).

    python scripts/knowledge_scaling.py <bench>/cases 60 mlx-community/Qwen3-4B-4bit ...

How much of the knowledge gap a bigger backbone buys, measured before paying to
train one: each slice's options are single letters, so one forward pass per
question scores every option. Needs `mlx-lm` (Apple silicon). A seeded sample
per slice, the same one `scripts/decision_bench.py --limit` draws.
"""

from __future__ import annotations

import ast
import json
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from trigon.backends.lm_score import prompt_for  # noqa: E402
from trigon.schema import render_state  # noqa: E402
from trigon.types import ChoiceQuestion  # noqa: E402

SLICES = ("mmlu_pro", "medqa_usmle", "medmcqa", "pubmedqa", "scienceqa_text")


def _parse(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


def _cases(root: pathlib.Path, name: str, n: int) -> list[dict]:
    rows = [json.loads(line) for line in open(root / f"{name}.jsonl") if line.strip()]
    rows = [r for r in rows if r["task_type"] == "choice"]
    return random.Random(f"limit:{name}").sample(rows, min(n, len(rows)))


def accuracy(model, tokenizer, rows: list[dict]) -> float:
    import mlx.core as mx

    right = 0
    for row in rows:
        q = _parse(row["question"])
        options = [{"name": k, "criteria": str(v)} for k, v in q["criteria"].items()]
        question = ChoiceQuestion(instructions=q["instructions"], options=options)
        prompt, candidates = prompt_for(render_state(_parse(row["state"])), question)
        ids = tokenizer.encode(prompt)[-4096:]
        first = [tokenizer.encode(c)[0] for c in candidates]
        logits = model(mx.array([ids]))[0, -1]
        pick = max(range(len(candidates)), key=lambda i: logits[first[i]].item())
        right += int(question.options[pick].name == row["gold"])
    return right / len(rows)


def main() -> int:
    import mlx.core as mx
    from mlx_lm import load

    root, n, models = pathlib.Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
    for name in models:
        model, tokenizer = load(name)
        started = time.time()
        out = {s: round(accuracy(model, tokenizer, _cases(root, s, n)), 3) for s in SLICES}
        mean = round(sum(out.values()) / len(out), 3)
        minutes = round((time.time() - started) / 60, 1)
        print(json.dumps({"model": name, "mean": mean, **out, "minutes": minutes}), flush=True)
        del model
        mx.clear_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
