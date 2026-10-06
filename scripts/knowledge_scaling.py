"""Zero-shot letter accuracy on the benchmark's knowledge slices, by backbone size (MLX, 4-bit).

    python scripts/knowledge_scaling.py <bench>/cases 60 mlx-community/Qwen3-4B-4bit ...

How much of the knowledge gap a bigger backbone buys, measured before paying to
train one: each slice's options are single letters, so one forward pass per
question scores every option. Needs `mlx-lm` (Apple silicon). A seeded sample
per slice, the same one `scripts/decision_bench.py --limit` draws.
"""
import sys, json, ast, random, time, pathlib
sys.path.insert(0, str(pathlib.Path.home() / "trigon/src"))
import mlx.core as mx
from mlx_lm import load
from trigon.backends.lm_score import prompt_for
from trigon.schema import render_state
from trigon.types import ChoiceQuestion

CASES = pathlib.Path(sys.argv[1]); N = int(sys.argv[2]); MODELS = sys.argv[3:]
SLICES = ("mmlu_pro", "medqa_usmle", "medmcqa", "pubmedqa", "scienceqa_text")

def parse(v): return ast.literal_eval(v) if isinstance(v, str) else v

def cases(sl):
    rows = [json.loads(l) for l in open(CASES / f"{sl}.jsonl") if l.strip()]
    rows = [r for r in rows if r["task_type"] == "choice"]
    return random.Random(f"limit:{sl}").sample(rows, min(N, len(rows)))

for name in MODELS:
    model, tok = load(name); started = time.time(); out = {}
    for sl in SLICES:
        right = 0; rows = cases(sl)
        for r in rows:
            q = parse(r["question"]); state = parse(r["state"])
            question = ChoiceQuestion(instructions=q["instructions"], options=[{"name": k, "criteria": str(v)} for k, v in q["criteria"].items()])
            prompt, cands = prompt_for(render_state(state), question)
            ids = tok.encode(prompt)[-4096:]
            first = [tok.encode(c)[0] for c in cands]
            logits = model(mx.array([ids]))[0, -1]
            pick = max(range(len(cands)), key=lambda i: logits[first[i]].item())
            right += int(question.options[pick].name == r["gold"])
        out[sl] = round(right / len(rows), 3)
    mean = round(sum(out.values()) / len(out), 3)
    print(json.dumps({"model": name, "mean": mean, **out, "minutes": round((time.time() - started) / 60, 1)}), flush=True)
    del model; mx.clear_cache()
