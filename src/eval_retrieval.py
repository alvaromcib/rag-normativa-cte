"""Retrieval evaluation on eval/questions.jsonl (30 questions written by hand, each with the section
of the regulation that answers it and the literal figures that must appear in the passage).

hit@k (section) : a chunk from the right document + section is among the top k
hit@k (passage) : a chunk containing every gold figure is among the top k
MRR             : mean reciprocal rank of the first passage hit
python src/eval_retrieval.py --modes bm25 dense hybrid
"""
import argparse
import json
import time
from pathlib import Path

from retrieve import Retriever

ROOT = Path(__file__).resolve().parents[1]


def passage_ok(c, q):
    return c["doc"] == q["doc"] and all(k.lower() in c["text"].lower() for k in q["keys"])


def run(r, qs, mode, K=10):
    rows, t0 = [], time.time()
    for q in qs:
        res = r.search(q["q"], K, mode)
        sec = [c["doc"] == q["doc"] and c["section"] == q["section"] for c in res]
        psg = [passage_ok(c, q) for c in res]
        first = next((i + 1 for i, ok in enumerate(psg) if ok), None)
        rows.append({"id": q["id"], "first_passage_rank": first, "top": [f"{c['doc']} {c['section']}" for c in res[:3]],
                     "sec@1": sec[0], "sec@5": any(sec[:5]), "psg@1": psg[0], "psg@5": any(psg[:5]),
                     "psg@10": any(psg)})
    n = len(rows)
    m = {k: round(sum(x[k] for x in rows) / n, 3) for k in ("sec@1", "sec@5", "psg@1", "psg@5", "psg@10")}
    m["mrr"] = round(sum(1 / x["first_passage_rank"] for x in rows if x["first_passage_rank"]) / n, 3)
    m["ms_per_query"] = round(1000 * (time.time() - t0) / n, 1)
    return m, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["bm25", "dense", "hybrid"])
    a = ap.parse_args()
    qs = [json.loads(l) for l in (ROOT / "eval" / "questions.jsonl").open(encoding="utf-8")]
    r = Retriever(dense=a.modes != ["bm25"])
    out = {"n_questions": len(qs), "n_chunks": len(r.chunks), "modes": {}}
    for mode in a.modes:
        m, rows = run(r, qs, mode)
        out["modes"][mode] = {"metrics": m, "per_question": rows}
        print(mode.ljust(7), m)
    (ROOT / "results").mkdir(exist_ok=True)
    f = ROOT / "results" / ("retrieval_eval.json" if len(a.modes) > 1 else f"retrieval_eval_{a.modes[0]}.json")
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", f)
