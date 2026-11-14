"""Runs the full RAG on every evaluation question (30 in scope + 3 out of scope) with the local LLM and
scores each answer automatically:
  correct_figure : the answer contains every gold figure of the question
  cites_gold     : at least one cited passage is from the gold document + section
  abstained      : for out-of-scope questions, the model answered "no lo encuentro"
Writes results/answers.json. Run on a machine with Ollama (qwen2.5:7b + bge-m3).
"""
import json
import platform
import re
import time
from pathlib import Path

from rag import LLM, answer
from retrieve import Retriever

ROOT = Path(__file__).resolve().parents[1]


def norm(s):
    return s.lower().replace(" ", " ").replace("m²", "m2")


if __name__ == "__main__":
    load = lambda f: [json.loads(l) for l in (ROOT / "eval" / f).open(encoding="utf-8")]
    qs, oos = load("questions.jsonl"), load("out_of_scope.jsonl")
    r, rows, t0 = Retriever(), [], time.time()
    for q in qs + oos:
        a = answer(q["q"], r)
        a["id"] = q["id"]
        if "keys" in q:
            a.update(gold={"doc": q["doc"], "section": q["section"], "answer_key": q["answer_key"]},
                     correct_figure=norm(q["answer_key"]) in norm(a["answer"]),
                     cites_gold=any(c["doc"] == q["doc"] and c["section"] == q["section"] for c in a["citations"]))
        else:
            a.update(out_of_scope=True, abstained=a["not_found"])
        rows.append(a)
        print(f"{q['id']:>3}  {a['s_llm']:>5}s  {a['answer'][:110]!r}")
    ins = [x for x in rows if "gold" in x]
    out_ = [x for x in rows if x.get("out_of_scope")]
    summary = {"model": LLM, "embeddings": "bge-m3", "machine": platform.processor() or platform.machine(),
               "n_in_scope": len(ins), "n_out_of_scope": len(out_),
               "correct_figure": round(sum(x["correct_figure"] for x in ins) / len(ins), 3),
               "cites_gold": round(sum(x["cites_gold"] for x in ins) / len(ins), 3),
               "false_not_found": sum(x["not_found"] for x in ins),
               "abstained_out_of_scope": sum(x["abstained"] for x in out_),
               "s_llm_median": sorted(x["s_llm"] for x in rows)[len(rows) // 2],
               "minutes_total": round((time.time() - t0) / 60, 1)}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "answers.json").write_text(json.dumps({"summary": summary, "answers": rows},
                                                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
