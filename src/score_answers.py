"""Scores results/answers.json (raw model outputs, untouched) into four outcomes per in-scope question:

  correct   : the answer states the gold figure and cites a passage that contains it
  unsupported: states the gold figure but no cited passage contains it (bad citation)
  wrong     : states something else
  abstained : only says "No lo encuentro..." although the answer is in the documents

The model sometimes appends "No lo encuentro..." after a valid answer. rag.clean() removes that tail;
those cases are counted separately as a format slip. Writes results/answers_scored.json.
"""
import json
import re
from pathlib import Path

from rag import NOT_FOUND, clean

ROOT = Path(__file__).resolve().parents[1]


def norm(s):
    s = s.lower().replace("m²", "m2").replace(" ", "").replace(" ", "")
    s = re.sub(r"(?<=\d)\.(?=\d{3})", "", s)
    return re.sub(r"\s+", "", s)


if __name__ == "__main__":
    d = json.loads((ROOT / "results" / "answers.json").read_text(encoding="utf-8"))
    rows = []
    for a in d["answers"]:
        text, slip = clean(a["answer"])
        r = {"id": a["id"], "q": a["q"], "answer": text, "format_slip": slip, "citations": a["citations"],
             "retrieved": a["retrieved"], "s_llm": a["s_llm"]}
        if "gold" not in a:
            r.update(out_of_scope=True, outcome="abstained_ok" if text == NOT_FOUND else "answered_out_of_scope")
        else:
            k = norm(a["gold"]["answer_key"])
            if text == NOT_FOUND:
                outcome = "abstained"
            elif k in norm(text):
                outcome = "correct" if any(k in norm(c["text"]) for c in a["citations"]) else "unsupported"
            else:
                outcome = "wrong"
            r.update(gold=a["gold"], outcome=outcome)
        rows.append(r)
    ins = [r for r in rows if "gold" in r]
    oos = [r for r in rows if r.get("out_of_scope")]
    count = lambda o: sum(r["outcome"] == o for r in ins)
    summary = {**{k: d["summary"][k] for k in ("model", "embeddings", "machine", "s_llm_median", "minutes_total")},
               "n_in_scope": len(ins), "correct": count("correct"), "unsupported": count("unsupported"),
               "wrong": count("wrong"), "abstained": count("abstained"),
               "oos_abstained": sum(r["outcome"] == "abstained_ok" for r in oos), "n_out_of_scope": len(oos),
               "format_slips": sum(r["format_slip"] for r in rows)}
    (ROOT / "results" / "answers_scored.json").write_text(
        json.dumps({"summary": summary, "answers": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
