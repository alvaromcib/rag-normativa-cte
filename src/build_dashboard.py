"""Collects the evaluation outputs into results/results.json and renders docs/ (ES) and docs/en/ (EN)."""
import collections
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"

# Why each miss happened, read from the retrieved passages (see results/answers.json).
WHY = {
    1: {"es": "Mezcló dos casos del mismo párrafo: tomó los 50 m de plantas con más de una salida en vez de los 25 m generales.",
        "en": "Mixed two cases from the same paragraph: took 50 m (floors with more than one exit) instead of the general 25 m."},
    2: {"es": "La cifra está en una tabla cuyas columnas se separan al extraer el PDF; el modelo tomó otra fila.",
        "en": "The figure sits in a table whose columns come apart when the PDF is extracted; the model took another row."},
    11: {"es": "Dio el 16 % de las rampas de vehículos (una excepción) en vez del 12 % general.",
         "en": "Gave 16 % (vehicle ramps, an exception) instead of the general 12 %."},
    24: {"es": "Tomó el umbral de otra fila de la misma tabla (100 kW, edificios de viviendas) en vez de 20 kW para industria.",
         "en": "Took the threshold from another row of the same table (100 kW, residential buildings) instead of 20 kW for industry."},
    7: {"es": "El pasaje correcto se recuperó, pero es una tabla aplanada y el modelo no supo leerla.",
        "en": "The right passage was retrieved, but it is a flattened table and the model could not read it."},
    17: {"es": "El pasaje correcto se recuperó, pero el modelo no vio la respuesta y se abstuvo.",
         "en": "The right passage was retrieved, but the model did not see the answer and abstained."},
    9: {"es": "Fallo de búsqueda: «barandilla» trajo pasajes del REBT y no del DB-SUA.",
        "en": "Retrieval miss: “barandilla” brought REBT passages instead of DB-SUA."},
    10: {"es": "Respuesta correcta, pero citó los fragmentos como [2, 3] y el evaluador de esta ejecución solo leía [n]; corregido en el código.",
         "en": "Correct answer, but it cited passages as [2, 3] and this run's scorer only read [n]; fixed in the code."},
    20: {"es": "Cifra correcta, pero citó un pasaje de la ITC-BT-52 en vez de la tabla de la ITC-BT-25.",
         "en": "Correct figure, but it cited an ITC-BT-52 passage instead of the ITC-BT-25 table."},
}

if __name__ == "__main__":
    chunks = [json.loads(l) for l in (ROOT / "data" / "chunks.jsonl").open(encoding="utf-8")]
    ret = json.loads((R / "retrieval_eval.json").read_text(encoding="utf-8"))
    ans = json.loads((R / "answers_scored.json").read_text(encoding="utf-8"))
    qs = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "eval" / "questions.jsonl").open(encoding="utf-8")}
    per_doc = collections.Counter(c["doc"] for c in chunks)
    pages = collections.defaultdict(int)
    for c in chunks:
        pages[c["doc"]] = max(pages[c["doc"]], c["page"])
    rank = {m: {r["id"]: r["first_passage_rank"] for r in v["per_question"]} for m, v in ret["modes"].items()}
    answers = []
    for a in ans["answers"]:
        q = qs.get(a["id"], {})
        answers.append({"id": a["id"], "q": a["q"], "answer": a["answer"], "outcome": a["outcome"],
                        "gold": a.get("gold"), "why": WHY.get(a["id"]), "s": a["s_llm"],
                        "citations": [{"n": c["n"], "cite": c["cite"], "text": c["text"]} for c in a["citations"]],
                        "rank": {m: rank[m].get(a["id"]) for m in rank} if q else None})
    out = {"corpus": {"chunks": len(chunks), "docs": [{"doc": d, "chunks": per_doc[d], "pages": pages[d],
                                                        "sections": len({c["section"] for c in chunks if c["doc"] == d})}
                                                       for d in ("DB-SI", "DB-SUA", "DB-HE", "REBT")]},
           "retrieval": {m: v["metrics"] for m, v in ret["modes"].items()},
           "summary": ans["summary"], "answers": answers}
    (R / "results.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("results.json", (R / "results.json").stat().st_size // 1024, "KB")
    subprocess.run([sys.executable, str(ROOT / "shared" / "build.py"), str(ROOT)], check=True)
