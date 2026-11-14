"""Question answering over the indexed regulations: hybrid retrieval + a local LLM (Ollama) that must
answer only from the retrieved passages and cite them.

python src/rag.py "¿Qué altura mínima deben tener las barandillas?"
"""
import argparse
import json
import re
import time

import requests

from retrieve import OLLAMA, Retriever, cite

LLM = "qwen2.5:7b"
NOT_FOUND = "No lo encuentro en los documentos indexados."
SYSTEM = f"""Eres un asistente técnico que responde dudas sobre normativa española de edificación
(CTE DB-SI, DB-SUA, DB-HE) y el Reglamento Electrotécnico de Baja Tensión.
Reglas:
- Responde SOLO con la información de los fragmentos numerados que se te dan. No uses conocimiento propio.
- Da la cifra o condición concreta y sus excepciones relevantes, en 1-4 frases, en español.
- Cita los fragmentos que usas con su número entre corchetes, por ejemplo [2].
- Si los fragmentos no contienen la respuesta, responde exactamente: "{NOT_FOUND}"
"""


def clean(text):
    """Drops a trailing "No lo encuentro..." that the model sometimes appends after a valid answer.
    Returns (text, slipped)."""
    rest = re.sub(r"\[\d+\]", "", text.replace(NOT_FOUND, "")).strip(" .\n")
    if NOT_FOUND in text and rest:
        return re.sub(r"\s*(\[\d+\]\s*)?" + re.escape(NOT_FOUND), "", text).strip(), True
    return (NOT_FOUND if NOT_FOUND in text else text.strip()), False


def build_prompt(q, ctx):
    blocks = "\n\n".join(f"[{i}] ({cite(c)})\n{c['text']}" for i, c in enumerate(ctx, 1))
    return f"Fragmentos:\n\n{blocks}\n\nPregunta: {q}\nRespuesta:"


def answer(q, retriever, k=6, model=LLM):
    t0 = time.time()
    ctx = retriever.search(q, k, "hybrid")
    t1 = time.time()
    r = requests.post(f"{OLLAMA}/api/chat", timeout=900, json={
        "model": model, "stream": False, "options": {"temperature": 0, "num_ctx": 8192},
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(q, ctx)}]})
    r.raise_for_status()
    raw = r.json()["message"]["content"].strip()
    text, _ = clean(raw)
    refs = re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text)            # [2] and [2, 3]
    used = sorted({int(n) for g in refs for n in g.split(",") if 1 <= int(n) <= len(ctx)})
    return {"q": q, "answer": raw, "not_found": NOT_FOUND.rstrip(".") in raw,
            "citations": [{"n": n, "cite": cite(ctx[n - 1]), "doc": ctx[n - 1]["doc"],
                           "section": ctx[n - 1]["section"], "text": ctx[n - 1]["text"]} for n in used],
            "retrieved": [cite(c) for c in ctx],
            "s_retrieval": round(t1 - t0, 2), "s_llm": round(time.time() - t1, 1), "model": model}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("q")
    ap.add_argument("--model", default=LLM)
    a = ap.parse_args()
    out = answer(a.q, Retriever(), model=a.model)
    print(out["answer"], "\n")
    for c in out["citations"]:
        print(f"  [{c['n']}] {c['cite']}")
