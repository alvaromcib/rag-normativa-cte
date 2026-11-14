# rag-normativa-cte — local assistant for Spanish building regulations

Ask questions in Spanish about the **Código Técnico de la Edificación** (DB-SI fire safety, DB-SUA safety of use
and accessibility, DB-HE energy saving) and the **Reglamento Electrotécnico de Baja Tensión** (RD 842/2002 + ITC-BT),
and get a short answer with the exact document, section and page it comes from. Everything runs locally:
no document or question leaves the machine.

> Work in progress — evaluation results and dashboard coming next.

## Pipeline

1. **Ingest** (`src/ingest.py`): official PDFs → 1,743 passages that keep their structure
   (document · section/ITC/article · sub-section · page). Running headers, page numbers and table-of-contents
   lines are removed.
2. **Retrieve** (`src/retrieve.py`): hybrid search — BM25 for literal figures and codes (`ITC-BT-25`, `0,90 m`)
   plus multilingual embeddings (`bge-m3` via Ollama) for meaning, fused with Reciprocal Rank Fusion.
3. **Answer** (`src/rag.py`): `qwen2.5:7b` via Ollama answers only from the retrieved passages, cites them,
   and says *"No lo encuentro en los documentos indexados"* when they don't contain the answer.
4. **Evaluate** (`eval/`): 30 hand-written questions with the section and figure that answer them, plus
   3 questions about regulations that are *not* indexed, to check the assistant abstains.

## Run

```bash
pip install -r requirements.txt
ollama pull bge-m3 && ollama pull qwen2.5:7b
python src/retrieve.py --build          # embedding index
python src/rag.py "¿Qué altura mínima deben tener las barandillas?"
python src/eval_retrieval.py            # bm25 / dense / hybrid
python src/eval_answers.py              # full RAG on the 33 questions
```

On Windows, `scripts/run_local_windows.bat` does all of it. The PDFs are not in the repo:
`python src/get_docs.py` downloads them from codigotecnico.org and boe.es (only needed to re-run the ingestion).

Sources: CTE DB-SI, DB-SUA, DB-HE (codigotecnico.org) and RD 842/2002 consolidated text (BOE, last update 03/09/2025).
This is a demo with public documents; it does not replace reading the regulation or a professional's judgement.
