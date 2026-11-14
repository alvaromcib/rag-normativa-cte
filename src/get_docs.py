"""Downloads the four official PDFs into data/pdf/ (they are not committed to the repo).
Only needed to rebuild data/chunks.jsonl with src/ingest.py."""
from pathlib import Path

import requests

PDF = Path(__file__).resolve().parents[1] / "data" / "pdf"
SOURCES = {
    "CTE_DB-SI.pdf": "https://www.codigotecnico.org/pdf/Documentos/SI/DBSI.pdf",
    "CTE_DB-SUA.pdf": "https://www.codigotecnico.org/pdf/Documentos/SUA/DBSUA.pdf",
    "CTE_DB-HE.pdf": "https://www.codigotecnico.org/pdf/Documentos/HE/DBHE.pdf",
    "REBT_RD842-2002_consolidado.pdf": "https://www.boe.es/buscar/pdf/2002/BOE-A-2002-18099-consolidado.pdf",
}

if __name__ == "__main__":
    PDF.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        f = PDF / name
        if f.exists():
            print("ok  ", name); continue
        r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        f.write_bytes(r.content)
        print("get ", name, f"{len(r.content) / 1e6:.1f} MB")
