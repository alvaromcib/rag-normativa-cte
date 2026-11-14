"""Extract the official PDFs and split them into chunks that keep their place in the document.

Each chunk carries: document (DB-SI, DB-SUA, DB-HE, REBT), section (e.g. "SI 3", "ITC-BT-19",
"Artículo 16"), numbered sub-section (e.g. "4.2 Cálculo") and page, so every answer can cite them.

Usage: python src/ingest.py            -> data/chunks.jsonl
"""
import json
import re
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "pdf"
DOCS = {  # file -> (short name, full title)
    "CTE_DB-SI.pdf": ("DB-SI", "CTE DB-SI Seguridad en caso de incendio"),
    "CTE_DB-SUA.pdf": ("DB-SUA", "CTE DB-SUA Seguridad de utilización y accesibilidad"),
    "CTE_DB-HE.pdf": ("DB-HE", "CTE DB-HE Ahorro de energía"),
    "REBT_RD842-2002_consolidado.pdf": ("REBT", "Reglamento electrotécnico para baja tensión (RD 842/2002) e ITC-BT"),
}
MAX_CHARS, OVERLAP = 1100, 200

NOISE = [re.compile(p) for p in (
    r"^Documento Básico (SI|SUA|HE)\b.*$",            # CTE running header
    r"^BOLETÍN OFICIAL DEL ESTADO$", r"^LEGISLACIÓN CONSOLIDADA$", r"^Página \d+$",   # BOE footer
    r"^\d{1,3}$", r"^\s*$",
)]
TOC = re.compile(r"(\. ?){6,}|\.{6,}")              # dot leaders: table of contents
CTE_SEC = re.compile(r"^Sección (SI|SUA|HE) ?(\d)\s*$")
CTE_SEC_INLINE = re.compile(r"^Sección (SI|SUA|HE) ?(\d)\s+[A-ZÁÉÍÓÚ]")
ITC = re.compile(r"^(ITC-BT-\d{2})\.?\s*$")
ART = re.compile(r"^(Artículo \d+)\.\s+\S")
ANEJO = re.compile(r"^Anejo (?:(?:SI|SUA|HE) )?([A-H])(?:[.:]?(?:\s+[A-ZÁÉÍÓÚ][^\n]{0,50})?)?\s*$")
SUB = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s{1,3}([A-ZÁÉÍÓÚ][^\n]{2,90})$")


def lines_with_pages(path: Path):
    for pno, page in enumerate(fitz.open(path), 1):
        for ln in page.get_text().splitlines():
            ln = ln.rstrip()
            if any(p.match(ln) for p in NOISE) or TOC.search(ln):
                continue
            yield pno, ln


def chunks_for(fname: str):
    short, title = DOCS[fname]
    is_rebt = short == "REBT"
    section = "Real Decreto" if is_rebt else "Introducción"
    sub, buf, start_page, out = "", [], None, []

    def flush():
        nonlocal buf, start_page
        text = re.sub(r"[ \t]+", " ", " ".join(buf)).strip()
        text = re.sub(r"(\w)- (\w)", r"\1\2", text)          # re-join hyphenated words
        if len(text) > 80:
            for i in range(0, len(text), MAX_CHARS - OVERLAP):
                piece = text[i:i + MAX_CHARS]
                if len(piece) > 80:
                    out.append({"doc": short, "doc_title": title, "section": section, "sub": sub,
                                "page": start_page, "text": piece})
                if i + MAX_CHARS >= len(text):
                    break
        buf, start_page = [], None

    prev = ""
    for pno, ln in lines_with_pages(PDF / fname):
        if is_rebt and pno < 5:                    # cover and table of contents
            continue
        s = ln.strip()
        new_sec = None
        if is_rebt:
            m = ITC.match(s) or ART.match(s)
            if m and not (ART.match(s) and section.startswith("ITC")):
                new_sec = m.group(1)
                # ITCs appear in increasing order: a lower number is a wrapped cross-reference, not a heading
                if new_sec.startswith("ITC") and section.startswith("ITC") and new_sec <= section:
                    new_sec = None
            if s.startswith("REGLAMENTO ELECTROTÉCNICO PARA BAJA TENSIÓN") and section == "Real Decreto" and pno > 5:
                new_sec = "Reglamento"
        else:
            m = CTE_SEC.match(s) or CTE_SEC_INLINE.match(s)
            a = ANEJO.match(s) if len(s) < 60 else None
            if m and pno > 3:
                new_sec = f"{m.group(1)} {m.group(2)}"
            elif a and pno > 5:
                new_sec = f"Anejo {a.group(1)}"
        if new_sec and new_sec != section:
            flush(); section, sub = new_sec, ""
        else:
            m = SUB.match(s)
            if m and not prev.endswith(",") and float(m.group(1).split(".")[0]) < 30:
                flush(); sub = f"{m.group(1)} {m.group(2).strip()}"
        if start_page is None:
            start_page = pno
        buf.append(s)
        prev = s
    flush()
    return out


def main():
    out = ROOT / "data" / "chunks.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for fname in DOCS:
            cs = chunks_for(fname)
            for c in cs:
                c["id"] = n; n += 1
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
            secs = sorted({c["section"] for c in cs})
            print(f"{fname}: {len(cs)} chunks, {len(secs)} sections")
    print("total", n, "->", out)


if __name__ == "__main__":
    main()
