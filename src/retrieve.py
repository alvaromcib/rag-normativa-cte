"""Hybrid retrieval over the regulation chunks: BM25 (keywords, numbers, ITC codes) + multilingual
embeddings (meaning), fused with Reciprocal Rank Fusion. Everything runs locally on CPU.

python src/retrieve.py --build            builds data/index/ (embeddings)
python src/retrieve.py "pregunta"         prints the top chunks with their citation
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "chunks.jsonl"
INDEX = ROOT / "data" / "index"
EMB_MODEL = "bge-m3"          # multilingual embedding model served by Ollama (local, CPU is fine)
OLLAMA = "http://localhost:11434"
STOP = set("""a al algo ante antes como con contra cual cuales cuando de del desde donde e el ella ellas ellos en entre
era es esta este esto estos hasta la las le les lo los mas me mi mientras muy no nos o otra otro para pero poco por
que quien se sea segun ser si sin sobre su sus tambien tan te tiene tu un una uno unos y ya debe deben cual cuanto
cuantos cuanta cuantas hay""".split())


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def tokens(s: str):
    return [t for t in re.findall(r"[a-z0-9]+(?:[.,][0-9]+)?", norm(s)) if t not in STOP]


def cite(c: dict) -> str:
    sub = f" · {c['sub']}" if c.get("sub") else ""
    return f"{c['doc']} · {c['section']}{sub} · p. {c['page']}"


class Retriever:
    def __init__(self, dense=True):
        self.chunks = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
        self.bm25 = BM25Okapi([tokens(self._ctx(c)) for c in self.chunks])
        self.model, self.emb = None, None
        if dense:
            f = INDEX / "emb.npy"
            self.emb = np.load(f).astype(np.float32) if f.exists() else None   # build with --build

    @staticmethod
    def _ctx(c):        # what gets indexed: document, section and sub-section titles + text
        return f"{c['doc_title']}. {c['section']}. {c.get('sub', '')}. {c['text']}"

    @staticmethod
    def embed(texts, batch=16):
        import requests
        out = []
        for i in range(0, len(texts), batch):
            r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMB_MODEL, "input": texts[i:i + batch]},
                              timeout=600)
            r.raise_for_status()
            out += r.json()["embeddings"]
            print(f"\r  embeddings {min(i + batch, len(texts))}/{len(texts)}", end="", flush=True)
        v = np.array(out, dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def build(self):
        INDEX.mkdir(parents=True, exist_ok=True)
        emb = self.embed([self._ctx(c) for c in self.chunks])
        np.save(INDEX / "emb.npy", emb.astype(np.float16))        # ~3.5 MB, committed with the repo
        print()
        return emb

    def rank_bm25(self, q):
        return np.argsort(-self.bm25.get_scores(tokens(q)))

    def rank_dense(self, q):
        if self.emb is None:
            raise SystemExit("No embedding index: run python src/retrieve.py --build")
        v = self.embed([q])[0]
        return np.argsort(-(self.emb @ v))

    def search(self, q, k=5, mode="hybrid", docs=None, rrf_k=60):
        if mode == "bm25":
            order = self.rank_bm25(q)
        elif mode == "dense":
            order = self.rank_dense(q)
        else:                                        # Reciprocal Rank Fusion of both rankings
            score = np.zeros(len(self.chunks))
            for order_ in (self.rank_bm25(q), self.rank_dense(q)):
                score[order_] += 1.0 / (rrf_k + np.arange(1, len(order_) + 1))
            order = np.argsort(-score)
        out = []
        for i in order:
            c = self.chunks[int(i)]
            if docs and c["doc"] not in docs:
                continue
            out.append(c)
            if len(out) == k:
                break
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("q", nargs="?")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "bm25", "dense"])
    a = ap.parse_args()
    r = Retriever(dense=a.mode != "bm25" and not a.build)
    if a.build:
        r.emb = r.build(); print("index ->", INDEX)
    if a.q:
        for c in r.search(a.q, a.k, a.mode):
            print(f"[{cite(c)}]\n  {c['text'][:300]}\n")
