# bm25_store.py
"""
Persistent BM25 index that stays in sync with ChromaDB.
Rebuilt from the vector store on startup, updated on every ingest.
"""
import json
import os
import pickle
import re
from pathlib import Path

BM25_CACHE = "bm25_cache.pkl"   # serialised index + corpus
_index = None                    # module-level singleton


# ── Text normalisation ────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


# ── Index wrapper ─────────────────────────────────────────────────────────────

class BM25Index:
    def __init__(self, docs: list[str], metadatas: list[dict], ids: list[str]):
        from rank_bm25 import BM25Okapi
        self.ids       = ids
        self.docs      = docs
        self.metadatas = metadatas
        tokenised      = [_tokenize(d) for d in docs]
        self.bm25      = BM25Okapi(tokenised)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked[:top_k]:
            if score <= 0:
                break
            results.append({
                "id":         self.ids[idx],
                "text":       self.docs[idx],
                "source":     self.metadatas[idx].get("source", "unknown"),
                "bm25_score": round(float(score), 4),
            })
        return results


# ── Build / load ──────────────────────────────────────────────────────────────

def _build_from_chroma() -> BM25Index | None:
    """Pull every document from ChromaDB and build a fresh BM25 index."""
    try:
        from vector_store import get_collection
        col = get_collection()
        count = col.count()
        if count == 0:
            return None
        # Fetch in batches of 5000 to avoid memory spikes
        all_docs, all_metas, all_ids = [], [], []
        batch = 5000
        for offset in range(0, count, batch):
            res = col.get(
                limit=batch,
                offset=offset,
                include=["documents", "metadatas"],
            )
            all_docs.extend(res["documents"])
            all_metas.extend(res["metadatas"])
            all_ids.extend(res["ids"])
        return BM25Index(all_docs, all_metas, all_ids)
    except Exception as e:
        print(f"[BM25] Could not build index from ChromaDB: {e}")
        return None


def _save(index: BM25Index):
    with open(BM25_CACHE, "wb") as f:
        pickle.dump(index, f)


def _load_cache() -> BM25Index | None:
    if not Path(BM25_CACHE).exists():
        return None
    try:
        with open(BM25_CACHE, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def get_index() -> BM25Index | None:
    global _index
    if _index is not None:
        return _index
    # Try cache first (fast), fall back to rebuild from Chroma
    _index = _load_cache()
    if _index is None:
        print("[BM25] No cache found — building index from ChromaDB...")
        _index = _build_from_chroma()
        if _index:
            _save(_index)
            print(f"[BM25] Built index with {len(_index.docs)} documents.")
    return _index


def add_documents(docs: list[str], metadatas: list[dict], ids: list[str]):
    """Extend the live index with newly ingested chunks (no full rebuild)."""
    global _index
    if _index is None:
        _index = get_index()

    if _index is None:
        # First documents ever — build fresh
        _index = BM25Index(docs, metadatas, ids)
    else:
        # Append and rebuild (BM25Okapi has no incremental add)
        all_docs   = _index.docs   + docs
        all_metas  = _index.metadatas + metadatas
        all_ids    = _index.ids    + ids
        _index     = BM25Index(all_docs, all_metas, all_ids)

    _save(_index)
    print(f"[BM25] Index updated — {len(_index.docs)} total documents.")