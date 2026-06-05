# agents/retrieval_agent.py
from vector_store import get_collection
import bm25_store


def _reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    id_key: str = "id",
    k: int = 60,
) -> list[dict]:
    scores:  dict[str, float] = {}
    doc_map: dict[str, dict]  = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            doc_id = doc[id_key]
            scores[doc_id]  = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            doc_map[doc_id] = doc
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    for doc_id, rrf_score in fused:
        entry = doc_map[doc_id].copy()
        entry["rrf_score"] = round(rrf_score, 6)
        result.append(entry)
    return result


class RetrievalAgent:
    def __init__(self, top_k: int = 8, vector_k: int = 12, bm25_k: int = 12):
        self.top_k      = top_k
        self.vector_k   = vector_k
        self.bm25_k     = bm25_k
        self.collection = get_collection()

    def _vector_search(self, query: str) -> list[dict]:
        results = self.collection.query(
            query_texts=[query],
            n_results=self.vector_k,
            include=["documents", "metadatas", "distances"],  # ids come back automatically
        )
        chunks = []
        # ids are always returned by ChromaDB even without being in include
        ids = results.get("ids", [[]])[0]
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            chunks.append({
                "id":         ids[i] if i < len(ids) else f"vec_{i}",
                "text":       doc,
                "source":     meta.get("source", "unknown"),
                "similarity": round(1 - dist, 4),
            })
        return chunks

    def _bm25_search(self, query: str) -> list[dict]:
        index = bm25_store.get_index()
        if index is None:
            return []
        return index.search(query, top_k=self.bm25_k)

    def run(self, query: str) -> list[dict]:
        vector_results = self._vector_search(query)
        bm25_results   = self._bm25_search(query)

        print(f"[Retrieval]    Vector: {len(vector_results)} | BM25: {len(bm25_results)}")

        fused = _reciprocal_rank_fusion([vector_results, bm25_results])
        final = fused[:self.top_k]

        vector_ids = {r["id"] for r in vector_results}
        bm25_ids   = {r["id"] for r in bm25_results}
        for chunk in final:
            in_v = chunk["id"] in vector_ids
            in_b = chunk["id"] in bm25_ids
            chunk["found_by"] = (
                "both"   if in_v and in_b else
                "vector" if in_v else
                "bm25"
            )

        return final