# orchestrator.py
from agents.query_rewriter_agent import QueryRewriterAgent
from agents.retrieval_agent import RetrievalAgent
from agents.reranker_agent import RerankerAgent
from agents.reasoning_agent import ReasoningAgent
from agents.response_agent import ResponseAgent


class Orchestrator:
    def __init__(self):
        self.rewriter  = QueryRewriterAgent()
        self.retriever = RetrievalAgent(top_k=16, vector_k=16, bm25_k=16)
        self.reranker  = RerankerAgent(top_k=6, score_threshold=0.4)
        self.reasoner  = ReasoningAgent()
        self.responder = ResponseAgent()

        # Conversation history for follow-up resolution
        self.history: list[dict] = []

    def run(self, query: str) -> dict:
        print(f"\n[Orchestrator] Original  : {query}")

        # 1 — Rewrite query for better retrieval
        rewritten = self.rewriter.run(query, history=self.history)
        print(f"[Rewriter]     Rewritten : {rewritten}")

        # 2 — Hybrid retrieval: cast a wide net (16 candidates)
        raw_chunks = self.retriever.run(rewritten)
        print(f"[Retrieval]    Got {len(raw_chunks)} chunks (vector + BM25 fused)")

        # 3 — Cross-encoder re-ranking: score each (query, chunk) pair jointly
        reranked_chunks = self.reranker.run(rewritten, raw_chunks)
        print(f"[Reranker]     Kept {len(reranked_chunks)} chunks (score ≥ {self.reranker.score_threshold})")

        # 4 — Reasoning: final relevance filter on the already-reranked set
        filtered_chunks = self.reasoner.run(rewritten, reranked_chunks)
        print(f"[Reasoning]    Kept {len(filtered_chunks)} chunks")

        # 5 — Generate response using original query (feels more natural)
        answer = self.responder.run(query, filtered_chunks)
        print(f"[Response]     Done")

        # Update history
        self.history.append({"role": "user",      "content": query})
        self.history.append({"role": "assistant",  "content": answer})
        if len(self.history) > 10:
            self.history = self.history[-10:]

        return {
            "query":            query,
            "rewritten_query":  rewritten,
            "answer":           answer,
            "sources":          list({c["source"] for c in filtered_chunks}),
            "chunks_retrieved": len(raw_chunks),
            "chunks_reranked":  len(reranked_chunks),
            "chunks_used":      len(filtered_chunks),
            "_chunks":          filtered_chunks,
        }