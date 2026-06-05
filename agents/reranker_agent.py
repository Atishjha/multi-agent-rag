# agents/reranker_agent.py
"""
Cross-encoder-style re-ranker using an LLM as the scoring model.

Why LLM re-ranking works like a cross-encoder:
- A bi-encoder (standard RAG) embeds query and chunk SEPARATELY, then compares
- A cross-encoder sees (query + chunk) TOGETHER, giving far richer relevance signal
- This agent sends each (query, chunk) pair to the LLM jointly — same principle
- Trade-off vs local cross-encoder: slower but no PyTorch dependency, often more accurate
  on domain-specific text since it uses the full LLM reasoning capacity
"""
import json
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a precise relevance scoring engine for a retrieval-augmented generation system.

Given a search query and a list of document chunks, score each chunk on how useful it would be for answering the query.

Scoring criteria (score 0.0 to 1.0):
- 1.0  : chunk directly and completely answers the query
- 0.8  : chunk contains highly relevant information that substantially helps answer the query
- 0.6  : chunk is related and provides some useful context
- 0.4  : chunk is loosely related but unlikely to improve the answer
- 0.2  : chunk is from the same domain but not relevant to this specific query
- 0.0  : chunk is irrelevant or would confuse the answer

Return ONLY a JSON array with one object per chunk, in the same order as input:
[
  {"chunk_index": 1, "score": 0.9, "reason": "one sentence"},
  {"chunk_index": 2, "score": 0.3, "reason": "one sentence"},
  ...
]

No markdown, no explanation outside the JSON."""


class RerankerAgent:
    def __init__(self, top_k: int = 5, score_threshold: float = 0.4):
        """
        top_k           — max chunks to return after re-ranking
        score_threshold — drop chunks below this score entirely
        """
        self.top_k           = top_k
        self.score_threshold = score_threshold

    def run(self, query: str, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return []

        # Build compact chunk summaries to stay within token limits
        formatted = "\n\n".join(
            f"[Chunk {i+1}] source: {c.get('source','?')} | "
            f"found_by: {c.get('found_by','?')} | "
            f"rrf_score: {c.get('rrf_score', c.get('similarity', 'n/a'))}\n"
            f"{c['text'][:600]}"          # cap each chunk at 600 chars
            for i, c in enumerate(chunks)
        )

        prompt = f"Query: {query}\n\nChunks to score:\n{formatted}"

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=600,
                temperature=0.0,        # fully deterministic scoring
            )

            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            scores = json.loads(raw)

            # Attach reranker score to each chunk
            scored = []
            for item in scores:
                idx   = item["chunk_index"] - 1
                score = float(item["score"])
                if 0 <= idx < len(chunks) and score >= self.score_threshold:
                    chunk = chunks[idx].copy()
                    chunk["rerank_score"]  = round(score, 4)
                    chunk["rerank_reason"] = item.get("reason", "")
                    scored.append(chunk)

            # Sort by reranker score descending, take top_k
            scored.sort(key=lambda x: x["rerank_score"], reverse=True)
            return scored[:self.top_k]

        except Exception as e:
            # Fail gracefully — return original chunks unmodified
            print(f"[Reranker] Error: {e} — falling back to original order")
            return chunks[:self.top_k]