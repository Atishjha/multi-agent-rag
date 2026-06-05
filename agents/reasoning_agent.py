# agents/reasoning_agent.py
import os, json
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class ReasoningAgent:
    def run(self, query: str, chunks: list[dict]) -> list[dict]:
        formatted = "\n\n".join(
            f"[Chunk {i+1}] (score: {c.get('similarity') or c.get('bm25_score') or c.get('rrf_score', 'n/a')}) "
            f"[found_by: {c.get('found_by', 'unknown')}]\n{c['text']}"
            for i, c in enumerate(chunks)
        )

        prompt = f"""You are a relevance reasoning agent.

Given the user query and retrieved document chunks, return ONLY a JSON array:
[
  {{"chunk_index": 1, "relevance_score": 8, "reason": "..."}},
  ...
]

Score each chunk 0-10 for relevance to the query. No other text, just the JSON array.

User query: {query}

Retrieved chunks:
{formatted}"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )

        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()

        scores = json.loads(text)

        filtered = sorted(
            [s for s in scores if s["relevance_score"] >= 6],
            key=lambda x: x["relevance_score"],
            reverse=True,
        )

        return [chunks[s["chunk_index"] - 1] for s in filtered]