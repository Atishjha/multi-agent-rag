# agents/query_rewriter_agent.py
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a search query optimization expert.

Your job is to rewrite a user's raw query into a detailed, search-optimized version that will retrieve better results from a vector database of technical documents.

Rules:
- Expand abbreviations and vague terms into full concepts
- Add relevant technical synonyms and related terms
- If the query is a follow-up (e.g. "explain more", "give an example"), expand it into a standalone question
- Keep the rewritten query as ONE clear sentence or question — no bullet points
- Never answer the question — only rewrite it
- If the query is already detailed and specific, return it unchanged

Reply with ONLY the rewritten query. No explanation, no prefix like "Rewritten:", just the query itself."""


class QueryRewriterAgent:
    def run(self, query: str, history: list[dict] | None = None) -> str:
        """
        Rewrites the query into a search-optimized version.
        Optionally accepts recent conversation history for follow-up resolution.
        history format: [{"role": "user"|"assistant", "content": "..."}]
        """
        context = ""
        if history:
            last_turns = history[-4:]  # last 2 exchanges
            context = "\n".join(
                f"{t['role'].capitalize()}: {t['content']}" for t in last_turns
            )
            context = f"\n\nRecent conversation context:\n{context}\n"

        user_msg = f"{context}\nOriginal query: {query}\n\nRewritten query:"

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=120,
            temperature=0.2,   # low temp = consistent, focused rewrites
        )

        rewritten = response.choices[0].message.content.strip()

        # Safety: if the model returns something absurdly long, fall back
        if len(rewritten) > 400:
            return query

        return rewritten