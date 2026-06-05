# agents/response_agent.py
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only the provided document context.
Always cite which source your answer comes from.
If the context doesn't contain enough information, say so clearly."""

def build_prompt(query: str, context_chunks: list[dict]) -> str:
    if not context_chunks:
        return query
    context = "\n\n---\n\n".join(
        f"Source: {c['source']}\n{c['text']}"
        for c in context_chunks
    )
    return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


class ResponseAgent:
    def run(self, query: str, context_chunks: list[dict]) -> str:
        if not context_chunks:
            return "I couldn't find relevant information to answer your question."

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(query, context_chunks)},
            ],
            max_tokens=1024,
        )
        return response.choices[0].message.content

    def stream(self, query: str, context_chunks: list[dict]):
        """Yields text tokens one by one."""
        if not context_chunks:
            yield "I couldn't find relevant information to answer your question."
            return

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(query, context_chunks)},
            ],
            max_tokens=1024,
            stream=True,         # ← this is the correct Groq streaming flag
        )

        for chunk in completion:
            token = chunk.choices[0].delta.content
            if token:
                yield token