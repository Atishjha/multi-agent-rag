# main.py
import json
import os
import time
from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from orchestrator import Orchestrator
from logger import log_query, patch_eval, get_all_logs, compute_stats
from ingest import extract_bytes, ingest_documents
from agents.eval_agent import EvalAgent

app = FastAPI(title="Multi-Agent RAG")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

orch      = Orchestrator()
evaluator = EvalAgent()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".ppt", ".txt"}
MAX_FILE_SIZE      = 200 * 1024 * 1024   # 200 MB


class QueryRequest(BaseModel):
    query: str


# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported type '{ext}'.")
    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large. Max 200 MB.")
    if not data:
        raise HTTPException(400, "File is empty.")
    try:
        text = extract_bytes(file.filename, data)
    except Exception as e:
        raise HTTPException(422, f"Could not extract text: {e}")
    if not text.strip():
        raise HTTPException(422, "No text could be extracted.")
    chunks_added = ingest_documents([{"text": text, "source": file.filename}])
    return {
        "filename":        file.filename,
        "size_bytes":      len(data),
        "words_extracted": len(text.split()),
        "chunks_added":    chunks_added,
        "message":         f"'{file.filename}' ingested — {chunks_added} chunks added.",
    }


# ── Non-streaming query ───────────────────────────────────────────────────────

@app.post("/query")
def query(req: QueryRequest):
    t0     = time.perf_counter()
    result = orch.run(req.query)
    elapsed = time.perf_counter() - t0

    entry_id = log_query(
        query            = req.query,
        answer           = result.get("answer", ""),
        sources          = result.get("sources", []),
        chunks_retrieved = result.get("chunks_retrieved", 0),
        chunks_used      = result.get("chunks_used", 0),
        elapsed_seconds  = elapsed,
        rewritten_query  = result.get("rewritten_query", ""),
    )

    evaluator.evaluate_async(
        query           = req.query,
        rewritten_query = result.get("rewritten_query", req.query),
        context_chunks  = result.get("_chunks", []),
        answer          = result.get("answer", ""),
        callback        = lambda scores: patch_eval(entry_id, scores),
    )

    result["elapsed_seconds"] = round(elapsed, 3)
    result.pop("_chunks", None)
    return result


# ── Streaming query ───────────────────────────────────────────────────────────

@app.post("/query/stream")
def query_stream(req: QueryRequest):
    def generate():
        t0              = time.perf_counter()
        full_answer     = []
        filtered_chunks = []
        rewritten       = req.query

        try:
            # Pipeline: rewrite → retrieve → rerank → reason → stream
            rewritten        = orch.rewriter.run(req.query, history=orch.history)
            raw_chunks       = orch.retriever.run(rewritten)
            reranked_chunks  = orch.reranker.run(rewritten, raw_chunks)
            filtered_chunks  = orch.reasoner.run(rewritten, reranked_chunks)
            sources          = list({c["source"] for c in filtered_chunks})

            yield f"data: {json.dumps({'type':'meta','sources':sources,'chunks_retrieved':len(raw_chunks),'chunks_reranked':len(reranked_chunks),'chunks_used':len(filtered_chunks),'rewritten_query':rewritten})}\n\n"

            for token in orch.responder.stream(req.query, filtered_chunks):
                full_answer.append(token)
                yield f"data: {json.dumps({'type':'token','text':token})}\n\n"

            elapsed = round(time.perf_counter() - t0, 3)
            yield f"data: {json.dumps({'type':'done','elapsed_seconds':elapsed})}\n\n"

            answer = "".join(full_answer)

            orch.history.append({"role": "user",     "content": req.query})
            orch.history.append({"role": "assistant", "content": answer})
            if len(orch.history) > 10:
                orch.history = orch.history[-10:]

            entry_id = log_query(
                query            = req.query,
                answer           = answer,
                sources          = sources,
                chunks_retrieved = len(raw_chunks),
                chunks_used      = len(filtered_chunks),
                elapsed_seconds  = elapsed,
                rewritten_query  = rewritten,
            )

            evaluator.evaluate_async(
                query           = req.query,
                rewritten_query = rewritten,
                context_chunks  = filtered_chunks,
                answer          = answer,
                callback        = lambda scores: patch_eval(entry_id, scores),
            )

        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"},
    )


# ── Analytics ─────────────────────────────────────────────────────────────────

@app.get("/stats")
def stats():
    return compute_stats()

@app.get("/logs")
def logs(limit: int = Query(default=20, le=200)):
    all_logs = get_all_logs()
    return {"total": len(all_logs), "logs": list(reversed(all_logs))[:limit]}

@app.delete("/logs")
def clear_logs():
    if os.path.exists("query_logs.json"):
        os.remove("query_logs.json")
    return {"message": "Logs cleared."}

@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")