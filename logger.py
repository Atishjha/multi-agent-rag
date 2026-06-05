# logger.py
import json
import os
import threading
from datetime import datetime, timezone

LOG_FILE = "query_logs.json"
_lock = threading.Lock()

def _load() -> list:
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def _save(logs: list):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

def log_query(
    query: str,
    answer: str,
    sources: list[str],
    chunks_retrieved: int,
    chunks_used: int,
    elapsed_seconds: float,
):
    entry = {
        "id": _next_id(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "answer_length": len(answer),
        "sources": sources,
        "chunks_retrieved": chunks_retrieved,
        "chunks_used": chunks_used,
        "chunks_filtered_out": chunks_retrieved - chunks_used,
        "filter_rate": round((chunks_retrieved - chunks_used) / max(chunks_retrieved, 1) * 100, 1),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }
    with _lock:
        logs = _load()
        logs.append(entry)
        _save(logs)
    return entry

def _next_id() -> int:
    logs = _load()
    return (logs[-1]["id"] + 1) if logs else 1

def get_all_logs() -> list:
    with _lock:
        return _load()

def compute_stats() -> dict:
    with _lock:
        logs = _load()

    if not logs:
        return {"total_queries": 0, "message": "No queries logged yet."}

    total = len(logs)
    times = [l["elapsed_seconds"] for l in logs]
    retrieved = [l["chunks_retrieved"] for l in logs]
    used = [l["chunks_used"] for l in logs]
    filter_rates = [l["filter_rate"] for l in logs]

    source_counts: dict[str, int] = {}
    for l in logs:
        for s in l.get("sources", []):
            source_counts[s] = source_counts.get(s, 0) + 1

    top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)

    recent = sorted(logs, key=lambda x: x["timestamp"], reverse=True)[:5]

    return {
        "total_queries": total,
        "response_time": {
            "avg_seconds": round(sum(times) / total, 3),
            "min_seconds": round(min(times), 3),
            "max_seconds": round(max(times), 3),
            "p90_seconds": round(_percentile(times, 90), 3),
        },
        "retrieval": {
            "avg_chunks_retrieved": round(sum(retrieved) / total, 1),
            "avg_chunks_used": round(sum(used) / total, 1),
            "avg_filter_rate_pct": round(sum(filter_rates) / total, 1),
        },
        "sources": {
            "unique_sources_hit": len(source_counts),
            "top_sources": [{"source": s, "hits": c} for s, c in top_sources[:5]],
        },
        "recent_queries": [
            {"id": l["id"], "query": l["query"][:80], "elapsed": l["elapsed_seconds"], "timestamp": l["timestamp"]}
            for l in recent
        ],
    }

def _percentile(data: list[float], pct: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)