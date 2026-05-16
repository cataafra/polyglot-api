"""
Benchmark helper for the semantic translation memory prototype.

Input CSV columns:
    audio_path,source_language,target_language,domain

Example:
    python scripts/semantic_benchmark.py samples.csv https://localhost --strategy context
"""

import argparse
import csv
import statistics
import time
import uuid
from pathlib import Path

import requests
import urllib3


def post_sample(base_url, row, strategy, session_id, verify_tls):
    endpoint = base_url.rstrip("/") + "/process_memory"
    audio_path = Path(row["audio_path"])
    with audio_path.open("rb") as audio_file:
        files = {"file": (audio_path.name, audio_file, "audio/wav")}
        data = {
            "language": row["target_language"],
            "speaker_id": row.get("speaker_id", "0"),
            "session_id": session_id,
            "source_language": row.get("source_language", "auto"),
            "domain": row.get("domain", "general"),
            "privacy_level": row.get("privacy_level", "transient"),
            "use_semantic_cache": "false" if strategy == "stateless" else "true",
            "cache_strategy": strategy,
        }
        start = time.time()
        response = requests.post(
            endpoint,
            files=files,
            data=data,
            verify=verify_tls,
        )
        elapsed = time.time() - start

    response.raise_for_status()
    return {
        "elapsed": elapsed,
        "cache": response.headers.get("X-Polyglot-Cache", "unknown"),
        "strategy": response.headers.get("X-Polyglot-Cache-Strategy", strategy),
        "similarity": response.headers.get("X-Polyglot-Similarity"),
        "decision": response.headers.get("X-Polyglot-Cache-Decision"),
        "server_total": response.headers.get("X-Polyglot-Total-Time"),
        "server_lookup": response.headers.get("X-Polyglot-Lookup-Time"),
        "server_inference": response.headers.get("X-Polyglot-Inference-Time"),
    }


def summarize(results):
    latencies = [result["elapsed"] for result in results]
    hits = sum(1 for result in results if result["cache"] == "hit")
    return {
        "requests": len(results),
        "hits": hits,
        "hit_rate": hits / len(results) if results else 0.0,
        "p50_latency": statistics.median(latencies) if latencies else 0.0,
        "p95_latency": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies, default=0.0),
        "avg_latency": statistics.mean(latencies) if latencies else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("base_url")
    parser.add_argument(
        "--strategy",
        choices=["stateless", "exact", "semantic", "context"],
        default="context",
    )
    parser.add_argument("--session-id", default=str(uuid.uuid4()))
    parser.add_argument("--verify-tls", action="store_true")
    args = parser.parse_args()

    if not args.verify_tls:
        urllib3.disable_warnings(category=urllib3.exceptions.InsecureRequestWarning)

    with Path(args.csv_path).open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    results = [
        post_sample(
            base_url=args.base_url,
            row=row,
            strategy=args.strategy,
            session_id=args.session_id,
            verify_tls=args.verify_tls,
        )
        for row in rows
    ]

    print("Summary")
    for key, value in summarize(results).items():
        print(f"{key}: {value}")

    print("\nRequests")
    for index, result in enumerate(results, start=1):
        print(
            f"{index}: cache={result['cache']} "
            f"strategy={result['strategy']} "
            f"elapsed={result['elapsed']:.4f}s "
            f"similarity={result['similarity']} "
            f"decision={result['decision']}"
        )


if __name__ == "__main__":
    main()
