"""
Evaluation runner for the database-augmented speech-to-speech API.

The runner sends a manifest of WAV files to /process_memory and writes
machine-readable artifacts for thesis/demo analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote

import requests
import urllib3


CACHE_LAYERS = ("miss", "audio_exact", "text_exact", "text_vector", "audio_vector")
STRATEGIES = ("stateless", "exact", "semantic", "context")


@dataclass(frozen=True)
class BenchmarkConfig:
    manifest: Path
    base_url: str
    output_dir: Path
    strategy: str
    session_id: str
    verify_tls: bool
    repeat_count: int
    warmup_count: int
    concurrency: int
    use_transcript_memory: bool
    save_audio: bool
    expected_cache_layer_column: str
    timeout_seconds: float
    database_url: str | None
    api_key: str | None


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    required = {"case_id", "audio_path", "source_language", "target_language"}
    missing = sorted(required - set(rows[0].keys())) if rows else sorted(required)
    if missing:
        raise ValueError(f"manifest is missing required columns: {', '.join(missing)}")
    return rows


def expand_rows(rows: Iterable[dict[str, str]], repeat_count: int) -> list[dict[str, str]]:
    expanded = []
    for row in rows:
        for repeat_index in range(1, repeat_count + 1):
            copy = dict(row)
            copy["repeat_index"] = str(repeat_index)
            expanded.append(copy)
    return expanded


def resolve_audio_path(manifest_path: Path, audio_path: str) -> Path:
    path = Path(audio_path)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def post_sample(
    config: BenchmarkConfig,
    row: dict[str, str],
    request_index: int,
    warmup: bool = False,
) -> dict[str, Any]:
    endpoint = config.base_url.rstrip("/") + "/process_memory"
    audio_path = resolve_audio_path(config.manifest, row["audio_path"])
    session_id = row.get("session_id") or config.session_id
    expected_layer = row.get(config.expected_cache_layer_column, "")
    expected_reuse = parse_bool(row.get("expected_reuse_allowed"), default=True)
    strategy = row.get("strategy") or config.strategy

    data = {
        "language": row["target_language"],
        "speaker_id": row.get("speaker_id") or "0",
        "session_id": session_id,
        "source_language": row.get("source_language") or "auto",
        "domain": row.get("domain") or "general",
        "privacy_level": row.get("privacy_level") or "transient",
        "use_semantic_cache": "false" if strategy == "stateless" else "true",
        "cache_strategy": strategy,
        "use_transcript_memory": str(config.use_transcript_memory).lower(),
    }

    started_at = time.time()
    with audio_path.open("rb") as audio_file:
        response = requests.post(
            endpoint,
            files={"file": (audio_path.name, audio_file, "audio/wav")},
            data=data,
            headers=auth_headers(config.api_key),
            verify=config.verify_tls,
            timeout=config.timeout_seconds,
        )
    elapsed = time.time() - started_at
    response.raise_for_status()

    if config.save_audio and not warmup:
        audio_dir = config.output_dir / "outputs"
        audio_dir.mkdir(parents=True, exist_ok=True)
        output_name = f"{request_index:04d}-{row.get('case_id', 'case')}.wav"
        (audio_dir / output_name).write_bytes(response.content)

    return parse_response(
        row=row,
        request_index=request_index,
        strategy=strategy,
        expected_cache_layer=expected_layer,
        expected_reuse_allowed=expected_reuse,
        elapsed=elapsed,
        headers=dict(response.headers),
        warmup=warmup,
    )


def parse_response(
    row: dict[str, str],
    request_index: int,
    strategy: str,
    expected_cache_layer: str,
    expected_reuse_allowed: bool,
    elapsed: float,
    headers: dict[str, str],
    warmup: bool = False,
) -> dict[str, Any]:
    cache = headers.get("X-Polyglot-Cache", "unknown")
    cache_layer = headers.get("X-Polyglot-Cache-Layer", "unknown")
    reuse_happened = cache == "hit"
    layer_matches = not expected_cache_layer or expected_cache_layer == cache_layer
    safe_reuse = expected_reuse_allowed or not reuse_happened

    return {
        "request_index": request_index,
        "warmup": warmup,
        "case_id": row.get("case_id", ""),
        "group_id": row.get("group_id", ""),
        "repeat_index": int(row.get("repeat_index") or 1),
        "workload": row.get("workload", ""),
        "audio_path": row.get("audio_path", ""),
        "source_language": row.get("source_language", "auto"),
        "target_language": row.get("target_language", ""),
        "speaker_id": row.get("speaker_id", "0"),
        "domain": row.get("domain", "general"),
        "privacy_level": row.get("privacy_level", "transient"),
        "strategy": headers.get("X-Polyglot-Cache-Strategy", strategy),
        "cache": cache,
        "cache_layer": cache_layer,
        "decision": headers.get("X-Polyglot-Cache-Decision") or headers.get("X-Polyglot-Decision", ""),
        "source_transcript": unquote(headers.get("X-Polyglot-Source-Transcript", "")),
        "normalized_text": unquote(headers.get("X-Polyglot-Normalized-Text", "")),
        "similarity": to_float(headers.get("X-Polyglot-Similarity")),
        "text_similarity": to_float(headers.get("X-Polyglot-Text-Similarity")),
        "lookup_time": to_float(headers.get("X-Polyglot-Lookup-Time")),
        "transcript_time": to_float(headers.get("X-Polyglot-Transcript-Time")),
        "inference_time": to_float(headers.get("X-Polyglot-Inference-Time")),
        "server_total_time": to_float(headers.get("X-Polyglot-Total-Time")),
        "client_total_time": elapsed,
        "translation_id": headers.get("X-Polyglot-Translation-Id", ""),
        "expected_cache_layer": expected_cache_layer,
        "expected_reuse_allowed": expected_reuse_allowed,
        "correct": bool(layer_matches and safe_reuse),
        "notes": row.get("notes", ""),
    }


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def auth_headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"X-Polyglot-Api-Key": api_key}


def run_benchmark(config: BenchmarkConfig) -> list[dict[str, Any]]:
    rows = read_manifest(config.manifest)
    workload = expand_rows(rows, config.repeat_count)

    for warmup_index, row in enumerate(workload[: config.warmup_count], start=1):
        post_sample(config, row, warmup_index, warmup=True)

    if config.concurrency <= 1:
        return [
            post_sample(config, row, request_index=index)
            for index, row in enumerate(workload, start=1)
        ]

    results = []
    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        futures = {
            executor.submit(post_sample, config, row, index): index
            for index, row in enumerate(workload, start=1)
        }
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda result: result["request_index"])


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    hits = sum(1 for result in results if result["cache"] == "hit")
    incorrect = sum(1 for result in results if not result["correct"])
    latencies = values(results, "client_total_time")
    inference = values(results, "inference_time")
    lookup = values(results, "lookup_time")
    transcript = values(results, "transcript_time")
    duration = sum(latencies)

    summary = {
        "requests": total,
        "hits": hits,
        "misses": sum(1 for result in results if result["cache"] == "miss"),
        "hit_rate": ratio(hits, total),
        "incorrect_cases": incorrect,
        "safe_reuse_rate": ratio(total - incorrect, total),
        "throughput_requests_per_minute": ratio(total * 60.0, duration),
        "p50_client_total_time": percentile(latencies, 50),
        "p95_client_total_time": percentile(latencies, 95),
        "p99_client_total_time": percentile(latencies, 99),
        "avg_client_total_time": mean(latencies),
        "avg_lookup_time": mean(lookup),
        "p95_lookup_time": percentile(lookup, 95),
        "avg_transcript_time": mean(transcript),
        "avg_inference_time": mean(inference),
    }
    for layer in CACHE_LAYERS:
        count = sum(1 for result in results if result["cache_layer"] == layer)
        summary[f"{layer}_rate"] = ratio(count, total)
    return summary


def group_summaries(results: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        groups.setdefault(str(result.get(key) or "unknown"), []).append(result)
    return [{"group": group, **summarize(group_results)} for group, group_results in sorted(groups.items())]


def values(results: list[dict[str, Any]], key: str) -> list[float]:
    return [float(result[key]) for result in results if result.get(key) is not None]


def mean(numbers: list[float]) -> float:
    return statistics.mean(numbers) if numbers else 0.0


def percentile(numbers: list[float], percentile_value: int) -> float:
    if not numbers:
        return 0.0
    ordered = sorted(numbers)
    index = math.ceil((percentile_value / 100) * len(ordered)) - 1
    return ordered[min(max(index, 0), len(ordered) - 1)]


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def write_outputs(config: BenchmarkConfig, results: list[dict[str, Any]]) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(config.output_dir / "raw_requests.jsonl", results)
    write_csv(config.output_dir / "summary.csv", [summarize(results)])
    write_csv(config.output_dir / "summary_by_strategy.csv", group_summaries(results, "strategy"))
    write_csv(config.output_dir / "summary_by_cache_layer.csv", group_summaries(results, "cache_layer"))
    write_csv(config.output_dir / "summary_by_workload.csv", group_summaries(results, "workload"))
    write_csv(config.output_dir / "latency_distribution.csv", latency_distribution(results))
    write_csv(config.output_dir / "db_stats.csv", collect_db_stats(config.database_url))
    write_csv(
        config.output_dir / "incorrect_reuse_cases.csv",
        [result for result in results if not result["correct"]],
    )
    write_markdown_report(config.output_dir / "evaluation_report.md", config, results)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latency_distribution(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for metric in ("client_total_time", "server_total_time", "lookup_time", "transcript_time", "inference_time"):
        metric_values = values(results, metric)
        rows.append(
            {
                "metric": metric,
                "count": len(metric_values),
                "avg": mean(metric_values),
                "p50": percentile(metric_values, 50),
                "p95": percentile(metric_values, 95),
                "p99": percentile(metric_values, 99),
                "min": min(metric_values, default=0.0),
                "max": max(metric_values, default=0.0),
            }
        )
    return rows


def collect_db_stats(database_url: str | None) -> list[dict[str, Any]]:
    if not database_url:
        return [{"status": "not_configured"}]
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError:
        return [{"status": "psycopg_not_installed"}]

    tables = [
        "sessions",
        "audio_segments",
        "audio_translations",
        "audio_embeddings",
        "text_embeddings",
        "audit_events",
    ]
    rows: list[dict[str, Any]] = []
    try:
        with psycopg.connect(database_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        pg_database_size(current_database()) / 1024.0 / 1024.0 AS database_size_mb
                    """
                )
                database_size_mb = float(cur.fetchone()["database_size_mb"])
                for table in tables:
                    cur.execute(
                        """
                        SELECT
                            to_regclass(%s) IS NOT NULL AS exists,
                            COALESCE(pg_total_relation_size(to_regclass(%s)), 0) / 1024.0 / 1024.0 AS total_size_mb,
                            COALESCE(pg_indexes_size(to_regclass(%s)), 0) / 1024.0 / 1024.0 AS index_size_mb
                        """,
                        (table, table, table),
                    )
                    size_row = cur.fetchone()
                    count = 0
                    if size_row["exists"]:
                        cur.execute(f"SELECT COUNT(*) AS count FROM {table}")
                        count = int(cur.fetchone()["count"])
                    rows.append(
                        {
                            "status": "ok",
                            "table": table,
                            "rows": count,
                            "database_size_mb": database_size_mb,
                            "total_size_mb": float(size_row["total_size_mb"]),
                            "index_size_mb": float(size_row["index_size_mb"]),
                        }
                    )
        return rows
    except Exception as exc:
        return [{"status": "error", "error": str(exc)}]


def write_markdown_report(path: Path, config: BenchmarkConfig, results: list[dict[str, Any]]) -> None:
    summary = summarize(results)
    layers = group_summaries(results, "cache_layer")
    workloads = group_summaries(results, "workload")
    lines = [
        "# Semantic Memory Evaluation Report",
        "",
        f"- Manifest: `{config.manifest}`",
        f"- Strategy: `{config.strategy}`",
        f"- Session: `{config.session_id}`",
        f"- Requests: `{summary['requests']}`",
        f"- Hit rate: `{summary['hit_rate']:.2%}`",
        f"- Incorrect cases: `{summary['incorrect_cases']}`",
        f"- p50 client latency: `{summary['p50_client_total_time']:.4f}s`",
        f"- p95 client latency: `{summary['p95_client_total_time']:.4f}s`",
        f"- Average inference time: `{summary['avg_inference_time']:.4f}s`",
        f"- DB stats: `{'enabled' if config.database_url else 'not configured'}`",
        "",
        "## Cache Layers",
        "",
        "| Layer | Requests | Hit rate | p95 latency | Avg inference |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in layers:
        lines.append(
            f"| {row['group']} | {row['requests']} | {row['hit_rate']:.2%} | "
            f"{row['p95_client_total_time']:.4f}s | {row['avg_inference_time']:.4f}s |"
        )
    lines.extend(["", "## Workloads", "", "| Workload | Requests | Hit rate | Incorrect | p95 latency |", "|---|---:|---:|---:|---:|"])
    for row in workloads:
        lines.append(
            f"| {row['group']} | {row['requests']} | {row['hit_rate']:.2%} | "
            f"{row['incorrect_cases']} | {row['p95_client_total_time']:.4f}s |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `audio_exact` demonstrates repeat-audio reuse.",
            "- `text_exact` demonstrates naturally repeated speech when transcripts normalize identically.",
            "- `text_vector` should be treated conservatively and manually checked for false reuse.",
            "- `audio_vector` is an acoustic-fingerprint baseline, not a learned semantic speech embedding.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Polyglot semantic-memory evaluation workloads.")
    parser.add_argument("legacy_manifest", nargs="?", help="Backward-compatible manifest path.")
    parser.add_argument("legacy_base_url", nargs="?", help="Backward-compatible base URL.")
    parser.add_argument("--manifest", help="CSV manifest path.")
    parser.add_argument("--base-url", help="API base URL, e.g. https://localhost.")
    parser.add_argument("--output-dir", default="evaluation/results")
    parser.add_argument("--strategy", choices=STRATEGIES, default="context")
    parser.add_argument("--session-id", default=f"eval-{uuid.uuid4()}")
    parser.add_argument("--verify-tls", action="store_true")
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--warmup-count", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--use-transcript-memory", default="true")
    parser.add_argument("--save-audio", action="store_true")
    parser.add_argument("--expected-cache-layer-column", default="expected_cache_layer")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--database-url", default=os.getenv("POLYGLOT_DATABASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("POLYGLOT_API_TOKEN"))
    return parser


def config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    manifest = args.manifest or args.legacy_manifest
    base_url = args.base_url or args.legacy_base_url
    if not manifest or not base_url:
        raise SystemExit("--manifest and --base-url are required")
    return BenchmarkConfig(
        manifest=Path(manifest),
        base_url=base_url,
        output_dir=Path(args.output_dir),
        strategy=args.strategy,
        session_id=args.session_id,
        verify_tls=args.verify_tls,
        repeat_count=max(1, args.repeat_count),
        warmup_count=max(0, args.warmup_count),
        concurrency=max(1, args.concurrency),
        use_transcript_memory=parse_bool(args.use_transcript_memory, default=True),
        save_audio=args.save_audio,
        expected_cache_layer_column=args.expected_cache_layer_column,
        timeout_seconds=args.timeout_seconds,
        database_url=args.database_url,
        api_key=args.api_key,
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    config = config_from_args(args)
    if not config.verify_tls:
        urllib3.disable_warnings(category=urllib3.exceptions.InsecureRequestWarning)
    results = run_benchmark(config)
    write_outputs(config, results)

    summary = summarize(results)
    print("Summary")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nArtifacts written to: {config.output_dir}")


if __name__ == "__main__":
    main()
