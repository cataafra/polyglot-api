"""
Measure semantic-memory database behavior at controlled row counts.

This script intentionally resets the evaluation database for each scale level.
Use it only against the benchmark Postgres instance, never a production DB.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from seed_semantic_eval_db import seed_database, synthetic_audio_embedding  # noqa: E402
from semantic_benchmark import collect_db_stats  # noqa: E402
from polyglot_api.semantic_memory import vector_literal  # noqa: E402
from polyglot_api.text_semantics import build_text_embedding, hash_normalized_text  # noqa: E402


TABLES = [
    "audit_events",
    "terminology_memory",
    "text_embeddings",
    "audio_embeddings",
    "audio_translations",
    "audio_segments",
    "model_versions",
    "sessions",
]


def parse_sizes(value: str, include_1m: bool = False) -> list[int]:
    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if include_1m and 1_000_000 not in sizes:
        sizes.append(1_000_000)
    if not sizes:
        raise ValueError("at least one DB scale size is required")
    return sorted(set(sizes))


def reset_database(database_url: str) -> None:
    import psycopg

    schema_sql = (ROOT / "src" / "polyglot_api" / "db_schema.sql").read_text(encoding="utf-8")
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            for statement in schema_sql.split(";"):
                statement = statement.strip()
                if statement:
                    cur.execute(statement)
            cur.execute("TRUNCATE TABLE " + ", ".join(TABLES) + " RESTART IDENTITY CASCADE")
        conn.commit()


def summarize_db_stats(database_url: str) -> dict[str, Any]:
    rows = collect_db_stats(database_url)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return {"db_status": rows[0].get("status", "unknown") if rows else "unknown"}
    return {
        "db_status": "ok",
        "database_size_mb": max(float(row.get("database_size_mb") or 0.0) for row in ok_rows),
        "total_relation_size_mb": sum(float(row.get("total_size_mb") or 0.0) for row in ok_rows),
        "total_index_size_mb": sum(float(row.get("index_size_mb") or 0.0) for row in ok_rows),
        "audio_segments_rows": next(
            (int(row.get("rows") or 0) for row in ok_rows if row.get("table") == "audio_segments"),
            0,
        ),
        "text_embeddings_rows": next(
            (int(row.get("rows") or 0) for row in ok_rows if row.get("table") == "text_embeddings"),
            0,
        ),
        "audio_embeddings_rows": next(
            (int(row.get("rows") or 0) for row in ok_rows if row.get("table") == "audio_embeddings"),
            0,
        ),
    }


def measure_lookup_latencies(database_url: str, records: int, repeats: int) -> dict[str, Any]:
    import psycopg

    probe_index = max(0, records // 2)
    audio_hash = hashlib.sha256(f"audio:{probe_index}".encode("utf-8")).hexdigest()
    text_hash = hash_normalized_text(f"synthetic phrase {probe_index % 1000}")
    text_embedding = vector_literal(build_text_embedding(f"synthetic phrase {probe_index % 1000}"))
    audio_embedding = vector_literal(synthetic_audio_embedding(probe_index))

    queries = {
        "audio_exact": (
            """
            SELECT at.id
            FROM audio_translations at
            JOIN audio_segments aseg ON aseg.id = at.audio_segment_id
            WHERE aseg.source_audio_hash = %s
              AND aseg.source_language = 'ron'
              AND at.target_language = 'eng'
              AND at.speaker_id = %s
              AND aseg.domain = %s
              AND aseg.privacy_level = 'transient'
            LIMIT 1
            """,
            (audio_hash, str(probe_index % 4), "legal" if probe_index % 2 == 0 else "general"),
        ),
        "text_exact": (
            """
            SELECT at.id
            FROM audio_translations at
            JOIN audio_segments aseg ON aseg.id = at.audio_segment_id
            WHERE aseg.source_text_hash = %s
              AND aseg.source_language = 'ron'
              AND at.target_language = 'eng'
              AND at.speaker_id = %s
              AND aseg.domain = %s
              AND aseg.privacy_level = 'transient'
            LIMIT 1
            """,
            (text_hash, str(probe_index % 4), "legal" if probe_index % 2 == 0 else "general"),
        ),
        "text_vector": (
            """
            SELECT at.id
            FROM text_embeddings te
            JOIN audio_segments aseg ON aseg.id = te.audio_segment_id
            JOIN audio_translations at ON at.audio_segment_id = aseg.id
            WHERE aseg.source_language = 'ron'
              AND at.target_language = 'eng'
              AND aseg.privacy_level = 'transient'
            ORDER BY te.embedding <=> %s::vector
            LIMIT 1
            """,
            (text_embedding,),
        ),
        "audio_vector": (
            """
            SELECT at.id
            FROM audio_embeddings ae
            JOIN audio_segments aseg ON aseg.id = ae.audio_segment_id
            JOIN audio_translations at ON at.audio_segment_id = aseg.id
            WHERE aseg.source_language = 'ron'
              AND at.target_language = 'eng'
              AND aseg.privacy_level = 'transient'
            ORDER BY ae.embedding <=> %s::vector
            LIMIT 1
            """,
            (audio_embedding,),
        ),
    }

    metrics: dict[str, Any] = {}
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            for name, (query, params) in queries.items():
                durations = []
                hits = 0
                for _ in range(max(1, repeats)):
                    started_at = time.perf_counter()
                    cur.execute(query, params)
                    row = cur.fetchone()
                    durations.append(time.perf_counter() - started_at)
                    hits += 1 if row else 0
                metrics[f"{name}_hits"] = hits
                metrics[f"{name}_avg_ms"] = mean_ms(durations)
                metrics[f"{name}_p95_ms"] = percentile_ms(durations, 95)
    return metrics


def mean_ms(values: list[float]) -> float:
    return statistics.mean(values) * 1000.0 if values else 0.0


def percentile_ms(values: list[float], percentile_value: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((percentile_value / 100.0) * len(ordered)) - 1))
    return ordered[index] * 1000.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_scale_evaluation(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.confirm_reset:
        raise SystemExit("--confirm-reset is required because this script truncates the evaluation database")

    rows = []
    for records in parse_sizes(args.sizes, include_1m=args.include_1m):
        print(f"Preparing DB scale level: {records} rows")
        reset_database(args.database_url)
        started_seed = time.perf_counter()
        seed_database(args.database_url, records=records, batch_size=args.batch_size)
        seed_seconds = time.perf_counter() - started_seed
        rows.append(
            {
                "scale_rows": records,
                "seed_seconds": seed_seconds,
                **summarize_db_stats(args.database_url),
                **measure_lookup_latencies(args.database_url, records=records, repeats=args.repeats),
            }
        )

    output_path = Path(args.output)
    write_csv(output_path, rows)
    print(f"DB scale results written to {output_path}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run controlled DB scale evaluation.")
    parser.add_argument("--database-url", default=os.getenv("POLYGLOT_DATABASE_URL"), required=False)
    parser.add_argument("--output", default="evaluation/runpod/results/db_scale.csv")
    parser.add_argument("--sizes", default="0,10000,100000")
    parser.add_argument("--include-1m", action="store_true")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--confirm-reset", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.database_url:
        raise SystemExit("--database-url or POLYGLOT_DATABASE_URL is required")
    run_scale_evaluation(args)


if __name__ == "__main__":
    main()
