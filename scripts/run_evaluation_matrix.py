"""
Run research evaluation manifests across cache strategies.

This is the RunPod-facing orchestrator. It calls semantic_benchmark for each
manifest/strategy pair and then writes combined thesis artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "semantic_benchmark.py"
STRATEGIES = ("stateless", "exact", "semantic", "context")
RESET_TABLES = [
    "audit_events",
    "terminology_memory",
    "text_embeddings",
    "audio_embeddings",
    "audio_translations",
    "audio_segments",
    "model_versions",
    "sessions",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_command(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def run_matrix(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifests = [Path(item) for item in args.manifests]
    strategies = parse_strategies(args.strategies)
    output_root = Path(args.output_dir)
    all_rows: list[dict[str, Any]] = []
    run_id = args.run_id or f"runpod-{uuid.uuid4().hex[:8]}"

    for manifest in manifests:
        manifest_name = manifest.stem
        for strategy in strategies:
            strategy_output = output_root / manifest_name / strategy
            session_id = f"{run_id}-{manifest_name}-{strategy}"
            if args.reset_database:
                if not args.database_url:
                    raise SystemExit("--reset-database requires --database-url")
                reset_database(args.database_url)
            command = [
                sys.executable,
                str(SCRIPT),
                "--manifest",
                str(manifest),
                "--base-url",
                args.base_url,
                "--strategy",
                strategy,
                "--session-id",
                session_id,
                "--output-dir",
                str(strategy_output),
                "--use-transcript-memory",
                args.use_transcript_memory,
                "--timeout-seconds",
                str(args.timeout_seconds),
                "--pod-hour-usd",
                str(args.pod_hour_usd),
            ]
            if args.api_key:
                command.extend(["--api-key", args.api_key])
            if args.verify_tls:
                command.append("--verify-tls")
            if args.database_url:
                command.extend(["--database-url", args.database_url])
            if args.concurrency:
                command.extend(["--concurrency", str(args.concurrency)])
            if args.warmup_count:
                command.extend(["--warmup-count", str(args.warmup_count)])
            run_command(command)
            rows = read_jsonl(strategy_output / "raw_requests.jsonl")
            for row in rows:
                row["matrix_manifest"] = manifest_name
                row["matrix_run_id"] = run_id
            all_rows.extend(rows)

    return all_rows


def parse_strategies(value: str) -> list[str]:
    strategies = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(strategies) - set(STRATEGIES))
    if unknown:
        raise SystemExit(f"unknown strategies: {', '.join(unknown)}")
    return strategies or list(STRATEGIES)


def reset_database(database_url: str) -> None:
    import psycopg

    schema_sql = (ROOT / "src" / "polyglot_api" / "db_schema.sql").read_text(encoding="utf-8")
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            for statement in schema_sql.split(";"):
                statement = statement.strip()
                if statement:
                    cur.execute(statement)
            cur.execute("TRUNCATE TABLE " + ", ".join(RESET_TABLES) + " RESTART IDENTITY CASCADE")
        conn.commit()


def write_combined_outputs(output_root: Path, rows: list[dict[str, Any]], pod_hour_usd: float = 0.0) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import semantic_benchmark

    combined = output_root / "combined"
    combined.mkdir(parents=True, exist_ok=True)
    with (combined / "raw_requests.jsonl").open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(combined / "summary_by_strategy.csv", semantic_benchmark.group_summaries(rows, "strategy"))
    write_csv(combined / "summary_by_dataset.csv", semantic_benchmark.group_summaries(rows, "dataset"))
    write_csv(combined / "summary_by_cache_layer.csv", semantic_benchmark.group_summaries(rows, "cache_layer"))
    write_csv(combined / "summary_by_workload.csv", semantic_benchmark.group_summaries(rows, "workload"))
    write_csv(combined / "latency_distribution.csv", semantic_benchmark.latency_distribution(rows))
    write_csv(combined / "cache_confusion_matrix.csv", semantic_benchmark.confusion_matrix_rows(rows))
    write_csv(combined / "quality_metrics.csv", semantic_benchmark.quality_metrics_rows(rows))
    write_csv(combined / "cost_model.csv", semantic_benchmark.cost_model_rows(rows, pod_hour_usd=pod_hour_usd))
    write_csv(combined / "threshold_sweep.csv", semantic_benchmark.threshold_sweep_rows(rows))
    write_csv(combined / "incorrect_reuse_cases.csv", [row for row in rows if not row.get("correct")])
    semantic_benchmark.write_plots(combined, rows, pod_hour_usd)
    write_final_report(combined / "final_evaluation_report.md", rows, pod_hour_usd)


def write_final_report(path: Path, rows: list[dict[str, Any]], pod_hour_usd: float = 0.0) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import semantic_benchmark

    summary = semantic_benchmark.summarize(rows)
    strategy_rows = semantic_benchmark.group_summaries(rows, "strategy")
    dataset_rows = semantic_benchmark.group_summaries(rows, "dataset")
    confusion_rows = semantic_benchmark.confusion_matrix_rows(rows)
    cost_rows = semantic_benchmark.cost_model_rows(rows, pod_hour_usd)
    lines = [
        "# Polyglot Research Evaluation Report",
        "",
        f"- Requests: `{summary['requests']}`",
        f"- Overall hit rate: `{summary['hit_rate']:.2%}`",
        f"- Incorrect cases: `{summary['incorrect_cases']}`",
        f"- Safe reuse rate: `{summary['safe_reuse_rate']:.2%}`",
        "",
        "## Strategy Summary",
        "",
        "| Strategy | Requests | Hit rate | Incorrect | p95 latency | Avg inference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in strategy_rows:
        lines.append(
            f"| {row['group']} | {row['requests']} | {row['hit_rate']:.2%} | "
            f"{row['incorrect_cases']} | {row['p95_client_total_time']:.4f}s | {row['avg_inference_time']:.4f}s |"
        )
    lines.extend(["", "## Dataset Summary", "", "| Dataset | Requests | Hit rate | Incorrect | p95 latency |", "|---|---:|---:|---:|---:|"])
    for row in dataset_rows:
        lines.append(
            f"| {row['group']} | {row['requests']} | {row['hit_rate']:.2%} | "
            f"{row['incorrect_cases']} | {row['p95_client_total_time']:.4f}s |"
        )
    lines.extend(
        [
            "",
            "## Cache Safety",
            "",
            "| Strategy | True positive | False positive | False negative | True negative | Precision | Recall |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in confusion_rows:
        lines.append(
            f"| {row['strategy']} | {row['true_positive']} | {row['false_positive']} | "
            f"{row['false_negative']} | {row['true_negative']} | {row['precision']:.2%} | {row['recall']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Cost Model",
            "",
            "| Strategy | Correct hits | Avoided inference seconds | Estimated savings / 1k requests |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in cost_rows:
        lines.append(
            f"| {row['strategy']} | {row['correct_cache_hits']} | "
            f"{row['avoided_inference_seconds']:.4f} | "
            f"${row['estimated_savings_per_1000_requests_usd']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full Polyglot evaluation matrix.")
    parser.add_argument("--manifest", dest="manifests", action="append", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", default="evaluation/runpod/results")
    parser.add_argument("--strategies", default="stateless,exact,semantic,context")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--database-url", default="")
    parser.add_argument("--verify-tls", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup-count", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--use-transcript-memory", default="true")
    parser.add_argument("--pod-hour-usd", type=float, default=0.0)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--reset-database", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = run_matrix(args)
    write_combined_outputs(Path(args.output_dir), rows, pod_hour_usd=args.pod_hour_usd)
    print(f"Combined artifacts written to {Path(args.output_dir) / 'combined'}")


if __name__ == "__main__":
    main()
