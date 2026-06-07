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
    pod_hour_usd: float


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
    content_type = response.headers.get("content-type", "")
    if "audio/" not in content_type and "application/octet-stream" not in content_type:
        raise RuntimeError(
            f"expected audio response from {endpoint}, got {content_type or 'unknown'}: "
            f"{response.text[:500]}"
        )

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
    headers = {str(key).lower(): value for key, value in headers.items()}
    expected_cache_layer, expected_reuse_allowed = strategy_expectation(
        strategy,
        row,
        expected_cache_layer,
        expected_reuse_allowed,
    )
    cache = headers.get("x-polyglot-cache", "unknown")
    cache_layer = headers.get("x-polyglot-cache-layer", "unknown")
    reuse_happened = cache == "hit"
    layer_matches = not expected_cache_layer or expected_cache_layer == cache_layer
    safe_reuse = expected_reuse_allowed or not reuse_happened

    return {
        "request_index": request_index,
        "warmup": warmup,
        "case_id": row.get("case_id", ""),
        "dataset": row.get("dataset", ""),
        "split": row.get("split", ""),
        "group_id": row.get("group_id", ""),
        "reuse_group": row.get("reuse_group", row.get("group_id", "")),
        "repeat_index": int(row.get("repeat_index") or 1),
        "workload": row.get("workload", ""),
        "audio_path": row.get("audio_path", ""),
        "source_language": row.get("source_language", "auto"),
        "target_language": row.get("target_language", ""),
        "source_text": row.get("source_text", ""),
        "reference_text": row.get("reference_text", ""),
        "quality_required": parse_bool(row.get("quality_required"), default=bool(row.get("reference_text", ""))),
        "source_clip_id": row.get("source_clip_id", ""),
        "speaker_id": row.get("speaker_id", "0"),
        "domain": row.get("domain", "general"),
        "privacy_level": row.get("privacy_level", "transient"),
        "strategy": strategy,
        "cache_strategy": headers.get("x-polyglot-cache-strategy", strategy),
        "cache": cache,
        "cache_layer": cache_layer,
        "decision": headers.get("x-polyglot-cache-decision") or headers.get("x-polyglot-decision", ""),
        "source_transcript": unquote(headers.get("x-polyglot-source-transcript", "")),
        "normalized_text": unquote(headers.get("x-polyglot-normalized-text", "")),
        "hypothesis_text": unquote(
            headers.get("x-polyglot-hypothesis-text", "")
            or headers.get("x-polyglot-translation-text", "")
        ),
        "similarity": to_float(headers.get("x-polyglot-similarity")),
        "text_similarity": to_float(headers.get("x-polyglot-text-similarity")),
        "lookup_time": to_float(headers.get("x-polyglot-lookup-time")),
        "transcript_time": to_float(headers.get("x-polyglot-transcript-time")),
        "inference_time": to_float(headers.get("x-polyglot-inference-time")),
        "server_total_time": to_float(headers.get("x-polyglot-total-time")),
        "client_total_time": elapsed,
        "translation_id": headers.get("x-polyglot-translation-id", ""),
        "expected_cache_layer": expected_cache_layer,
        "expected_reuse_allowed": expected_reuse_allowed,
        "correct": bool(layer_matches and safe_reuse),
        "notes": row.get("notes", ""),
    }


def strategy_expectation(
    strategy: str,
    row: dict[str, str],
    expected_cache_layer: str,
    expected_reuse_allowed: bool,
) -> tuple[str, bool]:
    workload = row.get("workload", "")
    if strategy == "stateless":
        return "miss", expected_reuse_allowed
    if strategy == "exact":
        return ("audio_exact", expected_reuse_allowed) if workload == "exact_replay" else ("miss", expected_reuse_allowed)
    return expected_cache_layer, expected_reuse_allowed


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
        results = []
        total = len(workload)
        progress_interval = max(1, int(os.getenv("POLYGLOT_BENCHMARK_PROGRESS_INTERVAL", "100")))
        started_at = time.time()
        for index, row in enumerate(workload, start=1):
            results.append(post_sample(config, row, request_index=index))
            if index == total or index % progress_interval == 0:
                elapsed = max(time.time() - started_at, 1e-9)
                rate = index / elapsed
                remaining = (total - index) / rate if rate else 0.0
                print(
                    f"progress strategy={config.strategy} manifest={config.manifest.name} "
                    f"{index}/{total} rate={rate:.2f}/s eta_seconds={remaining:.0f}",
                    flush=True,
                )
        return results

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
    write_csv(config.output_dir / "summary_by_dataset.csv", group_summaries(results, "dataset"))
    write_csv(config.output_dir / "summary_by_cache_layer.csv", group_summaries(results, "cache_layer"))
    write_csv(config.output_dir / "summary_by_workload.csv", group_summaries(results, "workload"))
    write_csv(config.output_dir / "latency_distribution.csv", latency_distribution(results))
    write_csv(config.output_dir / "db_stats.csv", collect_db_stats(config.database_url))
    write_csv(config.output_dir / "cache_confusion_matrix.csv", confusion_matrix_rows(results))
    write_csv(config.output_dir / "quality_metrics.csv", quality_metrics_rows(results))
    write_csv(config.output_dir / "cost_model.csv", cost_model_rows(results, config.pod_hour_usd))
    write_csv(config.output_dir / "threshold_sweep.csv", threshold_sweep_rows(results))
    write_csv(
        config.output_dir / "incorrect_reuse_cases.csv",
        [result for result in results if not result["correct"]],
    )
    write_markdown_report(config.output_dir / "evaluation_report.md", config, results)
    write_markdown_report(config.output_dir / "final_evaluation_report.md", config, results)
    write_plots(config.output_dir, results, config.pod_hour_usd)


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


def write_plots(output_dir: Path, results: list[dict[str, Any]], pod_hour_usd: float = 0.0) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        (plots_dir / "plot_status.txt").write_text(
            "matplotlib is not installed; install requirements-eval.txt to render plots.\n",
            encoding="utf-8",
        )
        return

    strategy_rows = group_summaries(results, "strategy")
    cost_rows = cost_model_rows(results, pod_hour_usd)
    confusion_rows = confusion_matrix_rows(results)

    render_bar_chart(
        plt,
        plots_dir / "hit_rate_by_strategy.png",
        [row["group"] for row in strategy_rows],
        [row["hit_rate"] * 100.0 for row in strategy_rows],
        "Cache hit rate by strategy",
        "Hit rate (%)",
    )
    render_bar_chart(
        plt,
        plots_dir / "p95_latency_by_strategy.png",
        [row["group"] for row in strategy_rows],
        [row["p95_client_total_time"] for row in strategy_rows],
        "p95 client latency by strategy",
        "Seconds",
    )
    render_bar_chart(
        plt,
        plots_dir / "correct_hits_by_strategy.png",
        [row["strategy"] for row in cost_rows],
        [row["correct_cache_hits"] for row in cost_rows],
        "Correct avoided inferences by strategy",
        "Correct cache hits",
    )
    render_bar_chart(
        plt,
        plots_dir / "unsafe_reuse_by_strategy.png",
        [row["strategy"] for row in confusion_rows],
        [row["false_positive"] for row in confusion_rows],
        "Unsafe reuse cases by strategy",
        "False positives",
    )


def render_bar_chart(plt, path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    figure_width = max(6.0, 1.3 * max(1, len(labels)))
    fig, ax = plt.subplots(figsize=(figure_width, 4.0))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf"]
    ax.bar(labels, values, color=colors[: len(labels)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)
        tick.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


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


def confusion_matrix_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for strategy, strategy_results in group_by(results, "strategy").items():
        true_positive = sum(
            1
            for row in strategy_results
            if row.get("cache") == "hit" and row.get("expected_reuse_allowed", True) and row.get("correct")
        )
        false_positive = sum(
            1
            for row in strategy_results
            if row.get("cache") == "hit" and not row.get("expected_reuse_allowed", True)
        )
        false_negative = sum(
            1
            for row in strategy_results
            if row.get("cache") != "hit"
            and row.get("expected_reuse_allowed", True)
            and row.get("expected_cache_layer", "") not in {"", "miss"}
        )
        true_negative = sum(
            1
            for row in strategy_results
            if row.get("cache") != "hit"
            and (not row.get("expected_reuse_allowed", True) or row.get("expected_cache_layer", "") in {"", "miss"})
        )
        rows.append(
            {
                "strategy": strategy,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
                "precision": ratio(true_positive, true_positive + false_positive),
                "recall": ratio(true_positive, true_positive + false_negative),
                "f1": ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative),
            }
        )
    return rows


def quality_metrics_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key in ("strategy", "dataset"):
        for group, group_results in group_by(results, key).items():
            rows.append({"group_type": key, "group": group, **quality_summary(group_results)})
    return rows


def quality_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    source_pairs = [
        (row.get("source_text", ""), row.get("source_transcript", ""))
        for row in results
        if row.get("source_text") and row.get("source_transcript")
    ]
    translation_pairs = [
        (row.get("reference_text", ""), row.get("hypothesis_text", ""))
        for row in results
        if row.get("reference_text") and row.get("hypothesis_text")
    ]
    return {
        "source_wer": wer(source_pairs),
        "translation_bleu": bleu(translation_pairs),
        "translation_chrf": chrf(translation_pairs),
        "source_pairs": len(source_pairs),
        "translation_pairs": len(translation_pairs),
        "quality_required": sum(1 for row in results if row.get("quality_required")),
    }


def cost_model_rows(results: list[dict[str, Any]], pod_hour_usd: float) -> list[dict[str, Any]]:
    baseline_by_dataset = {
        dataset: mean(values(dataset_results, "inference_time"))
        for dataset, dataset_results in group_by(
            [row for row in results if row.get("strategy") == "stateless"],
            "dataset",
        ).items()
    }
    global_baseline = mean(values([row for row in results if row.get("strategy") == "stateless"], "inference_time"))
    rows = []
    for strategy, strategy_results in group_by(results, "strategy").items():
        correct_hits = [row for row in strategy_results if row.get("cache") == "hit" and row.get("correct")]
        avoided_seconds = sum(
            baseline_by_dataset.get(row.get("dataset", ""), global_baseline)
            for row in correct_hits
        )
        rows.append(
            {
                "strategy": strategy,
                "requests": len(strategy_results),
                "correct_cache_hits": len(correct_hits),
                "avoided_inference_seconds": avoided_seconds,
                "estimated_savings_usd": avoided_seconds / 3600.0 * pod_hour_usd,
                "estimated_savings_per_1000_requests_usd": (
                    avoided_seconds / max(len(strategy_results), 1) * 1000.0 / 3600.0 * pod_hour_usd
                ),
            }
        )
    return sorted(rows, key=lambda row: row["strategy"])


def threshold_sweep_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row for row in results
        if row.get("text_similarity") is not None or row.get("similarity") is not None
    ]
    rows = []
    for metric in ("text_similarity", "similarity"):
        values_with_labels = [
            (float(row[metric]), bool(row.get("expected_reuse_allowed")))
            for row in candidates
            if row.get(metric) is not None
        ]
        for threshold in [round(value / 100.0, 2) for value in range(70, 101, 2)]:
            accepted = [(score, allowed) for score, allowed in values_with_labels if score >= threshold]
            false_accepts = sum(1 for _, allowed in accepted if not allowed)
            true_accepts = sum(1 for _, allowed in accepted if allowed)
            rows.append(
                {
                    "metric": metric,
                    "threshold": threshold,
                    "accepted": len(accepted),
                    "true_accepts": true_accepts,
                    "false_accepts": false_accepts,
                    "precision": ratio(true_accepts, true_accepts + false_accepts),
                }
            )
    return rows


def group_by(results: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        groups.setdefault(str(result.get(key) or "unknown"), []).append(result)
    return groups


def wer(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    try:
        from jiwer import wer as jiwer_wer
        return float(jiwer_wer([reference for reference, _ in pairs], [hypothesis for _, hypothesis in pairs]))
    except ImportError:
        total_words = sum(len(reference.split()) for reference, _ in pairs)
        errors = sum(simple_edit_distance(reference.split(), hypothesis.split()) for reference, hypothesis in pairs)
        return ratio(errors, total_words)


def bleu(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    try:
        import sacrebleu
        return float(sacrebleu.corpus_bleu([hypothesis for _, hypothesis in pairs], [[reference for reference, _ in pairs]]).score)
    except ImportError:
        return 0.0


def chrf(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    try:
        import sacrebleu
        return float(sacrebleu.corpus_chrf([hypothesis for _, hypothesis in pairs], [[reference for reference, _ in pairs]]).score)
    except ImportError:
        return 0.0


def simple_edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, start=1):
        current = [i]
        for j, right_token in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if left_token == right_token else 1),
                )
            )
        previous = current
    return previous[-1]


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
    strategies = group_summaries(results, "strategy")
    datasets = group_summaries(results, "dataset")
    confusion = confusion_matrix_rows(results)
    quality = quality_metrics_rows(results)
    cost = cost_model_rows(results, config.pod_hour_usd)
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
        "## Strategy Summary",
        "",
        "| Strategy | Requests | Hit rate | Incorrect | Safe reuse | p95 latency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in strategies:
        lines.append(
            f"| {row['group']} | {row['requests']} | {row['hit_rate']:.2%} | "
            f"{row['incorrect_cases']} | {row['safe_reuse_rate']:.2%} | {row['p95_client_total_time']:.4f}s |"
        )
    lines.extend(
        [
            "",
            "## Dataset Summary",
            "",
            "| Dataset | Requests | Hit rate | Incorrect | p95 latency |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in datasets:
        lines.append(
            f"| {row['group']} | {row['requests']} | {row['hit_rate']:.2%} | "
            f"{row['incorrect_cases']} | {row['p95_client_total_time']:.4f}s |"
        )
    lines.extend(
        [
            "",
        "## Cache Layers",
        "",
        "| Layer | Requests | Hit rate | p95 latency | Avg inference |",
        "|---|---:|---:|---:|---:|",
        ]
    )
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
            "## Cache Safety",
            "",
            "| Strategy | TP | FP | FN | TN | Precision | Recall |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in confusion:
        lines.append(
            f"| {row['strategy']} | {row['true_positive']} | {row['false_positive']} | "
            f"{row['false_negative']} | {row['true_negative']} | {row['precision']:.2%} | {row['recall']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Quality Metrics",
            "",
            "| Group type | Group | Source WER | BLEU | chrF | Source pairs | Translation pairs |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in quality:
        lines.append(
            f"| {row['group_type']} | {row['group']} | {row['source_wer']:.4f} | "
            f"{row['translation_bleu']:.2f} | {row['translation_chrf']:.2f} | "
            f"{row['source_pairs']} | {row['translation_pairs']} |"
        )
    lines.extend(
        [
            "",
            "## Cost Model",
            "",
            "| Strategy | Correct hits | Avoided inference seconds | Savings / 1k requests |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in cost:
        lines.append(
            f"| {row['strategy']} | {row['correct_cache_hits']} | "
            f"{row['avoided_inference_seconds']:.4f} | "
            f"${row['estimated_savings_per_1000_requests_usd']:.4f} |"
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
            "- Translation BLEU/chrF require a text hypothesis header or external ASR over translated audio; otherwise the suite records zero translation pairs instead of fabricating quality numbers.",
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
    parser.add_argument("--pod-hour-usd", type=float, default=0.0)
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
        pod_hour_usd=args.pod_hour_usd,
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
