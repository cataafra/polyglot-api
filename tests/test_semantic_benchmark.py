import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "semantic_benchmark.py"
SPEC = importlib.util.spec_from_file_location("semantic_benchmark", SCRIPT_PATH)
semantic_benchmark = importlib.util.module_from_spec(SPEC)
sys.modules["semantic_benchmark"] = semantic_benchmark
SPEC.loader.exec_module(semantic_benchmark)


def test_parse_response_extracts_cache_layer_and_transcript_metrics():
    result = semantic_benchmark.parse_response(
        row={
            "case_id": "ro-caine-002",
            "group_id": "ro-caine",
            "expected_cache_layer": "text_exact",
            "expected_reuse_allowed": "true",
        },
        request_index=1,
        strategy="context",
        expected_cache_layer="text_exact",
        expected_reuse_allowed=True,
        elapsed=1.5,
        headers={
            "X-Polyglot-Cache": "hit",
            "X-Polyglot-Cache-Layer": "text_exact",
            "X-Polyglot-Cache-Decision": "normalized transcript match",
            "X-Polyglot-Source-Transcript": "c%C3%A2ine",
            "X-Polyglot-Normalized-Text": "caine",
            "X-Polyglot-Text-Similarity": "1.000000",
            "X-Polyglot-Lookup-Time": "0.0400",
            "X-Polyglot-Transcript-Time": "1.2100",
            "X-Polyglot-Inference-Time": "0.0000",
            "X-Polyglot-Total-Time": "1.2800",
        },
    )

    assert result["cache"] == "hit"
    assert result["cache_layer"] == "text_exact"
    assert result["source_transcript"] == "câine"
    assert result["normalized_text"] == "caine"
    assert result["text_similarity"] == 1.0
    assert result["correct"] is True


def test_parse_response_marks_unsafe_reuse_as_incorrect():
    result = semantic_benchmark.parse_response(
        row={"case_id": "ro-paine-negative"},
        request_index=1,
        strategy="context",
        expected_cache_layer="miss",
        expected_reuse_allowed=False,
        elapsed=0.5,
        headers={
            "X-Polyglot-Cache": "hit",
            "X-Polyglot-Cache-Layer": "text_vector",
        },
    )

    assert result["correct"] is False


def test_summarize_reports_layer_rates_and_latency_percentiles():
    results = [
        {
            "cache": "miss",
            "cache_layer": "miss",
            "correct": True,
            "client_total_time": 10.0,
            "lookup_time": 0.1,
            "transcript_time": 1.0,
            "inference_time": 8.0,
        },
        {
            "cache": "hit",
            "cache_layer": "text_exact",
            "correct": True,
            "client_total_time": 2.0,
            "lookup_time": 0.1,
            "transcript_time": 1.5,
            "inference_time": 0.0,
        },
    ]

    summary = semantic_benchmark.summarize(results)

    assert summary["requests"] == 2
    assert summary["hit_rate"] == 0.5
    assert summary["text_exact_rate"] == 0.5
    assert summary["p50_client_total_time"] == 2.0
    assert summary["p95_client_total_time"] == 10.0


def test_write_outputs_creates_expected_artifacts(tmp_path):
    config = semantic_benchmark.BenchmarkConfig(
        manifest=tmp_path / "manifest.csv",
        base_url="https://localhost",
        output_dir=tmp_path / "results",
        strategy="context",
        session_id="test-session",
        verify_tls=False,
        repeat_count=1,
        warmup_count=0,
        concurrency=1,
        use_transcript_memory=True,
        save_audio=False,
        expected_cache_layer_column="expected_cache_layer",
        timeout_seconds=1.0,
        database_url=None,
        api_key=None,
    )
    results = [
        {
            "request_index": 1,
            "cache": "hit",
            "cache_layer": "audio_exact",
            "strategy": "context",
            "workload": "replay",
            "correct": True,
            "client_total_time": 0.5,
            "server_total_time": 0.4,
            "lookup_time": 0.05,
            "transcript_time": 0.0,
            "inference_time": 0.0,
        }
    ]

    semantic_benchmark.write_outputs(config, results)

    assert (config.output_dir / "raw_requests.jsonl").exists()
    assert (config.output_dir / "summary.csv").exists()
    assert (config.output_dir / "db_stats.csv").exists()
    assert (config.output_dir / "evaluation_report.md").exists()


def test_auth_headers_are_only_sent_when_api_key_is_configured():
    assert semantic_benchmark.auth_headers(None) == {}
    assert semantic_benchmark.auth_headers("") == {}
    assert semantic_benchmark.auth_headers("demo-secret") == {
        "X-Polyglot-Api-Key": "demo-secret"
    }
