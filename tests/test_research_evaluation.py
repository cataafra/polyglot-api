import csv
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare_data = load_script("prepare_evaluation_data")
run_matrix = load_script("run_evaluation_matrix")
db_scale = load_script("run_db_scale_evaluation")


def test_common_voice_manifest_includes_reuse_and_safety_controls(tmp_path):
    source = tmp_path / "cv-corpus-25.0-2025-06-20" / "ro"
    clips = source / "clips"
    clips.mkdir(parents=True)
    for name in ("a.wav", "b.wav", "c.wav"):
        sf.write(clips / name, np.zeros(1600, dtype="float32"), 16000, format="WAV")
    (source / "validated.tsv").write_text(
        "\n".join(
            [
                "client_id\tpath\tsentence",
                "speaker-a\ta.wav\tAna are mere.",
                "speaker-b\tb.wav\tana are mere",
                "speaker-c\tc.wav\tAna are pere.",
            ]
        ),
        encoding="utf-8",
    )
    args = Namespace(
        output_root=str(tmp_path / "evaluation"),
        source_dir=str(source),
        max_groups=1,
        max_negative_controls=0,
    )

    manifest = prepare_data.prepare_common_voice(args)
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    workloads = {row["workload"]: row for row in rows}

    assert {"seed", "natural_repeat", "exact_replay", "speaker_control", "domain_control"} <= set(workloads)
    assert workloads["natural_repeat"]["expected_cache_layer"] == "text_exact"
    assert workloads["exact_replay"]["expected_cache_layer"] == "audio_exact"
    assert workloads["speaker_control"]["expected_reuse_allowed"] == "false"
    assert workloads["domain_control"]["domain"] == "medical"
    assert (manifest.with_suffix(".metadata.json")).exists()


def test_matrix_combined_outputs_include_thesis_artifacts(tmp_path):
    rows = [
        {
            "request_index": 1,
            "dataset": "common_voice_ro",
            "strategy": "stateless",
            "workload": "seed",
            "cache": "miss",
            "cache_layer": "miss",
            "correct": True,
            "expected_reuse_allowed": True,
            "expected_cache_layer": "miss",
            "client_total_time": 4.0,
            "server_total_time": 3.9,
            "lookup_time": 0.01,
            "transcript_time": 1.0,
            "inference_time": 3.0,
        },
        {
            "request_index": 2,
            "dataset": "common_voice_ro",
            "strategy": "context",
            "workload": "natural_repeat",
            "cache": "hit",
            "cache_layer": "text_exact",
            "correct": True,
            "expected_reuse_allowed": True,
            "expected_cache_layer": "text_exact",
            "client_total_time": 1.0,
            "server_total_time": 0.9,
            "lookup_time": 0.02,
            "transcript_time": 0.8,
            "inference_time": 0.0,
        },
    ]

    run_matrix.write_combined_outputs(tmp_path, rows, pod_hour_usd=2.0)
    combined = tmp_path / "combined"

    assert (combined / "raw_requests.jsonl").exists()
    assert (combined / "summary_by_dataset.csv").exists()
    assert (combined / "summary_by_workload.csv").exists()
    assert (combined / "cache_confusion_matrix.csv").exists()
    assert (combined / "cost_model.csv").exists()
    assert (combined / "final_evaluation_report.md").exists()
    assert json.loads((combined / "raw_requests.jsonl").read_text(encoding="utf-8").splitlines()[0])["dataset"] == "common_voice_ro"


def test_strategy_and_db_scale_parsers_reject_bad_inputs():
    assert run_matrix.parse_strategies("stateless,context") == ["stateless", "context"]
    assert db_scale.parse_sizes("0,10000,100000") == [0, 10000, 100000]
    assert db_scale.parse_sizes("10000", include_1m=True) == [10000, 1000000]

    try:
        run_matrix.parse_strategies("context,unsafe")
    except SystemExit as exc:
        assert "unsafe" in str(exc)
    else:
        raise AssertionError("expected bad strategy to fail")
