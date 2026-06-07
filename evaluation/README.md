# Polyglot Evaluation Suite

This directory is the evaluation layer for the hybrid-db semantic memory thesis chapter.

## Structure

- `manifests/`: generated CSV manifests for public datasets. Ignored by git.
- `corpora/`: generated WAV clips from public datasets. Ignored by git.
- `runpod/results/`: generated research artifacts. Ignored by git.
- `REPRODUCIBILITY.md`: exact commands for dataset preparation, RunPod execution, DB scale measurement, and final artifacts.

## Research Design

The final evaluation compares four strategies:

- `stateless`: semantic layer disabled; establishes inference-only latency/cost.
- `exact`: exact audio and normalized transcript replay.
- `semantic`: vector reuse without full context; useful for measuring possible unsafe reuse.
- `context`: full hybrid-db semantic layer with source/target/speaker/domain/privacy constraints.

Dataset roles:

- Common Voice Romanian 25.0: cache reuse and safety, especially repeated normalized sentences, exact replay, speaker controls, domain controls, and near-text negatives.
- FLEURS: Romanian-to-English quality and multilingual generalization to English.
- CoVoST 2: external speech-translation validation.

## Outputs

The thesis run writes:

```text
raw_requests.jsonl
summary_by_strategy.csv
summary_by_dataset.csv
summary_by_cache_layer.csv
summary_by_workload.csv
latency_distribution.csv
cache_confusion_matrix.csv
quality_metrics.csv
cost_model.csv
threshold_sweep.csv
incorrect_reuse_cases.csv
db_scale.csv
plots/*.png
final_evaluation_report.md
```

See `REPRODUCIBILITY.md` for the complete step-by-step procedure.
