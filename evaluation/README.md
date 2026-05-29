# Polyglot Semantic Memory Evaluation

This directory contains reproducible evaluation assets for the database-augmented speech-to-speech prototype.

## Evaluation Levels

1. **Demo evaluation** in `evaluation/demo/`
   - Small manually recorded WAV set.
   - Shows cache layers visually and reproducibly.
   - Best for presentations.

2. **Benchmark evaluation**
   - Larger manifests built from Common Voice, FLEURS, CoVoST 2, or IWSLT samples.
   - Measures latency, hit rate, safety, storage overhead, and scalability.

## Run The Demo Benchmark

Record the files referenced by `evaluation/demo/manifest.csv`, then run:

```powershell
python scripts/semantic_benchmark.py `
  --manifest evaluation/demo/manifest.csv `
  --base-url https://localhost `
  --strategy context `
  --session-id demo-eval-001 `
  --output-dir evaluation/demo/results `
  --use-transcript-memory true
```

Generated artifacts:

```text
raw_requests.jsonl
summary.csv
summary_by_strategy.csv
summary_by_cache_layer.csv
summary_by_workload.csv
latency_distribution.csv
incorrect_reuse_cases.csv
evaluation_report.md
```

## Recommended Public Datasets

- **Mozilla Common Voice**: speaker variation, accents, repeated/common phrases.
- **FLEURS**: multilingual speech coverage across controlled utterances.
- **CoVoST 2**: speech translation evaluation with translation references.
- **IWSLT speech translation data**: formal speech translation benchmarks.

For thesis-grade safety analysis, add manual labels for whether reuse is allowed.

## Database-Scale Evaluation

If full Seamless inference is too slow for large scale tests, seed synthetic rows and measure database behavior separately:

```powershell
python scripts/seed_semantic_eval_db.py `
  --database-url $env:POLYGLOT_DATABASE_URL `
  --records 10000
```

Then run the benchmark with `--database-url` so `db_stats.csv` includes row counts and storage/index sizes.

## Interpretation Rules

- `audio_exact`: exact audio/hash reuse.
- `text_exact`: naturally repeated speech with the same normalized transcript.
- `text_vector`: conservative transcript-vector reuse; manually inspect false positives.
- `audio_vector`: handcrafted acoustic-fingerprint baseline, not learned semantic speech understanding.
- `miss`: Seamless M4T inference ran and the result may be stored.
