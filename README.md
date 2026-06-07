# Polyglot API

This is a FastAPI application that uses the SeamlessM4Tv2ForSpeechToSpeech model from Hugging Face to process audio files.

The project/distribution name is `polyglot-api`. The import package remains `polyglot_api`, which is the standard Python convention because import names cannot contain dashes.

## Features

- Process audio files and return translated audio.
- Preserve the existing Seamless M4T speech-to-speech inference path.
- Optional Postgres + pgvector audio/transcript memory for database-augmented caching.
- Logging with Gunicorn and colorlog.
- Dockerized application.
- API documentation with Swagger UI.

## Project Layout

- `RUNPOD_DEPLOY.md`: GPU Pod deployment guide for API + local pgvector memory.
- `evaluation/REPRODUCIBILITY.md`: research-grade dataset, RunPod, matrix, DB scale, and report procedure.
- `src/polyglot_api/app.py`: FastAPI adapter and HTTP response handling.
- `src/polyglot_api/translation_pipeline.py`: audio decoding, memory lookup, inference orchestration, and experiment headers.
- `src/polyglot_api/translator.py`: Seamless M4T model loading and speech-to-speech inference.
- `src/polyglot_api/audio_fingerprint.py`: deterministic audio fingerprint and vector generation.
- `src/polyglot_api/text_semantics.py`: transcript normalization, hashing, and deterministic text-vector generation.
- `src/polyglot_api/semantic_memory.py`: Postgres + pgvector memory adapter.
- `src/polyglot_api/db_schema.sql`: database schema for sessions, audio vectors, translated audio, provenance, and audit events.
 - `scripts/`: model download, evaluation, and synthetic DB seeding helpers.
- `evaluation/`: thesis benchmark design and reproducibility notes.
- `tests/`: focused unit tests for stable modules.

## Transcript-Aware Semantic Translation Memory

The API can optionally use Postgres + pgvector as a model-versioned speech-to-speech memory. It keeps the user-facing Seamless M4T speech-to-speech path, but it can extract a transcript first when `source_language` is explicit:

```text
audio segment
-> audio fingerprint + optional Seamless transcript
-> exact audio lookup
-> normalized transcript lookup
-> transcript vector lookup
-> audio vector lookup
-> cache hit: return stored translated audio bytes
-> cache miss: run Seamless M4T and store translated audio
```

The existing `/process` and `/process_memory` endpoints remain audio-in/audio-out compatible. They also accept optional research fields:

- `session_id`
- `source_language`
- `domain`
- `privacy_level`
- `use_semantic_cache`
- `use_transcript_memory`
- `cache_strategy`: `stateless`, `exact`, `semantic`, or `context`

Transcript memory is skipped when `source_language=auto`, so old clients continue to work. For the strongest demo, set an explicit source language such as `ron`. The system normalizes transcripts before hashing, so `câine`, `Caine!`, and `câine.` all map to `caine`.

### Environment

- `POLYGLOT_API_TOKEN`: optional shared token required through `X-Polyglot-Api-Key` when set.
- `POLYGLOT_DATABASE_URL`: Postgres DSN. If omitted, semantic memory is disabled.
- `POLYGLOT_AUTO_INIT_DB`: initialize schema on startup. Default: `true`.
- `POLYGLOT_SEMANTIC_CACHE_ENABLED`: default cache toggle. Default: `false`.
- `POLYGLOT_TRANSCRIPT_MEMORY_ENABLED`: default transcript-memory toggle. Default: `true`.
- `POLYGLOT_CACHE_STRATEGY`: default strategy. Default: `context`.
- `POLYGLOT_SIMILARITY_THRESHOLD`: default audio-vector threshold. Default: `0.98`.
- `POLYGLOT_TEXT_SIMILARITY_THRESHOLD`: default transcript-vector threshold. Default: `0.92`.
- `POLYGLOT_TRANSLATION_MODEL_VERSION`: provenance label for stored translations.
- `POLYGLOT_AUDIO_EMBEDDING_MODEL_VERSION`: provenance label for the audio fingerprint.
- `POLYGLOT_TEXT_EMBEDDING_MODEL_VERSION`: provenance label for text embeddings.

### Local pgvector stack

```bash
docker compose -f docker-compose.semantic.yml up --build
```

### Repeat-audio test

Send the same WAV twice to `/process_memory`. The first request should miss and run Seamless. The second request should hit the database and return with `X-Polyglot-Inference-Time: 0.0000`.

```bash
mkdir -p headers outputs

curl -k -D headers/headers-1.txt -o outputs/out-1.wav \
  -F "file=@../polyglot-tkinter-app/tests/performance/sample2.wav;type=audio/wav" \
  -F "language=eng" \
  -F "speaker_id=0" \
  -F "source_language=ron" \
  -F "session_id=test-audio-cache" \
  -F "domain=demo" \
  -F "privacy_level=transient" \
  -F "use_semantic_cache=true" \
  -F "cache_strategy=context" \
  https://localhost/process_memory
```

```bash
curl -k -D headers/headers-2.txt -o outputs/out-2.wav \
  -F "file=@../polyglot-tkinter-app/tests/performance/sample2.wav;type=audio/wav" \
  -F "language=eng" \
  -F "speaker_id=0" \
  -F "source_language=ron" \
  -F "session_id=test-audio-cache" \
  -F "domain=demo" \
  -F "privacy_level=transient" \
  -F "use_semantic_cache=true" \
  -F "cache_strategy=context" \
  https://localhost/process_memory
```

Responses include experiment headers such as `X-Polyglot-Cache`, `X-Polyglot-Cache-Layer`, `X-Polyglot-Source-Transcript`, `X-Polyglot-Normalized-Text`, `X-Polyglot-Text-Similarity`, `X-Polyglot-Lookup-Time`, `X-Polyglot-Transcript-Time`, and `X-Polyglot-Inference-Time`. Transcript text headers are UTF-8 percent-encoded for HTTP safety; the Tkinter client decodes them before display.

### Real speech demo

For a live demo, use the Tkinter app in Demo Mode or send two different recordings with the same spoken word:

1. Set `source_language=ron`, `language=eng`, `use_semantic_cache=true`, and `use_transcript_memory=true`.
2. Say `câine` once. The first request should miss, run Seamless, and store translated audio.
3. Say `câine` again naturally. The second request can hit `X-Polyglot-Cache-Layer: text_exact` even when the waveform is not identical.

### Evaluation runners

For thesis-grade evaluation, prepare public dataset manifests and run the strategy matrix:

```bash
python scripts/prepare_evaluation_data.py all \
  --output-root evaluation \
  --source-dir /path/to/common-voice-ro-25.0

python scripts/run_evaluation_matrix.py \
  --manifest evaluation/manifests/common_voice_ro_manifest.csv \
  --manifest evaluation/manifests/fleurs_manifest.csv \
  --manifest evaluation/manifests/covost2_manifest.csv \
  --base-url http://127.0.0.1:8000 \
  --output-dir evaluation/runpod/results
```

Full local and RunPod commands are in `evaluation/REPRODUCIBILITY.md`.

For database-scale tests that do not require thousands of Seamless inferences, seed synthetic records:

```bash
python scripts/seed_semantic_eval_db.py \
  --database-url "$POLYGLOT_DATABASE_URL" \
  --records 10000
```
