# Polyglot API

This is a FastAPI application that uses the SeamlessM4Tv2ForSpeechToSpeech model from Hugging Face to process audio files.

The project/distribution name is `polyglot-api`. The import package remains `polyglot_api`, which is the standard Python convention because import names cannot contain dashes.

## Features

- Process audio files and return translated audio.
- Preserve the existing Seamless M4T speech-to-speech inference path.
- Optional Postgres + pgvector audio memory for database-augmented caching.
- Logging with Gunicorn and colorlog.
- Dockerized application.
- API documentation with Swagger UI.

## Project Layout

- `src/polyglot_api/app.py`: FastAPI adapter and HTTP response handling.
- `src/polyglot_api/translation_pipeline.py`: audio decoding, memory lookup, inference orchestration, and experiment headers.
- `src/polyglot_api/translator.py`: Seamless M4T model loading and speech-to-speech inference.
- `src/polyglot_api/audio_fingerprint.py`: deterministic audio fingerprint and vector generation.
- `src/polyglot_api/semantic_memory.py`: Postgres + pgvector memory adapter.
- `src/polyglot_api/db_schema.sql`: database schema for sessions, audio vectors, translated audio, provenance, and audit events.
- `scripts/`: model download and benchmark helpers.
- `tests/`: focused unit tests for stable modules.

## Audio Semantic Translation Memory

The API can optionally use Postgres + pgvector as a model-versioned speech-to-speech memory. It works directly on uploaded audio segments:

```text
audio segment
-> audio fingerprint / vector
-> database lookup
-> cache hit: return stored translated audio
-> cache miss: run Seamless M4T and store translated audio
```

The existing `/process` and `/process_memory` endpoints remain audio-in/audio-out compatible. They also accept optional research fields:

- `session_id`
- `source_language`
- `domain`
- `privacy_level`
- `use_semantic_cache`
- `cache_strategy`: `stateless`, `exact`, `semantic`, or `context`

No transcript is required. The v1 fingerprint is deterministic and lightweight, so it is best for exact and near-duplicate audio reuse. A learned speech embedding model can replace it later for stronger semantic generalization.

### Environment

- `POLYGLOT_DATABASE_URL`: Postgres DSN. If omitted, semantic memory is disabled.
- `POLYGLOT_AUTO_INIT_DB`: initialize schema on startup. Default: `true`.
- `POLYGLOT_SEMANTIC_CACHE_ENABLED`: default cache toggle. Default: `false`.
- `POLYGLOT_CACHE_STRATEGY`: default strategy. Default: `context`.
- `POLYGLOT_SIMILARITY_THRESHOLD`: default audio-vector threshold. Default: `0.98`.
- `POLYGLOT_TRANSLATION_MODEL_VERSION`: provenance label for stored translations.
- `POLYGLOT_AUDIO_EMBEDDING_MODEL_VERSION`: provenance label for the audio fingerprint.

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

Responses include experiment headers such as `X-Polyglot-Cache`, `X-Polyglot-Similarity`, `X-Polyglot-Lookup-Time`, and `X-Polyglot-Inference-Time`.

### Benchmark helper

```bash
python scripts/semantic_benchmark.py samples.csv https://localhost --strategy context
```

The CSV must contain:

```csv
audio_path,source_language,target_language,domain
```
