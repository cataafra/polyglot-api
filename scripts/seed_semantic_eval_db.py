"""
Seed synthetic semantic-memory records for database-scale evaluation.

This is for DB lookup/storage benchmarks, not translation-quality evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyglot_api.audio_fingerprint import _normalize_vector  # noqa: E402
from polyglot_api.semantic_memory import vector_literal  # noqa: E402
from polyglot_api.text_semantics import build_text_embedding, hash_normalized_text  # noqa: E402


def synthetic_audio_embedding(index: int, dimensions: int = 384) -> list[float]:
    values = []
    seed = hashlib.sha256(f"audio:{index}".encode("utf-8")).digest()
    for offset in range(dimensions):
        byte = seed[offset % len(seed)]
        values.append((byte / 255.0) - 0.5)
    return _normalize_vector(values, dimensions)


def seed_database(database_url: str, records: int, batch_size: int) -> None:
    import psycopg

    schema_sql = (ROOT / "src" / "polyglot_api" / "db_schema.sql").read_text(encoding="utf-8")
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            for statement in schema_sql.split(";"):
                statement = statement.strip()
                if statement:
                    cur.execute(statement)
            conn.commit()

        inserted = 0
        while inserted < records:
            batch_count = min(batch_size, records - inserted)
            with conn.cursor() as cur:
                for batch_offset in range(batch_count):
                    index = inserted + batch_offset
                    session_id = uuid.uuid5(uuid.NAMESPACE_URL, f"polyglot:synthetic:{index // 20}")
                    segment_id = uuid.uuid4()
                    translation_id = uuid.uuid4()
                    audio_embedding_id = uuid.uuid4()
                    text_embedding_id = uuid.uuid4()
                    external_session = f"synthetic-{index // 20}"
                    normalized_text = f"synthetic phrase {index % 1000}"
                    source_text_hash = hash_normalized_text(normalized_text)
                    audio_hash = hashlib.sha256(f"audio:{index}".encode("utf-8")).hexdigest()
                    domain = "legal" if index % 2 == 0 else "general"
                    speaker_id = str(index % 4)

                    cur.execute(
                        """
                        INSERT INTO sessions (
                            id,
                            external_session_id,
                            anonymized_user_id,
                            domain,
                            privacy_level,
                            retention_expires_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (external_session_id) DO NOTHING
                        """,
                        (
                            session_id,
                            external_session,
                            hashlib.sha256(external_session.encode("utf-8")).hexdigest(),
                            domain,
                            "transient",
                            datetime.now(timezone.utc) + timedelta(days=30),
                        ),
                    )
                    cur.execute(
                        """
                        INSERT INTO audio_segments (
                            id,
                            session_id,
                            source_language,
                            source_audio_hash,
                            source_transcript,
                            normalized_source_text,
                            source_text_hash,
                            transcript_model_name,
                            transcript_model_version,
                            duration_seconds,
                            input_samplerate,
                            channels,
                            speaker_id,
                            domain,
                            privacy_level
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            segment_id,
                            session_id,
                            "ron",
                            audio_hash,
                            normalized_text,
                            normalized_text,
                            source_text_hash,
                            "synthetic-transcript",
                            "v1",
                            1.0 + (index % 10) / 10.0,
                            16000,
                            1,
                            speaker_id,
                            domain,
                            "transient",
                        ),
                    )
                    cur.execute(
                        """
                        INSERT INTO audio_translations (
                            id,
                            audio_segment_id,
                            target_language,
                            translated_audio,
                            output_samplerate,
                            speaker_id,
                            translation_model_name,
                            translation_model_version
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            translation_id,
                            segment_id,
                            "eng",
                            b"RIFFsyntheticWAVE",
                            16000,
                            speaker_id,
                            "synthetic-translation",
                            "v1",
                        ),
                    )
                    cur.execute(
                        """
                        INSERT INTO audio_embeddings (
                            id,
                            audio_segment_id,
                            embedding_model_name,
                            embedding_model_version,
                            source_audio_hash,
                            embedding
                        )
                        VALUES (%s, %s, %s, %s, %s, %s::vector)
                        """,
                        (
                            audio_embedding_id,
                            segment_id,
                            "synthetic-audio",
                            "v1",
                            audio_hash,
                            vector_literal(synthetic_audio_embedding(index)),
                        ),
                    )
                    cur.execute(
                        """
                        INSERT INTO text_embeddings (
                            id,
                            audio_segment_id,
                            embedding_model_name,
                            embedding_model_version,
                            source_text_hash,
                            embedding
                        )
                        VALUES (%s, %s, %s, %s, %s, %s::vector)
                        """,
                        (
                            text_embedding_id,
                            segment_id,
                            "polyglot-char-ngram-text",
                            "v1",
                            source_text_hash,
                            vector_literal(build_text_embedding(normalized_text)),
                        ),
                    )
                cur.execute(
                    """
                    INSERT INTO audit_events (event_type, details)
                    VALUES (%s, %s)
                    """,
                    (
                        "synthetic_seed",
                        json.dumps({"records": batch_count, "start_index": inserted}),
                    ),
                )
            conn.commit()
            inserted += batch_count
            print(f"seeded {inserted}/{records}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed synthetic semantic-memory rows.")
    parser.add_argument("--database-url", default=os.getenv("POLYGLOT_DATABASE_URL"), required=False)
    parser.add_argument("--records", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    if not args.database_url:
        raise SystemExit("--database-url or POLYGLOT_DATABASE_URL is required")
    seed_database(args.database_url, records=max(1, args.records), batch_size=max(1, args.batch_size))


if __name__ == "__main__":
    main()
