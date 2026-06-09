import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .audio_fingerprint import AudioFingerprint
from .text_semantics import (
    DEFAULT_TEXT_EMBEDDING_MODEL,
    DEFAULT_TEXT_EMBEDDING_VERSION,
    TextFingerprint,
)

logger = logging.getLogger(__name__)


DEFAULT_TRANSLATION_MODEL = "facebook/seamless-m4t-v2-large"
DEFAULT_MODEL_VERSION = "local-or-huggingface"
DEFAULT_AUDIO_EMBEDDING_MODEL = "polyglot-audio-fingerprint"
DEFAULT_AUDIO_EMBEDDING_VERSION = "v1"
ALLOWED_PRIVACY_LEVELS = {"transient", "private", "internal", "public"}


@dataclass
class SemanticMemoryMetadata:
    session_id: str
    source_language: str
    target_language: str
    speaker_id: str
    domain: str
    privacy_level: str
    cache_strategy: str
    use_semantic_cache: bool
    use_transcript_memory: bool


@dataclass
class CacheLookupResult:
    hit: bool
    strategy: str
    decision_reason: str
    similarity: Optional[float] = None
    text_similarity: Optional[float] = None
    translation_id: Optional[str] = None
    audio_bytes: Optional[bytes] = None
    output_samplerate: Optional[int] = None
    cache_layer: str = "miss"
    source_transcript: Optional[str] = None
    normalized_source_text: Optional[str] = None


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def anonymize_identifier(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    salt = os.getenv("POLYGLOT_PRIVACY_SALT", "polyglot-research")
    return hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()


def normalize_privacy_level(value: Optional[str]) -> str:
    privacy_level = (value or "transient").strip().lower()
    if privacy_level not in ALLOWED_PRIVACY_LEVELS:
        return "transient"
    return privacy_level


def retention_expiry(privacy_level: str) -> datetime:
    retention_days = {
        "transient": int(os.getenv("POLYGLOT_RETENTION_TRANSIENT_DAYS", "1")),
        "private": int(os.getenv("POLYGLOT_RETENTION_PRIVATE_DAYS", "7")),
        "internal": int(os.getenv("POLYGLOT_RETENTION_INTERNAL_DAYS", "30")),
        "public": int(os.getenv("POLYGLOT_RETENTION_PUBLIC_DAYS", "365")),
    }
    days = retention_days[normalize_privacy_level(privacy_level)]
    return datetime.now(timezone.utc) + timedelta(days=days)


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


class NullSemanticMemory:
    enabled = False

    def initialize(self) -> None:
        return None

    def lookup(
        self,
        metadata: SemanticMemoryMetadata,
        fingerprint: AudioFingerprint,
        text_fingerprint: Optional[TextFingerprint] = None,
    ) -> CacheLookupResult:
        return CacheLookupResult(
            hit=False,
            strategy="disabled",
            decision_reason="semantic memory is disabled",
        )

    def lookup_audio_exact(
        self,
        metadata: SemanticMemoryMetadata,
        fingerprint: AudioFingerprint,
    ) -> CacheLookupResult:
        return CacheLookupResult(False, "disabled", "semantic memory is disabled")

    def store(
        self,
        metadata: SemanticMemoryMetadata,
        fingerprint: AudioFingerprint,
        translated_audio: bytes,
        output_samplerate: int,
        text_fingerprint: Optional[TextFingerprint] = None,
    ) -> None:
        return None

    def expire_old_records(self) -> int:
        return 0

    def health(self) -> Dict[str, Any]:
        return {"enabled": False, "status": False}


class PostgresSemanticMemory:
    enabled = True

    def __init__(
        self,
        dsn: str,
        embedding_dimensions: int = 384,
        similarity_threshold: float = 0.98,
        text_similarity_threshold: float = 0.92,
        translation_model_name: str = DEFAULT_TRANSLATION_MODEL,
        translation_model_version: str = DEFAULT_MODEL_VERSION,
        audio_embedding_model_name: str = DEFAULT_AUDIO_EMBEDDING_MODEL,
        audio_embedding_model_version: str = DEFAULT_AUDIO_EMBEDDING_VERSION,
        text_embedding_model_name: str = DEFAULT_TEXT_EMBEDDING_MODEL,
        text_embedding_model_version: str = DEFAULT_TEXT_EMBEDDING_VERSION,
        transcript_model_name: str = DEFAULT_TRANSLATION_MODEL,
        transcript_model_version: str = DEFAULT_MODEL_VERSION,
    ):
        self.dsn = dsn
        self.embedding_dimensions = embedding_dimensions
        self.similarity_threshold = similarity_threshold
        self.text_similarity_threshold = text_similarity_threshold
        self.translation_model_name = translation_model_name
        self.translation_model_version = translation_model_version
        self.audio_embedding_model_name = audio_embedding_model_name
        self.audio_embedding_model_version = audio_embedding_model_version
        self.text_embedding_model_name = text_embedding_model_name
        self.text_embedding_model_version = text_embedding_model_version
        self.transcript_model_name = transcript_model_name
        self.transcript_model_version = transcript_model_version
        self._psycopg = None
        self._dict_row = None

    def _load_driver(self) -> bool:
        if self._psycopg is not None:
            return True
        try:
            import psycopg
            from psycopg.rows import dict_row

            self._psycopg = psycopg
            self._dict_row = dict_row
            return True
        except ImportError:
            logger.warning("psycopg is not installed; semantic memory is unavailable")
            self.enabled = False
            return False

    def _connect(self):
        if not self._load_driver():
            raise RuntimeError("psycopg is not installed")
        return self._psycopg.connect(self.dsn, row_factory=self._dict_row)

    def initialize(self) -> None:
        if not self._load_driver():
            return
        schema_path = Path(__file__).with_name("db_schema.sql")
        schema_sql = schema_path.read_text(encoding="utf-8")
        with self._connect() as conn:
            with conn.cursor() as cur:
                for statement in schema_sql.split(";"):
                    statement = statement.strip()
                    if statement:
                        cur.execute(statement)
                self._ensure_model_versions(cur)
            conn.commit()
        logger.info("Audio semantic memory schema initialized")

    def _ensure_model_versions(self, cur) -> None:
        versions = [
            ("translation", self.translation_model_name, self.translation_model_version),
            ("audio_embedding", self.audio_embedding_model_name, self.audio_embedding_model_version),
            ("text_embedding", self.text_embedding_model_name, self.text_embedding_model_version),
            ("transcript", self.transcript_model_name, self.transcript_model_version),
        ]
        for model_type, model_name, version in versions:
            cur.execute(
                """
                INSERT INTO model_versions (model_type, model_name, version)
                VALUES (%s, %s, %s)
                ON CONFLICT (model_type, model_name, version) DO NOTHING
                """,
                (model_type, model_name, version),
            )

    def health(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "status": False}
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 AS ok")
                    row = cur.fetchone()
            return {
                "enabled": True,
                "status": bool(row and row["ok"] == 1),
                "mode": "audio+transcript",
                "transcript_memory": True,
                "similarity_threshold": self.similarity_threshold,
                "text_similarity_threshold": self.text_similarity_threshold,
                "embedding_dimensions": self.embedding_dimensions,
                "embedding_model": self.audio_embedding_model_name,
                "embedding_version": self.audio_embedding_model_version,
                "text_embedding_model": self.text_embedding_model_name,
                "text_embedding_version": self.text_embedding_model_version,
            }
        except Exception as exc:
            logger.warning("Semantic memory health check failed: %s", exc)
            return {"enabled": True, "status": False, "error": "semantic memory health check failed"}

    def lookup(
        self,
        metadata: SemanticMemoryMetadata,
        fingerprint: AudioFingerprint,
        text_fingerprint: Optional[TextFingerprint] = None,
    ) -> CacheLookupResult:
        if not self.enabled:
            return CacheLookupResult(False, "disabled", "semantic memory disabled")
        if not metadata.use_semantic_cache:
            return CacheLookupResult(False, "stateless", "semantic cache disabled by request")

        strategy = (metadata.cache_strategy or "context").lower()
        if strategy == "stateless":
            return CacheLookupResult(False, "stateless", "stateless baseline selected")

        try:
            exact = self._lookup_exact(metadata, fingerprint)
            if exact.hit:
                return exact
            if strategy == "exact":
                return exact

            text_lookup_enabled = self._can_use_text_memory(metadata, text_fingerprint)
            if text_lookup_enabled:
                text_exact = self._lookup_text_exact(metadata, text_fingerprint)
                if text_exact.hit:
                    return text_exact

            if strategy not in {"semantic", "context"}:
                return CacheLookupResult(False, strategy, "unknown cache strategy")

            if text_lookup_enabled:
                text_vector = self._lookup_text_vector(
                    metadata=metadata,
                    text_fingerprint=text_fingerprint,
                    context_aware=strategy == "context",
                )
                if text_vector.hit:
                    return text_vector

            return self._lookup_vector(
                metadata=metadata,
                fingerprint=fingerprint,
                context_aware=strategy == "context",
            )
        except Exception as exc:
            logger.exception("Audio semantic lookup failed")
            return CacheLookupResult(False, strategy, f"lookup error: {exc}")

    def lookup_audio_exact(
        self,
        metadata: SemanticMemoryMetadata,
        fingerprint: AudioFingerprint,
    ) -> CacheLookupResult:
        if not self.enabled:
            return CacheLookupResult(False, "disabled", "semantic memory disabled")
        if not metadata.use_semantic_cache:
            return CacheLookupResult(False, "stateless", "semantic cache disabled by request")
        if (metadata.cache_strategy or "").lower() == "stateless":
            return CacheLookupResult(False, "stateless", "stateless baseline selected")
        try:
            return self._lookup_exact(metadata, fingerprint)
        except Exception as exc:
            logger.exception("Audio exact semantic lookup failed")
            return CacheLookupResult(False, "exact", f"audio exact lookup error: {exc}")

    @staticmethod
    def _can_use_text_memory(
        metadata: SemanticMemoryMetadata,
        text_fingerprint: Optional[TextFingerprint],
    ) -> bool:
        return bool(
            metadata.use_transcript_memory
            and metadata.source_language
            and metadata.source_language.lower() != "auto"
            and text_fingerprint
            and text_fingerprint.normalized_text
        )

    def _lookup_exact(
        self,
        metadata: SemanticMemoryMetadata,
        fingerprint: AudioFingerprint,
    ) -> CacheLookupResult:
        query = """
            SELECT
                at.id::text AS translation_id,
                at.translated_audio,
                at.output_samplerate
            FROM audio_translations at
            JOIN audio_segments aseg ON aseg.id = at.audio_segment_id
            JOIN sessions s ON s.id = aseg.session_id
            WHERE aseg.source_audio_hash = %s
              AND aseg.source_language = %s
              AND at.target_language = %s
              AND at.speaker_id = %s
              AND at.translation_model_name = %s
              AND at.translation_model_version = %s
              AND aseg.domain = %s
              AND aseg.privacy_level = %s
              AND s.retention_expires_at > NOW()
            ORDER BY at.created_at DESC
            LIMIT 1
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        fingerprint.audio_hash,
                        metadata.source_language,
                        metadata.target_language,
                        metadata.speaker_id,
                        self.translation_model_name,
                        self.translation_model_version,
                        metadata.domain,
                        metadata.privacy_level,
                    ),
                )
                row = cur.fetchone()

        if not row:
            self._audit(None, "audio_exact_miss", {"audio_hash": fingerprint.audio_hash})
            return CacheLookupResult(False, "exact", "no exact audio hash match")

        self._audit(None, "audio_exact_hit", {"translation_id": row["translation_id"]})
        return CacheLookupResult(
            hit=True,
            strategy="exact",
            decision_reason="exact audio fingerprint match",
            similarity=1.0,
            translation_id=row["translation_id"],
            audio_bytes=bytes(row["translated_audio"]) if row["translated_audio"] else None,
            output_samplerate=row["output_samplerate"],
            cache_layer="audio_exact",
        )

    def _lookup_text_exact(
        self,
        metadata: SemanticMemoryMetadata,
        text_fingerprint: TextFingerprint,
    ) -> CacheLookupResult:
        query = """
            SELECT
                at.id::text AS translation_id,
                at.translated_audio,
                at.output_samplerate,
                aseg.source_transcript,
                aseg.normalized_source_text
            FROM audio_translations at
            JOIN audio_segments aseg ON aseg.id = at.audio_segment_id
            JOIN sessions s ON s.id = aseg.session_id
            WHERE aseg.source_text_hash = %s
              AND aseg.source_language = %s
              AND at.target_language = %s
              AND at.speaker_id = %s
              AND at.translation_model_name = %s
              AND at.translation_model_version = %s
              AND aseg.transcript_model_name = %s
              AND aseg.transcript_model_version = %s
              AND aseg.domain = %s
              AND aseg.privacy_level = %s
              AND s.retention_expires_at > NOW()
            ORDER BY at.created_at DESC
            LIMIT 1
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        text_fingerprint.text_hash,
                        metadata.source_language,
                        metadata.target_language,
                        metadata.speaker_id,
                        self.translation_model_name,
                        self.translation_model_version,
                        self.transcript_model_name,
                        self.transcript_model_version,
                        metadata.domain,
                        metadata.privacy_level,
                    ),
                )
                row = cur.fetchone()

        if not row:
            self._audit(
                None,
                "text_exact_miss",
                {"source_text_hash": text_fingerprint.text_hash},
            )
            return CacheLookupResult(
                False,
                "exact",
                "no normalized transcript match",
                cache_layer="miss",
                source_transcript=text_fingerprint.source_transcript,
                normalized_source_text=text_fingerprint.normalized_text,
            )

        self._audit(None, "text_exact_hit", {"translation_id": row["translation_id"]})
        return CacheLookupResult(
            hit=True,
            strategy="exact",
            decision_reason="normalized transcript match",
            text_similarity=1.0,
            translation_id=row["translation_id"],
            audio_bytes=bytes(row["translated_audio"]) if row["translated_audio"] else None,
            output_samplerate=row["output_samplerate"],
            cache_layer="text_exact",
            source_transcript=row["source_transcript"] or text_fingerprint.source_transcript,
            normalized_source_text=row["normalized_source_text"] or text_fingerprint.normalized_text,
        )

    def _lookup_text_vector(
        self,
        metadata: SemanticMemoryMetadata,
        text_fingerprint: TextFingerprint,
        context_aware: bool,
    ) -> CacheLookupResult:
        filters = [
            "aseg.source_language = %(source_language)s",
            "at.target_language = %(target_language)s",
            "at.speaker_id = %(speaker_id)s",
            "at.translation_model_name = %(translation_model_name)s",
            "at.translation_model_version = %(translation_model_version)s",
            "aseg.transcript_model_name = %(transcript_model_name)s",
            "aseg.transcript_model_version = %(transcript_model_version)s",
            "te.embedding_model_name = %(embedding_model_name)s",
            "te.embedding_model_version = %(embedding_model_version)s",
            "aseg.privacy_level = %(privacy_level)s",
            "s.retention_expires_at > NOW()",
        ]
        if context_aware:
            filters.extend(
                [
                    "aseg.domain = %(domain)s",
                ]
            )

        query = f"""
            SELECT
                at.id::text AS translation_id,
                at.translated_audio,
                at.output_samplerate,
                aseg.source_transcript,
                aseg.normalized_source_text,
                1 - (te.embedding <=> %(embedding)s::vector) AS similarity
            FROM text_embeddings te
            JOIN audio_segments aseg ON aseg.id = te.audio_segment_id
            JOIN sessions s ON s.id = aseg.session_id
            JOIN audio_translations at ON at.audio_segment_id = aseg.id
            WHERE {' AND '.join(filters)}
            ORDER BY te.embedding <=> %(embedding)s::vector
            LIMIT 1
        """
        params = {
            "source_language": metadata.source_language,
            "target_language": metadata.target_language,
            "speaker_id": metadata.speaker_id,
            "translation_model_name": self.translation_model_name,
            "translation_model_version": self.translation_model_version,
            "transcript_model_name": self.transcript_model_name,
            "transcript_model_version": self.transcript_model_version,
            "embedding_model_name": self.text_embedding_model_name,
            "embedding_model_version": self.text_embedding_model_version,
            "domain": metadata.domain,
            "privacy_level": metadata.privacy_level,
            "embedding": vector_literal(text_fingerprint.embedding),
        }
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                row = cur.fetchone()

        strategy = "context" if context_aware else "semantic"
        if not row:
            self._audit(None, f"text_{strategy}_miss", {"reason": "no text vector candidates"})
            return CacheLookupResult(
                False,
                strategy,
                "no text vector candidates",
                cache_layer="miss",
                source_transcript=text_fingerprint.source_transcript,
                normalized_source_text=text_fingerprint.normalized_text,
            )

        similarity = float(row["similarity"])
        if similarity < self.text_similarity_threshold:
            self._audit(
                None,
                f"text_{strategy}_reject",
                {
                    "translation_id": row["translation_id"],
                    "similarity": similarity,
                    "threshold": self.text_similarity_threshold,
                },
            )
            return CacheLookupResult(
                False,
                strategy,
                "best text vector candidate below threshold",
                text_similarity=similarity,
                translation_id=row["translation_id"],
                cache_layer="miss",
                source_transcript=text_fingerprint.source_transcript,
                normalized_source_text=text_fingerprint.normalized_text,
            )

        self._audit(
            None,
            f"text_{strategy}_hit",
            {"translation_id": row["translation_id"], "similarity": similarity},
        )
        return CacheLookupResult(
            hit=True,
            strategy=strategy,
            decision_reason="text vector candidate accepted",
            text_similarity=similarity,
            translation_id=row["translation_id"],
            audio_bytes=bytes(row["translated_audio"]) if row["translated_audio"] else None,
            output_samplerate=row["output_samplerate"],
            cache_layer="text_vector",
            source_transcript=row["source_transcript"] or text_fingerprint.source_transcript,
            normalized_source_text=row["normalized_source_text"] or text_fingerprint.normalized_text,
        )

    def _lookup_vector(
        self,
        metadata: SemanticMemoryMetadata,
        fingerprint: AudioFingerprint,
        context_aware: bool,
    ) -> CacheLookupResult:
        filters = [
            "aseg.source_language = %(source_language)s",
            "at.target_language = %(target_language)s",
            "at.speaker_id = %(speaker_id)s",
            "at.translation_model_name = %(translation_model_name)s",
            "at.translation_model_version = %(translation_model_version)s",
            "ae.embedding_model_name = %(embedding_model_name)s",
            "ae.embedding_model_version = %(embedding_model_version)s",
            "aseg.privacy_level = %(privacy_level)s",
            "s.retention_expires_at > NOW()",
        ]
        if context_aware:
            filters.extend(
                [
                    "aseg.domain = %(domain)s",
                ]
            )

        query = f"""
            SELECT
                at.id::text AS translation_id,
                at.translated_audio,
                at.output_samplerate,
                1 - (ae.embedding <=> %(embedding)s::vector) AS similarity
            FROM audio_embeddings ae
            JOIN audio_segments aseg ON aseg.id = ae.audio_segment_id
            JOIN sessions s ON s.id = aseg.session_id
            JOIN audio_translations at ON at.audio_segment_id = aseg.id
            WHERE {' AND '.join(filters)}
            ORDER BY ae.embedding <=> %(embedding)s::vector
            LIMIT 1
        """
        params = {
            "source_language": metadata.source_language,
            "target_language": metadata.target_language,
            "speaker_id": metadata.speaker_id,
            "translation_model_name": self.translation_model_name,
            "translation_model_version": self.translation_model_version,
            "embedding_model_name": self.audio_embedding_model_name,
            "embedding_model_version": self.audio_embedding_model_version,
            "domain": metadata.domain,
            "privacy_level": metadata.privacy_level,
            "embedding": vector_literal(fingerprint.embedding),
        }
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                row = cur.fetchone()

        strategy = "context" if context_aware else "semantic"
        if not row:
            self._audit(None, f"audio_{strategy}_miss", {"reason": "no vector candidates"})
            return CacheLookupResult(False, strategy, "no audio vector candidates")

        similarity = float(row["similarity"])
        if similarity < self.similarity_threshold:
            self._audit(
                None,
                f"audio_{strategy}_reject",
                {
                    "translation_id": row["translation_id"],
                    "similarity": similarity,
                    "threshold": self.similarity_threshold,
                },
            )
            return CacheLookupResult(
                False,
                strategy,
                "best audio vector candidate below threshold",
                similarity=similarity,
                translation_id=row["translation_id"],
            )

        self._audit(
            None,
            f"audio_{strategy}_hit",
            {"translation_id": row["translation_id"], "similarity": similarity},
        )
        return CacheLookupResult(
            hit=True,
            strategy=strategy,
            decision_reason="audio vector candidate accepted",
            similarity=similarity,
            translation_id=row["translation_id"],
            audio_bytes=bytes(row["translated_audio"]) if row["translated_audio"] else None,
            output_samplerate=row["output_samplerate"],
            cache_layer="audio_vector",
        )

    def store(
        self,
        metadata: SemanticMemoryMetadata,
        fingerprint: AudioFingerprint,
        translated_audio: bytes,
        output_samplerate: int,
        text_fingerprint: Optional[TextFingerprint] = None,
    ) -> None:
        if not self.enabled:
            return

        session_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"polyglot:{metadata.session_id}")
        audio_segment_uuid = uuid.uuid4()
        audio_embedding_uuid = uuid.uuid4()
        text_embedding_uuid = uuid.uuid4()
        translation_uuid = uuid.uuid4()
        transcript_model_name = self.transcript_model_name if text_fingerprint else None
        transcript_model_version = self.transcript_model_version if text_fingerprint else None

        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
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
                        ON CONFLICT (external_session_id)
                        DO UPDATE SET
                            domain = EXCLUDED.domain,
                            privacy_level = EXCLUDED.privacy_level,
                            retention_expires_at = GREATEST(
                                sessions.retention_expires_at,
                                EXCLUDED.retention_expires_at
                            )
                        """,
                        (
                            session_uuid,
                            metadata.session_id,
                            anonymize_identifier(metadata.session_id),
                            metadata.domain,
                            metadata.privacy_level,
                            retention_expiry(metadata.privacy_level),
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
                        ON CONFLICT DO NOTHING
                        RETURNING id
                        """,
                        (
                            audio_segment_uuid,
                            session_uuid,
                            metadata.source_language,
                            fingerprint.audio_hash,
                            text_fingerprint.source_transcript if text_fingerprint else None,
                            text_fingerprint.normalized_text if text_fingerprint else None,
                            text_fingerprint.text_hash if text_fingerprint else None,
                            transcript_model_name,
                            transcript_model_version,
                            fingerprint.duration_seconds,
                            fingerprint.sample_rate,
                            fingerprint.channels,
                            metadata.speaker_id,
                            metadata.domain,
                            metadata.privacy_level,
                        ),
                    )
                    segment_row = cur.fetchone()
                    if segment_row:
                        audio_segment_uuid = segment_row["id"]
                    else:
                        cur.execute(
                            """
                            SELECT id
                            FROM audio_segments
                            WHERE session_id = %s
                              AND source_language = %s
                              AND source_audio_hash = %s
                              AND speaker_id = %s
                              AND domain = %s
                              AND privacy_level = %s
                              AND COALESCE(transcript_model_name, '') = COALESCE(%s, '')
                              AND COALESCE(transcript_model_version, '') = COALESCE(%s, '')
                            ORDER BY created_at DESC
                            LIMIT 1
                            """,
                            (
                                session_uuid,
                                metadata.source_language,
                                fingerprint.audio_hash,
                                metadata.speaker_id,
                                metadata.domain,
                                metadata.privacy_level,
                                transcript_model_name,
                                transcript_model_version,
                            ),
                        )
                        segment_row = cur.fetchone()
                        if segment_row:
                            audio_segment_uuid = segment_row["id"]
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
                        ON CONFLICT (
                            audio_segment_id,
                            target_language,
                            speaker_id,
                            translation_model_name,
                            translation_model_version
                        )
                        DO UPDATE SET
                            translated_audio = EXCLUDED.translated_audio,
                            output_samplerate = EXCLUDED.output_samplerate
                        RETURNING id
                        """,
                        (
                            translation_uuid,
                            audio_segment_uuid,
                            metadata.target_language,
                            translated_audio,
                            output_samplerate,
                            metadata.speaker_id,
                            self.translation_model_name,
                            self.translation_model_version,
                        ),
                    )
                    translation_row = cur.fetchone()
                    if translation_row:
                        translation_uuid = translation_row["id"]
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
                        ON CONFLICT (
                            audio_segment_id,
                            embedding_model_name,
                            embedding_model_version
                        )
                        DO UPDATE SET
                            source_audio_hash = EXCLUDED.source_audio_hash,
                            embedding = EXCLUDED.embedding
                        """,
                        (
                            audio_embedding_uuid,
                            audio_segment_uuid,
                            self.audio_embedding_model_name,
                            self.audio_embedding_model_version,
                            fingerprint.audio_hash,
                            vector_literal(fingerprint.embedding),
                        ),
                    )
                    if text_fingerprint:
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
                            ON CONFLICT (
                                audio_segment_id,
                                embedding_model_name,
                                embedding_model_version
                            )
                            DO UPDATE SET
                                source_text_hash = EXCLUDED.source_text_hash,
                                embedding = EXCLUDED.embedding
                            """,
                            (
                                text_embedding_uuid,
                                audio_segment_uuid,
                                self.text_embedding_model_name,
                                self.text_embedding_model_version,
                                text_fingerprint.text_hash,
                                vector_literal(text_fingerprint.embedding),
                            ),
                        )
                    cur.execute(
                        """
                        INSERT INTO audit_events (session_id, event_type, details)
                        VALUES (%s, %s, %s)
                        """,
                        (
                            session_uuid,
                            "store_audio_translation",
                            json.dumps(
                                {
                                    "translation_id": str(translation_uuid),
                                    "audio_hash": fingerprint.audio_hash,
                                    "source_text_hash": text_fingerprint.text_hash if text_fingerprint else None,
                                    "has_transcript": text_fingerprint is not None,
                                    "cache_strategy": metadata.cache_strategy,
                                    "target_language": metadata.target_language,
                                    "duration_seconds": fingerprint.duration_seconds,
                                }
                            ),
                        ),
                    )
                conn.commit()
        except Exception:
            logger.exception("Failed to store audio semantic memory")

    def expire_old_records(self) -> int:
        if not self.enabled:
            return 0
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM sessions
                    WHERE retention_expires_at <= NOW()
                    RETURNING id
                    """
                )
                deleted = cur.fetchall()
                cur.execute(
                    """
                    INSERT INTO audit_events (event_type, details)
                    VALUES (%s, %s)
                    """,
                    (
                        "retention_expiration",
                        json.dumps({"deleted_sessions": len(deleted)}),
                    ),
                )
            conn.commit()
        return len(deleted)

    def _audit(
        self,
        session_id: Optional[str],
        event_type: str,
        details: Dict[str, Any],
    ) -> None:
        if not self.enabled:
            return
        session_uuid = None
        if session_id:
            session_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"polyglot:{session_id}")
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO audit_events (session_id, event_type, details)
                        VALUES (%s, %s, %s)
                        """,
                        (session_uuid, event_type, json.dumps(details)),
                    )
                conn.commit()
        except Exception:
            logger.debug("Failed to write semantic memory audit event", exc_info=True)


def build_semantic_memory() -> Union[NullSemanticMemory, PostgresSemanticMemory]:
    dsn = os.getenv("POLYGLOT_DATABASE_URL")
    if not dsn:
        return NullSemanticMemory()

    memory = PostgresSemanticMemory(
        dsn=dsn,
        embedding_dimensions=int(os.getenv("POLYGLOT_EMBEDDING_DIMENSIONS", "384")),
        similarity_threshold=float(os.getenv("POLYGLOT_SIMILARITY_THRESHOLD", "0.98")),
        text_similarity_threshold=float(os.getenv("POLYGLOT_TEXT_SIMILARITY_THRESHOLD", "0.92")),
        translation_model_name=os.getenv("POLYGLOT_TRANSLATION_MODEL", DEFAULT_TRANSLATION_MODEL),
        translation_model_version=os.getenv("POLYGLOT_TRANSLATION_MODEL_VERSION", DEFAULT_MODEL_VERSION),
        audio_embedding_model_name=os.getenv("POLYGLOT_AUDIO_EMBEDDING_MODEL", DEFAULT_AUDIO_EMBEDDING_MODEL),
        audio_embedding_model_version=os.getenv(
            "POLYGLOT_AUDIO_EMBEDDING_MODEL_VERSION",
            DEFAULT_AUDIO_EMBEDDING_VERSION,
        ),
        text_embedding_model_name=os.getenv("POLYGLOT_TEXT_EMBEDDING_MODEL", DEFAULT_TEXT_EMBEDDING_MODEL),
        text_embedding_model_version=os.getenv(
            "POLYGLOT_TEXT_EMBEDDING_MODEL_VERSION",
            DEFAULT_TEXT_EMBEDDING_VERSION,
        ),
        transcript_model_name=os.getenv("POLYGLOT_TRANSCRIPT_MODEL", DEFAULT_TRANSLATION_MODEL),
        transcript_model_version=os.getenv("POLYGLOT_TRANSCRIPT_MODEL_VERSION", DEFAULT_MODEL_VERSION),
    )
    if parse_bool(os.getenv("POLYGLOT_AUTO_INIT_DB"), default=True):
        try:
            memory.initialize()
        except Exception:
            logger.exception("Failed to initialize semantic memory schema")
            memory.enabled = False
    return memory
