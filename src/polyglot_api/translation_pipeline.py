import os
import time
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import Optional
from urllib.parse import quote

import soundfile as sf

from .audio_fingerprint import build_audio_fingerprint
from .semantic_memory import (
    CacheLookupResult,
    SemanticMemoryMetadata,
    normalize_privacy_level,
    parse_bool,
)
from .text_semantics import TextFingerprint, build_text_fingerprint

logger = logging.getLogger(__name__)


@dataclass
class TranslationRequest:
    target_language: str
    speaker_id: int
    session_id: Optional[str] = None
    source_language: Optional[str] = None
    domain: Optional[str] = None
    privacy_level: Optional[str] = None
    use_semantic_cache: Optional[str] = None
    cache_strategy: Optional[str] = None
    use_transcript_memory: Optional[str] = None


@dataclass
class TranslationResult:
    audio_bytes: bytes
    headers: dict[str, str]


class TranslationPipeline:
    def __init__(self, translator, semantic_memory):
        self.translator = translator
        self.semantic_memory = semantic_memory

    def process(self, contents: bytes, request: TranslationRequest) -> TranslationResult:
        start = time.time()
        audio_data, sample_rate = self._read_audio(contents)
        fingerprint = build_audio_fingerprint(
            audio_data,
            sample_rate,
            dimensions=int(os.getenv("POLYGLOT_EMBEDDING_DIMENSIONS", "384")),
        )
        metadata = self._build_metadata(request)

        lookup_start = time.time()
        lookup = self._lookup_audio_exact(metadata, fingerprint)
        lookup_time = time.time() - lookup_start
        if lookup.hit and lookup.audio_bytes:
            total_time = time.time() - start
            return TranslationResult(
                audio_bytes=lookup.audio_bytes,
                headers=self._headers(
                    lookup,
                    total_time,
                    inference_time=0.0,
                    lookup_time=lookup_time,
                    transcript_time=0.0,
                ),
            )

        text_fingerprint = None
        transcript_time = 0.0
        if metadata.cache_strategy != "exact":
            transcript_start = time.time()
            text_fingerprint = self._build_transcript_fingerprint(audio_data, sample_rate, metadata)
            transcript_time = time.time() - transcript_start

            lookup_start = time.time()
            lookup = self.semantic_memory.lookup(metadata, fingerprint, text_fingerprint=text_fingerprint)
            lookup = self._attach_transcript_context(lookup, text_fingerprint)
            lookup_time += time.time() - lookup_start

            if lookup.hit and lookup.audio_bytes:
                total_time = time.time() - start
                return TranslationResult(
                    audio_bytes=lookup.audio_bytes,
                    headers=self._headers(
                        lookup,
                        total_time,
                        inference_time=0.0,
                        lookup_time=lookup_time,
                        transcript_time=transcript_time,
                    ),
                )

        inference_start = time.time()
        output_bytes = self.translator.translate(
            audio_data=audio_data,
            sample_rate=sample_rate,
            target_language=request.target_language,
            speaker_id=request.speaker_id,
        )
        inference_time = time.time() - inference_start

        if metadata.use_semantic_cache and metadata.cache_strategy != "stateless":
            self.semantic_memory.store(
                metadata=metadata,
                fingerprint=fingerprint,
                translated_audio=output_bytes,
                output_samplerate=sample_rate,
                text_fingerprint=text_fingerprint,
            )

        total_time = time.time() - start
        return TranslationResult(
            audio_bytes=output_bytes,
            headers=self._headers(lookup, total_time, inference_time, lookup_time, transcript_time),
        )

    @staticmethod
    def _read_audio(contents: bytes):
        return sf.read(BytesIO(contents))

    def _lookup_audio_exact(
        self,
        metadata: SemanticMemoryMetadata,
        fingerprint,
    ) -> CacheLookupResult:
        lookup_audio_exact = getattr(self.semantic_memory, "lookup_audio_exact", None)
        if not lookup_audio_exact:
            return CacheLookupResult(False, "exact", "audio exact lookup unavailable")
        return lookup_audio_exact(metadata, fingerprint)

    @staticmethod
    def _build_metadata(request: TranslationRequest) -> SemanticMemoryMetadata:
        return SemanticMemoryMetadata(
            session_id=request.session_id or "default-session",
            source_language=request.source_language or "auto",
            target_language=request.target_language,
            speaker_id=str(request.speaker_id),
            domain=request.domain or "general",
            privacy_level=normalize_privacy_level(request.privacy_level),
            cache_strategy=request.cache_strategy or os.getenv("POLYGLOT_CACHE_STRATEGY", "context"),
            use_semantic_cache=parse_bool(
                request.use_semantic_cache,
                default=parse_bool(os.getenv("POLYGLOT_SEMANTIC_CACHE_ENABLED"), default=False),
            ),
            use_transcript_memory=parse_bool(
                request.use_transcript_memory,
                default=parse_bool(os.getenv("POLYGLOT_TRANSCRIPT_MEMORY_ENABLED"), default=True),
            ),
        )

    def _build_transcript_fingerprint(
        self,
        audio_data,
        sample_rate: int,
        metadata: SemanticMemoryMetadata,
    ) -> Optional[TextFingerprint]:
        if not metadata.use_semantic_cache or not metadata.use_transcript_memory:
            return None
        if not metadata.source_language or metadata.source_language.lower() == "auto":
            return None
        try:
            transcript = self.translator.transcribe(
                audio_data=audio_data,
                sample_rate=sample_rate,
                source_language=metadata.source_language,
            )
        except Exception as exc:
            logger.warning("transcript_memory_failed source_language=%s error=%s", metadata.source_language, exc)
            return None
        fingerprint = build_text_fingerprint(
            transcript,
            dimensions=int(os.getenv("POLYGLOT_EMBEDDING_DIMENSIONS", "384")),
        )
        if not fingerprint.normalized_text:
            return None
        return fingerprint

    @staticmethod
    def _attach_transcript_context(
        lookup: CacheLookupResult,
        text_fingerprint: Optional[TextFingerprint],
    ) -> CacheLookupResult:
        if not text_fingerprint:
            return lookup
        if not lookup.source_transcript:
            lookup.source_transcript = text_fingerprint.source_transcript
        if not lookup.normalized_source_text:
            lookup.normalized_source_text = text_fingerprint.normalized_text
        return lookup

    @staticmethod
    def _headers(
        lookup: CacheLookupResult,
        total_time: float,
        inference_time: float,
        lookup_time: float,
        transcript_time: float,
    ) -> dict[str, str]:
        headers = {
            "X-Polyglot-Cache": "hit" if lookup.hit else "miss",
            "X-Polyglot-Cache-Strategy": lookup.strategy,
            "X-Polyglot-Cache-Layer": lookup.cache_layer,
            "X-Polyglot-Cache-Decision": lookup.decision_reason[:256],
            "X-Polyglot-Decision": lookup.decision_reason[:256],
            "X-Polyglot-Model-Version": os.getenv(
                "POLYGLOT_TRANSLATION_MODEL_VERSION",
                "local-or-huggingface",
            ),
            "X-Polyglot-Total-Time": f"{total_time:.4f}",
            "X-Polyglot-Transcript-Time": f"{transcript_time:.4f}",
            "X-Polyglot-Inference-Time": f"{inference_time:.4f}",
            "X-Polyglot-Lookup-Time": f"{lookup_time:.4f}",
        }
        if lookup.similarity is not None:
            headers["X-Polyglot-Similarity"] = f"{lookup.similarity:.6f}"
        if lookup.text_similarity is not None:
            headers["X-Polyglot-Text-Similarity"] = f"{lookup.text_similarity:.6f}"
        if lookup.translation_id:
            headers["X-Polyglot-Translation-Id"] = lookup.translation_id
        if lookup.source_transcript:
            headers["X-Polyglot-Source-Transcript"] = _safe_header_text(lookup.source_transcript)
        if lookup.normalized_source_text:
            headers["X-Polyglot-Normalized-Text"] = _safe_header_text(lookup.normalized_source_text)
        return {key: _header_value(value) for key, value in headers.items()}


def _safe_header_text(value: str) -> str:
    return quote(value[:256], safe="")


def _header_value(value) -> str:
    if isinstance(value, bytes):
        return value.decode("latin-1", errors="replace")
    return str(value)
