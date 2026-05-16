import os
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import soundfile as sf

from .audio_fingerprint import build_audio_fingerprint
from .semantic_memory import (
    CacheLookupResult,
    SemanticMemoryMetadata,
    parse_bool,
)


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
        lookup = self.semantic_memory.lookup(metadata, fingerprint)
        lookup_time = time.time() - lookup_start

        if lookup.hit and lookup.audio_bytes:
            total_time = time.time() - start
            return TranslationResult(
                audio_bytes=lookup.audio_bytes,
                headers=self._headers(lookup, total_time, inference_time=0.0, lookup_time=lookup_time),
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
            )

        total_time = time.time() - start
        return TranslationResult(
            audio_bytes=output_bytes,
            headers=self._headers(lookup, total_time, inference_time, lookup_time),
        )

    @staticmethod
    def _read_audio(contents: bytes):
        return sf.read(BytesIO(contents))

    @staticmethod
    def _build_metadata(request: TranslationRequest) -> SemanticMemoryMetadata:
        return SemanticMemoryMetadata(
            session_id=request.session_id or "default-session",
            source_language=request.source_language or "auto",
            target_language=request.target_language,
            speaker_id=str(request.speaker_id),
            domain=request.domain or "general",
            privacy_level=request.privacy_level or "transient",
            cache_strategy=request.cache_strategy or os.getenv("POLYGLOT_CACHE_STRATEGY", "context"),
            use_semantic_cache=parse_bool(
                request.use_semantic_cache,
                default=parse_bool(os.getenv("POLYGLOT_SEMANTIC_CACHE_ENABLED"), default=False),
            ),
        )

    @staticmethod
    def _headers(
        lookup: CacheLookupResult,
        total_time: float,
        inference_time: float,
        lookup_time: float,
    ) -> dict[str, str]:
        headers = {
            "X-Polyglot-Cache": "hit" if lookup.hit else "miss",
            "X-Polyglot-Cache-Strategy": lookup.strategy,
            "X-Polyglot-Cache-Decision": lookup.decision_reason[:256],
            "X-Polyglot-Decision": lookup.decision_reason[:256],
            "X-Polyglot-Model-Version": os.getenv(
                "POLYGLOT_TRANSLATION_MODEL_VERSION",
                "local-or-huggingface",
            ),
            "X-Polyglot-Total-Time": f"{total_time:.4f}",
            "X-Polyglot-Inference-Time": f"{inference_time:.4f}",
            "X-Polyglot-Lookup-Time": f"{lookup_time:.4f}",
        }
        if lookup.similarity is not None:
            headers["X-Polyglot-Similarity"] = f"{lookup.similarity:.6f}"
        if lookup.translation_id:
            headers["X-Polyglot-Translation-Id"] = lookup.translation_id
        return headers
