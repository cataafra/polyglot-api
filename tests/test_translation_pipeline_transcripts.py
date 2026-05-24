import json
import time

import numpy as np

from polyglot_api.audio_fingerprint import build_audio_fingerprint
from polyglot_api.semantic_memory import CacheLookupResult, PostgresSemanticMemory, SemanticMemoryMetadata
from polyglot_api.text_semantics import build_text_fingerprint
from polyglot_api.translation_pipeline import TranslationPipeline, TranslationRequest


class FakeTranslator:
    def __init__(self, transcript="câine", fail_transcript=False, transcript_delay=0.0):
        self.transcript = transcript
        self.fail_transcript = fail_transcript
        self.transcript_delay = transcript_delay
        self.transcribe_calls = 0
        self.translate_calls = 0

    def transcribe(self, audio_data, sample_rate, source_language):
        self.transcribe_calls += 1
        if self.transcript_delay:
            time.sleep(self.transcript_delay)
        if self.fail_transcript:
            raise RuntimeError("transcript failed")
        return self.transcript

    def translate(self, audio_data, sample_rate, target_language, speaker_id):
        self.translate_calls += 1
        return b"fresh-output"


class RecordingMemory:
    enabled = True

    def __init__(self, lookup_result=None, audio_exact_result=None):
        self.lookup_result = lookup_result or CacheLookupResult(False, "context", "miss")
        self.audio_exact_result = audio_exact_result or CacheLookupResult(False, "exact", "no exact audio hash match")
        self.audio_exact_lookups = []
        self.lookups = []
        self.stores = []

    def lookup_audio_exact(self, metadata, fingerprint):
        self.audio_exact_lookups.append((metadata, fingerprint))
        return self.audio_exact_result

    def lookup(self, metadata, fingerprint, text_fingerprint=None):
        self.lookups.append((metadata, fingerprint, text_fingerprint))
        return self.lookup_result

    def store(self, metadata, fingerprint, translated_audio, output_samplerate, text_fingerprint=None):
        self.stores.append((metadata, fingerprint, translated_audio, output_samplerate, text_fingerprint))


class FakeCursor:
    def __init__(self):
        self.audit_details = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query, params=None):
        if "INSERT INTO audit_events" in query:
            self.audit_details.append(json.loads(params[2]))


class FakeConnection:
    def __init__(self, cursor):
        self.cursor_instance = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        return None


class OrderedMemory(PostgresSemanticMemory):
    def __init__(self, *, audio_hit=False, text_hit=False, text_vector_hit=False):
        self.enabled = True
        self.similarity_threshold = 0.98
        self.text_similarity_threshold = 0.92
        self.audio_hit = audio_hit
        self.text_hit = text_hit
        self.text_vector_hit = text_vector_hit
        self.calls = []

    def _lookup_exact(self, metadata, fingerprint):
        self.calls.append("audio_exact")
        return CacheLookupResult(
            self.audio_hit,
            "exact",
            "audio",
            audio_bytes=b"audio-hit" if self.audio_hit else None,
            cache_layer="audio_exact" if self.audio_hit else "miss",
        )

    def _lookup_text_exact(self, metadata, text_fingerprint):
        self.calls.append("text_exact")
        return CacheLookupResult(
            self.text_hit,
            "exact",
            "text",
            audio_bytes=b"text-hit" if self.text_hit else None,
            cache_layer="text_exact" if self.text_hit else "miss",
        )

    def _lookup_text_vector(self, metadata, text_fingerprint, context_aware):
        self.calls.append("text_vector")
        return CacheLookupResult(
            self.text_vector_hit,
            "context" if context_aware else "semantic",
            "text vector",
            audio_bytes=b"text-vector-hit" if self.text_vector_hit else None,
            cache_layer="text_vector" if self.text_vector_hit else "miss",
        )

    def _lookup_vector(self, metadata, fingerprint, context_aware):
        self.calls.append("audio_vector")
        return CacheLookupResult(False, "context" if context_aware else "semantic", "audio vector miss")


def test_pipeline_uses_transcript_memory_with_explicit_source_language(monkeypatch):
    translator = FakeTranslator(transcript="Câine!")
    memory = RecordingMemory()
    pipeline = TranslationPipeline(translator=translator, semantic_memory=memory)
    monkeypatch.setattr(pipeline, "_read_audio", lambda contents: (_audio(), 16000))

    result = pipeline.process(
        b"wav",
        TranslationRequest(
            target_language="eng",
            speaker_id=0,
            source_language="ron",
            use_semantic_cache="true",
            use_transcript_memory="true",
            cache_strategy="context",
        ),
    )

    assert result.audio_bytes == b"fresh-output"
    assert translator.transcribe_calls == 1
    assert translator.translate_calls == 1
    assert memory.lookups[0][2].normalized_text == "caine"
    assert memory.stores[0][4].normalized_text == "caine"
    assert result.headers["X-Polyglot-Source-Transcript"] == "C%C3%A2ine%21"
    assert result.headers["X-Polyglot-Normalized-Text"] == "caine"
    assert "X-Polyglot-Transcript-Time" in result.headers


def test_pipeline_audio_exact_hit_returns_before_transcription(monkeypatch):
    translator = FakeTranslator(transcript="câine")
    memory = RecordingMemory(
        audio_exact_result=CacheLookupResult(
            True,
            "exact",
            "exact audio fingerprint match",
            audio_bytes=b"cached-output",
            cache_layer="audio_exact",
        )
    )
    pipeline = TranslationPipeline(translator=translator, semantic_memory=memory)
    monkeypatch.setattr(pipeline, "_read_audio", lambda contents: (_audio(), 16000))

    result = pipeline.process(
        b"wav",
        TranslationRequest(
            target_language="eng",
            speaker_id=0,
            source_language="ron",
            use_semantic_cache="true",
            use_transcript_memory="true",
            cache_strategy="context",
        ),
    )

    assert result.audio_bytes == b"cached-output"
    assert result.headers["X-Polyglot-Cache-Layer"] == "audio_exact"
    assert result.headers["X-Polyglot-Transcript-Time"] == "0.0000"
    assert result.headers["X-Polyglot-Inference-Time"] == "0.0000"
    assert translator.transcribe_calls == 0
    assert translator.translate_calls == 0
    assert memory.lookups == []
    assert memory.stores == []


def test_text_cache_hit_reports_transcript_time_without_inference(monkeypatch):
    translator = FakeTranslator(transcript="Câine!", transcript_delay=0.01)
    memory = RecordingMemory(
        lookup_result=CacheLookupResult(
            True,
            "exact",
            "normalized transcript match",
            text_similarity=1.0,
            audio_bytes=b"cached-output",
            cache_layer="text_exact",
        )
    )
    pipeline = TranslationPipeline(translator=translator, semantic_memory=memory)
    monkeypatch.setattr(pipeline, "_read_audio", lambda contents: (_audio(), 16000))

    result = pipeline.process(
        b"wav",
        TranslationRequest(
            target_language="eng",
            speaker_id=0,
            source_language="ron",
            use_semantic_cache="true",
            use_transcript_memory="true",
            cache_strategy="context",
        ),
    )

    assert result.audio_bytes == b"cached-output"
    assert result.headers["X-Polyglot-Cache-Layer"] == "text_exact"
    assert float(result.headers["X-Polyglot-Transcript-Time"]) > 0.0
    assert result.headers["X-Polyglot-Inference-Time"] == "0.0000"
    assert translator.translate_calls == 0


def test_pipeline_skips_transcript_memory_when_source_language_is_auto(monkeypatch):
    translator = FakeTranslator()
    memory = RecordingMemory()
    pipeline = TranslationPipeline(translator=translator, semantic_memory=memory)
    monkeypatch.setattr(pipeline, "_read_audio", lambda contents: (_audio(), 16000))

    pipeline.process(
        b"wav",
        TranslationRequest(
            target_language="eng",
            speaker_id=0,
            source_language="auto",
            use_semantic_cache="true",
            use_transcript_memory="true",
            cache_strategy="context",
        ),
    )

    assert translator.transcribe_calls == 0
    assert memory.lookups[0][2] is None
    assert memory.stores[0][4] is None


def test_pipeline_transcript_failure_falls_back_to_audio_only(monkeypatch):
    translator = FakeTranslator(fail_transcript=True)
    memory = RecordingMemory()
    pipeline = TranslationPipeline(translator=translator, semantic_memory=memory)
    monkeypatch.setattr(pipeline, "_read_audio", lambda contents: (_audio(), 16000))

    result = pipeline.process(
        b"wav",
        TranslationRequest(
            target_language="eng",
            speaker_id=0,
            source_language="ron",
            use_semantic_cache="true",
            use_transcript_memory="true",
            cache_strategy="context",
        ),
    )

    assert result.audio_bytes == b"fresh-output"
    assert translator.transcribe_calls == 1
    assert translator.translate_calls == 1
    assert memory.lookups[0][2] is None
    assert "X-Polyglot-Transcript-Time" in result.headers


def test_semantic_store_audit_omits_raw_and_normalized_transcript(monkeypatch):
    cursor = FakeCursor()
    memory = PostgresSemanticMemory(dsn="postgresql://unused")
    monkeypatch.setattr(memory, "_connect", lambda: FakeConnection(cursor))
    text_fingerprint = build_text_fingerprint("Câine!")

    memory.store(
        metadata=_metadata(),
        fingerprint=_fingerprint(),
        translated_audio=b"translated",
        output_samplerate=16000,
        text_fingerprint=text_fingerprint,
    )

    audit_details = cursor.audit_details[-1]
    assert audit_details["source_text_hash"] == text_fingerprint.text_hash
    assert audit_details["has_transcript"] is True
    assert "normalized_source_text" not in audit_details
    assert "source_transcript" not in audit_details
    assert "caine" not in json.dumps(audit_details)


def test_lookup_order_prefers_audio_exact_before_text_exact():
    memory = OrderedMemory(audio_hit=True, text_hit=True)

    result = memory.lookup(_metadata(), _fingerprint(), text_fingerprint=build_text_fingerprint("câine"))

    assert result.hit is True
    assert result.cache_layer == "audio_exact"
    assert memory.calls == ["audio_exact"]


def test_lookup_order_prefers_text_exact_before_text_vector_and_audio_vector():
    memory = OrderedMemory(text_hit=True, text_vector_hit=True)

    result = memory.lookup(_metadata(), _fingerprint(), text_fingerprint=build_text_fingerprint("câine"))

    assert result.hit is True
    assert result.cache_layer == "text_exact"
    assert memory.calls == ["audio_exact", "text_exact"]


def test_lookup_order_uses_text_vector_before_audio_vector():
    memory = OrderedMemory(text_vector_hit=True)

    result = memory.lookup(_metadata(), _fingerprint(), text_fingerprint=build_text_fingerprint("câine"))

    assert result.hit is True
    assert result.cache_layer == "text_vector"
    assert memory.calls == ["audio_exact", "text_exact", "text_vector"]


def _audio():
    return np.sin(np.linspace(0, 1, 16000)).astype("float32")


def _fingerprint():
    return build_audio_fingerprint(_audio(), 16000)


def _metadata():
    return SemanticMemoryMetadata(
        session_id="demo",
        source_language="ron",
        target_language="eng",
        speaker_id="0",
        domain="demo",
        privacy_level="transient",
        cache_strategy="context",
        use_semantic_cache=True,
        use_transcript_memory=True,
    )
