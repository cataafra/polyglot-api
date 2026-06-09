import importlib

import pytest
from fastapi import HTTPException


def test_internal_server_error_hides_exception_details(monkeypatch):
    monkeypatch.setenv("POLYGLOT_TRANSLATOR_MODE", "deterministic")
    monkeypatch.delenv("POLYGLOT_DATABASE_URL", raising=False)

    app_module = importlib.import_module("polyglot_api.app")

    exc = app_module._internal_server_error("There was an error processing the file")

    assert isinstance(exc, HTTPException)
    assert exc.status_code == 500
    assert exc.detail == "There was an error processing the file"
    assert "Traceback" not in exc.detail
    assert "psycopg" not in exc.detail


def test_process_upload_preserves_zero_speaker_id(monkeypatch):
    monkeypatch.setenv("POLYGLOT_TRANSLATOR_MODE", "deterministic")
    monkeypatch.delenv("POLYGLOT_DATABASE_URL", raising=False)

    app_module = importlib.import_module("polyglot_api.app")

    class FakeFile:
        file = None

    class FakePipeline:
        def process(self, contents, request):
            assert request.speaker_id == 0
            raise RuntimeError("stop before audio processing")

    FakeFile.file = type("FileObj", (), {"read": lambda self: b"wav"})()
    monkeypatch.setattr(app_module, "pipeline", FakePipeline())

    with pytest.raises(RuntimeError):
        app_module._process_upload(
            file=FakeFile(),
            language="eng",
            speaker_id=0,
            session_id=None,
            source_language=None,
            domain=None,
            privacy_level=None,
            use_semantic_cache=None,
            cache_strategy=None,
            use_transcript_memory=None,
        )
