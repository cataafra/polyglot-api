import pytest
from fastapi import HTTPException

from polyglot_api.auth import require_api_token


def test_api_token_is_disabled_when_env_is_unset(monkeypatch):
    monkeypatch.delenv("POLYGLOT_API_TOKEN", raising=False)

    assert require_api_token() is True


def test_api_token_accepts_matching_header(monkeypatch):
    monkeypatch.setenv("POLYGLOT_API_TOKEN", "demo-secret")

    assert require_api_token("demo-secret") is True


def test_api_token_rejects_missing_or_wrong_header(monkeypatch):
    monkeypatch.setenv("POLYGLOT_API_TOKEN", "demo-secret")

    with pytest.raises(HTTPException) as exc:
        require_api_token("wrong")

    assert exc.value.status_code == 401
