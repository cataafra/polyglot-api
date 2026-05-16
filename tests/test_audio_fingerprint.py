import numpy as np

from polyglot_api.audio_fingerprint import build_audio_fingerprint


def test_audio_fingerprint_is_stable_for_same_audio():
    audio = np.sin(np.linspace(0, 10, 16000)).astype("float32")

    first = build_audio_fingerprint(audio, 16000)
    second = build_audio_fingerprint(audio, 16000)

    assert first.audio_hash == second.audio_hash
    assert first.embedding == second.embedding
    assert len(first.embedding) == 384


def test_audio_fingerprint_tracks_channel_count():
    audio = np.zeros((16000, 2), dtype="float32")

    fingerprint = build_audio_fingerprint(audio, 16000)

    assert fingerprint.channels == 2
    assert fingerprint.duration_seconds == 1.0
