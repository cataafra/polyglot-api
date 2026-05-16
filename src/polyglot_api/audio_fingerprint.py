import hashlib
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class AudioFingerprint:
    audio_hash: str
    embedding: list[float]
    duration_seconds: float
    sample_rate: int
    channels: int


def build_audio_fingerprint(audio_data, sample_rate: int, dimensions: int = 384) -> AudioFingerprint:
    mono_audio, channels = _mono_float_audio(audio_data)
    duration_seconds = float(mono_audio.size / sample_rate) if sample_rate else 0.0

    quantized = np.clip(mono_audio, -1.0, 1.0)
    pcm16 = (quantized * 32767.0).astype(np.int16).tobytes()
    audio_hash = hashlib.sha256(
        pcm16 + str(sample_rate).encode("utf-8") + str(channels).encode("utf-8")
    ).hexdigest()

    features = (
        _chunk_features(mono_audio, chunks=96)
        + _spectral_features(mono_audio, sample_rate, buckets=80)
        + _global_features(mono_audio, duration_seconds)
    )
    return AudioFingerprint(
        audio_hash=audio_hash,
        embedding=_normalize_vector(features, dimensions),
        duration_seconds=duration_seconds,
        sample_rate=int(sample_rate),
        channels=channels,
    )


def _mono_float_audio(audio_data):
    audio = np.asarray(audio_data)
    channels = 1
    if audio.ndim == 2:
        channels = int(audio.shape[1])
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32, copy=False)
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs > 1.0:
        audio = audio / max_abs
    return audio, channels


def _chunk_features(audio: np.ndarray, chunks: int) -> list[float]:
    if audio.size == 0:
        return [0.0] * (chunks * 3)

    features = []
    for chunk in np.array_split(audio, chunks):
        if chunk.size == 0:
            features.extend([0.0, 0.0, 0.0])
            continue
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        peak = float(np.max(np.abs(chunk)))
        zero_crossings = float(np.mean(np.abs(np.diff(np.signbit(chunk)))))
        features.extend([rms, peak, zero_crossings])
    return features


def _spectral_features(audio: np.ndarray, sample_rate: int, buckets: int) -> list[float]:
    if audio.size == 0:
        return [0.0] * buckets

    window = audio[: min(audio.size, sample_rate * 8)]
    if window.size < 2:
        return [0.0] * buckets

    spectrum = np.abs(np.fft.rfft(window))
    if float(np.sum(spectrum)) == 0.0:
        return [0.0] * buckets

    values = [float(np.mean(bucket)) for bucket in np.array_split(spectrum, buckets)]
    total = sum(values)
    if total == 0:
        return values
    return [value / total for value in values]


def _global_features(audio: np.ndarray, duration_seconds: float) -> list[float]:
    return [
        duration_seconds / 30.0,
        float(np.mean(np.abs(audio))) if audio.size else 0.0,
        float(np.std(audio)) if audio.size else 0.0,
        float(np.max(np.abs(audio))) if audio.size else 0.0,
    ]


def _normalize_vector(values: list[float], dimensions: int) -> list[float]:
    if len(values) < dimensions:
        values = values + [0.0] * (dimensions - len(values))
    elif len(values) > dimensions:
        values = values[:dimensions]

    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [value / norm for value in values]
