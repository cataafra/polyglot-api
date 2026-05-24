import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass


DEFAULT_TEXT_EMBEDDING_MODEL = "polyglot-char-ngram-text"
DEFAULT_TEXT_EMBEDDING_VERSION = "v1"


@dataclass
class TextFingerprint:
    source_transcript: str
    normalized_text: str
    text_hash: str
    embedding: list[float]


def build_text_fingerprint(transcript: str, dimensions: int = 384) -> TextFingerprint:
    normalized = normalize_transcript(transcript)
    return TextFingerprint(
        source_transcript=(transcript or "").strip(),
        normalized_text=normalized,
        text_hash=hash_normalized_text(normalized),
        embedding=build_text_embedding(normalized, dimensions=dimensions),
    )


def normalize_transcript(transcript: str) -> str:
    text = (transcript or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"_+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def hash_normalized_text(normalized_text: str) -> str:
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def build_text_embedding(normalized_text: str, dimensions: int = 384) -> list[float]:
    vector = [0.0] * dimensions
    if not normalized_text:
        return vector

    tokens = normalized_text.split()
    features = tokens + [f"token:{token}" for token in tokens]
    compact = normalized_text.replace(" ", "_")
    for ngram_size in (2, 3, 4):
        if len(compact) < ngram_size:
            continue
        features.extend(compact[index : index + ngram_size] for index in range(len(compact) - ngram_size + 1))

    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
