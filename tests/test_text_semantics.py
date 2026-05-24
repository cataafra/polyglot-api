from polyglot_api.text_semantics import build_text_fingerprint, normalize_transcript


def test_transcript_normalization_removes_case_punctuation_and_diacritics():
    assert normalize_transcript("câine") == "caine"
    assert normalize_transcript("Caine!") == "caine"
    assert normalize_transcript("câine.") == "caine"
    assert normalize_transcript("  CÂINE,\n mare! ") == "caine mare"


def test_text_fingerprint_hashes_equivalent_normalized_transcripts_identically():
    first = build_text_fingerprint("câine")
    second = build_text_fingerprint("Caine!")

    assert first.normalized_text == "caine"
    assert first.text_hash == second.text_hash


def test_text_embedding_is_deterministic_and_pgvector_sized():
    first = build_text_fingerprint("hai sa vedem cache-ul")
    second = build_text_fingerprint("hai sa vedem cache-ul")

    assert first.embedding == second.embedding
    assert len(first.embedding) == 384
    assert any(value != 0.0 for value in first.embedding)
