CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY,
    external_session_id TEXT UNIQUE NOT NULL,
    anonymized_user_id TEXT,
    domain TEXT NOT NULL DEFAULT 'general',
    privacy_level TEXT NOT NULL DEFAULT 'transient',
    retention_expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_versions (
    id BIGSERIAL PRIMARY KEY,
    model_type TEXT NOT NULL,
    model_name TEXT NOT NULL,
    version TEXT NOT NULL,
    compatible_with TEXT[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (model_type, model_name, version)
);

CREATE TABLE IF NOT EXISTS audio_segments (
    id UUID PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    source_language TEXT NOT NULL DEFAULT 'auto',
    source_audio_hash TEXT NOT NULL,
    duration_seconds DOUBLE PRECISION NOT NULL,
    input_samplerate INTEGER NOT NULL,
    channels INTEGER NOT NULL,
    speaker_id TEXT NOT NULL,
    domain TEXT NOT NULL DEFAULT 'general',
    privacy_level TEXT NOT NULL DEFAULT 'transient',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audio_segments_hash
    ON audio_segments (source_audio_hash);

CREATE INDEX IF NOT EXISTS idx_audio_segments_context
    ON audio_segments (source_language, domain, privacy_level, speaker_id);

CREATE TABLE IF NOT EXISTS audio_translations (
    id UUID PRIMARY KEY,
    audio_segment_id UUID NOT NULL REFERENCES audio_segments(id) ON DELETE CASCADE,
    target_language TEXT NOT NULL,
    translated_audio BYTEA NOT NULL,
    audio_format TEXT NOT NULL DEFAULT 'wav',
    output_samplerate INTEGER NOT NULL,
    speaker_id TEXT NOT NULL,
    translation_model_name TEXT NOT NULL,
    translation_model_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audio_translations_lookup
    ON audio_translations (
        target_language,
        speaker_id,
        translation_model_name,
        translation_model_version
    );

CREATE TABLE IF NOT EXISTS audio_embeddings (
    id UUID PRIMARY KEY,
    audio_segment_id UUID NOT NULL REFERENCES audio_segments(id) ON DELETE CASCADE,
    embedding_model_name TEXT NOT NULL,
    embedding_model_version TEXT NOT NULL,
    source_audio_hash TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audio_embeddings_hash
    ON audio_embeddings (source_audio_hash);

CREATE INDEX IF NOT EXISTS idx_audio_embeddings_vector
    ON audio_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE TABLE IF NOT EXISTS terminology_memory (
    id UUID PRIMARY KEY,
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    domain TEXT NOT NULL DEFAULT 'general',
    source_language TEXT NOT NULL,
    target_language TEXT NOT NULL,
    source_term TEXT NOT NULL,
    preferred_translation TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (
        session_id,
        domain,
        source_language,
        target_language,
        source_term
    )
);

CREATE TABLE IF NOT EXISTS audit_events (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    event_type TEXT NOT NULL,
    details JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_events_session_time
    ON audit_events (session_id, created_at DESC);
