#!/usr/bin/env bash
set -euo pipefail

export PGDATA="${PGDATA:-/workspace/polyglot-postgres}"
export POLYGLOT_DATABASE_URL="${POLYGLOT_DATABASE_URL:-postgresql://postgres@127.0.0.1:5432/polyglot}"
export POLYGLOT_AUTO_INIT_DB="${POLYGLOT_AUTO_INIT_DB:-true}"
export POLYGLOT_SEMANTIC_CACHE_ENABLED="${POLYGLOT_SEMANTIC_CACHE_ENABLED:-true}"
export POLYGLOT_TRANSCRIPT_MEMORY_ENABLED="${POLYGLOT_TRANSCRIPT_MEMORY_ENABLED:-true}"
export POLYGLOT_CACHE_STRATEGY="${POLYGLOT_CACHE_STRATEGY:-context}"
export POLYGLOT_SIMILARITY_THRESHOLD="${POLYGLOT_SIMILARITY_THRESHOLD:-0.98}"
export POLYGLOT_TEXT_SIMILARITY_THRESHOLD="${POLYGLOT_TEXT_SIMILARITY_THRESHOLD:-0.92}"

PG_BIN="$(ls -d /usr/lib/postgresql/*/bin | sort -V | tail -n 1)"
DB_NAME="${POLYGLOT_DB_NAME:-polyglot}"
API_PID=""

run_as_postgres() {
    runuser -u postgres -- "$@"
}

stop_services() {
    if [[ -n "${API_PID}" ]]; then
        kill -TERM "${API_PID}" 2>/dev/null || true
    fi
    if [[ -f "${PGDATA}/postmaster.pid" ]]; then
        run_as_postgres "${PG_BIN}/pg_ctl" -D "${PGDATA}" -m fast stop >/dev/null 2>&1 || true
    fi
}

trap stop_services EXIT
trap 'exit 143' TERM INT

mkdir -p "$(dirname "${PGDATA}")"
chown -R postgres:postgres "$(dirname "${PGDATA}")"

if [[ ! -f "${PGDATA}/PG_VERSION" ]]; then
    echo "Initializing Postgres data directory at ${PGDATA}"
    run_as_postgres "${PG_BIN}/initdb" -D "${PGDATA}"
    {
        echo "listen_addresses = '127.0.0.1'"
        echo "port = 5432"
    } >> "${PGDATA}/postgresql.conf"
    {
        echo "host all all 127.0.0.1/32 trust"
        echo "host all all ::1/128 trust"
    } >> "${PGDATA}/pg_hba.conf"
fi

echo "Starting Postgres"
run_as_postgres "${PG_BIN}/pg_ctl" -D "${PGDATA}" -l "${PGDATA}/postgres.log" -w start

DB_EXISTS="$(run_as_postgres "${PG_BIN}/psql" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" || true)"
if [[ "${DB_EXISTS}" != "1" ]]; then
    run_as_postgres "${PG_BIN}/createdb" "${DB_NAME}"
fi
run_as_postgres "${PG_BIN}/psql" -d "${DB_NAME}" -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "Starting Polyglot API on 0.0.0.0:8000"
gunicorn \
    -b 0.0.0.0:8000 \
    polyglot_api.app:app \
    --workers "${WORKER_COUNT:-1}" \
    --timeout "${GUNICORN_TIMEOUT:-600}" \
    -k uvicorn.workers.UvicornWorker \
    --log-level info \
    --access-logfile - \
    --error-logfile - &

API_PID="$!"
wait "${API_PID}"
