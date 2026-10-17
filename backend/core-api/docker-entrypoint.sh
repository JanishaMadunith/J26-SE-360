#!/bin/sh
# Apply database migrations (retrying while PostgreSQL starts), then start the API.
set -e

tries=0
until alembic upgrade head; do
    tries=$((tries + 1))
    if [ "$tries" -ge 10 ]; then
        echo "Database still unavailable after $tries attempts - giving up." >&2
        exit 1
    fi
    echo "Waiting for the database (attempt $tries)..." >&2
    sleep 3
done

# UVICORN_EXTRA_ARGS lets docker-compose add e.g. "--reload" for development.
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers ${UVICORN_EXTRA_ARGS}
