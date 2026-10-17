"""Turn sensor_events into a TimescaleDB hypertable when the extension is available.

Guarded: on plain PostgreSQL (or SQLite) this migration does nothing, so every environment can run
`alembic upgrade head`. The docker-compose database uses the TimescaleDB image, where it takes effect.

Revision ID: 0002
Revises: 0001
"""
from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UPGRADE_SQL = """
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb') THEN
        CREATE EXTENSION IF NOT EXISTS timescaledb;
        PERFORM create_hypertable('sensor_events', 'ts', if_not_exists => TRUE, migrate_data => TRUE,
                                  chunk_time_interval => INTERVAL '7 days');
        RAISE NOTICE 'sensor_events is now a TimescaleDB hypertable';
    ELSE
        RAISE NOTICE 'timescaledb not available - sensor_events stays a plain PostgreSQL table';
    END IF;
END
$$;
"""


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    # A hypertable cannot be turned back into a plain table in place; data stays intact either way.
    pass
