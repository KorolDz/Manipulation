import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from app.core.constants import DATABASE_PATH
from app.core.models import AnalysisResult, HistoryRecord


class HistoryRepository:
    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self):
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    file_size INTEGER,
                    status TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    confidence REAL,
                    raw_result TEXT NOT NULL,
                    error_message TEXT,
                    duration REAL,
                    technical_info TEXT NOT NULL DEFAULT '{}',
                    findings TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            self.migrate(connection)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_analysis_history_created_at "
                "ON analysis_history(created_at DESC)"
            )

    def migrate(self, connection):
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(analysis_history)").fetchall()
        }
        migrations = {
            "duration": "ALTER TABLE analysis_history ADD COLUMN duration REAL",
            "technical_info": "ALTER TABLE analysis_history ADD COLUMN technical_info TEXT NOT NULL DEFAULT '{}'",
            "findings": "ALTER TABLE analysis_history ADD COLUMN findings TEXT NOT NULL DEFAULT '[]'",
        }

        for column, statement in migrations.items():
            if column not in columns:
                connection.execute(statement)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def add(self, result: AnalysisResult):
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO analysis_history (
                    created_at,
                    file_path,
                    file_name,
                    media_type,
                    file_size,
                    status,
                    verdict,
                    confidence,
                    raw_result,
                    error_message,
                    duration,
                    technical_info,
                    findings
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(timespec="seconds"),
                    result.file_path,
                    result.file_name,
                    result.media_type,
                    result.file_size,
                    result.status,
                    result.verdict,
                    result.confidence,
                    result.raw_result,
                    result.error_message,
                    result.duration,
                    self.encode_json(result.technical_info, {}),
                    self.encode_json(result.findings, []),
                ),
            )
            return cursor.lastrowid

    def list_recent(self, limit=100):
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    created_at,
                    file_path,
                    file_name,
                    media_type,
                    file_size,
                    status,
                    verdict,
                    confidence,
                    raw_result,
                    error_message,
                    duration,
                    technical_info,
                    findings
                FROM analysis_history
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [self.row_to_record(row) for row in rows]

    def clear(self):
        with self.connect() as connection:
            connection.execute("DELETE FROM analysis_history")

    @staticmethod
    def row_to_record(row):
        return HistoryRecord(
            id=row["id"],
            created_at=row["created_at"],
            file_path=row["file_path"],
            file_name=row["file_name"],
            media_type=row["media_type"],
            file_size=row["file_size"],
            status=row["status"],
            verdict=row["verdict"],
            confidence=row["confidence"],
            raw_result=row["raw_result"],
            error_message=row["error_message"],
            duration=row["duration"],
            technical_info=HistoryRepository.decode_json(row["technical_info"], {}),
            findings=HistoryRepository.decode_json(row["findings"], []),
        )

    @staticmethod
    def encode_json(value, fallback):
        try:
            return json.dumps(value if value is not None else fallback, ensure_ascii=False)
        except TypeError:
            return json.dumps(fallback, ensure_ascii=False)

    @staticmethod
    def decode_json(value, fallback):
        if not value:
            return fallback

        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return fallback

        return decoded if decoded is not None else fallback
