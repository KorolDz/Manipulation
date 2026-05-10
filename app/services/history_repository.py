import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from app.core.constants import DATABASE_PATH
from app.core.models import AnalysisResult, HistoryRecord


HISTORY_RETENTION_LIMIT = 500
SOURCE_PATH_POLICY = "filename_only"
INTEGRITY_PASSED = "Пройдено"
INTEGRITY_FAILED = "Нарушено"
INTEGRITY_UNKNOWN = "Не проверено"
EVIDENCE_FRAME_PATH_KEYS = {
    "audio_evidence_frame_path",
    "video_evidence_frame_path",
}


class HistoryRepository:
    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.apply_storage_permissions()
        self.initialize()
        self.apply_storage_permissions()

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

        self.sanitize_legacy_paths(connection)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        self.apply_connection_pragmas(connection)
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
            created_at = datetime.now().isoformat(timespec="seconds")
            technical_info = self.prepare_technical_info(result, created_at)
            safe_file_name = self.safe_file_name(result.file_name or result.file_path)
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
                    created_at,
                    safe_file_name,
                    safe_file_name,
                    result.media_type,
                    result.file_size,
                    result.status,
                    result.verdict,
                    result.confidence,
                    result.raw_result,
                    result.error_message,
                    result.duration,
                    self.encode_json(technical_info, {}),
                    self.encode_json(result.findings, []),
                ),
            )
            self.prune_old_records(connection)
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
            rows = connection.execute("SELECT technical_info FROM analysis_history").fetchall()
            evidence_paths = self.evidence_paths_from_rows(rows)
            connection.execute("DELETE FROM analysis_history")
        self.delete_evidence_files(evidence_paths)
        self.vacuum()

    def row_to_record(self, row):
        technical_info = self.decode_json(row["technical_info"], {})
        findings = self.decode_json(row["findings"], [])
        technical_info["database_integrity_status"] = self.database_integrity_status(row, technical_info, findings)

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
            technical_info=technical_info,
            findings=findings,
        )

    def prepare_technical_info(self, result, created_at):
        technical_info = dict(result.technical_info or {})
        technical_info["source_path_policy"] = SOURCE_PATH_POLICY
        technical_info.pop("database_integrity_status", None)
        technical_info.pop("database_record_hash", None)
        technical_info["database_record_hash"] = self.calculate_record_hash(
            created_at=created_at,
            file_name=self.safe_file_name(result.file_name or result.file_path),
            media_type=result.media_type,
            file_size=result.file_size,
            status=result.status,
            verdict=result.verdict,
            confidence=result.confidence,
            sha256=technical_info.get("sha256"),
            raw_result=result.raw_result,
            findings=result.findings,
        )
        return technical_info

    def database_integrity_status(self, row, technical_info, findings):
        stored_hash = technical_info.get("database_record_hash")
        if not stored_hash:
            return INTEGRITY_UNKNOWN

        actual_hash = self.calculate_record_hash(
            created_at=row["created_at"],
            file_name=row["file_name"],
            media_type=row["media_type"],
            file_size=row["file_size"],
            status=row["status"],
            verdict=row["verdict"],
            confidence=row["confidence"],
            sha256=technical_info.get("sha256"),
            raw_result=row["raw_result"],
            findings=findings,
        )
        return INTEGRITY_PASSED if stored_hash == actual_hash else INTEGRITY_FAILED

    def prune_old_records(self, connection):
        rows_to_prune = connection.execute(
            """
            SELECT technical_info
            FROM analysis_history
            WHERE id NOT IN (
                SELECT id
                FROM analysis_history
                ORDER BY created_at DESC, id DESC
                LIMIT ?
            )
            """,
            (HISTORY_RETENTION_LIMIT,),
        ).fetchall()
        evidence_paths_to_prune = self.evidence_paths_from_rows(rows_to_prune)

        connection.execute(
            """
            DELETE FROM analysis_history
            WHERE id NOT IN (
                SELECT id
                FROM analysis_history
                ORDER BY created_at DESC, id DESC
                LIMIT ?
            )
            """,
            (HISTORY_RETENTION_LIMIT,),
        )
        remaining_rows = connection.execute("SELECT technical_info FROM analysis_history").fetchall()
        remaining_evidence_paths = self.evidence_paths_from_rows(remaining_rows)
        self.delete_evidence_files(evidence_paths_to_prune - remaining_evidence_paths)

    def evidence_paths_from_rows(self, rows):
        evidence_paths = set()
        for row in rows:
            technical_info = self.decode_json(row["technical_info"], {})
            for key in EVIDENCE_FRAME_PATH_KEYS:
                evidence_path = technical_info.get(key)
                if evidence_path:
                    evidence_paths.add(str(evidence_path))
        return evidence_paths

    def delete_evidence_files(self, evidence_paths):
        for evidence_path in evidence_paths:
            resolved_path = self.resolve_managed_evidence_path(evidence_path)
            if resolved_path is None or not resolved_path.is_file():
                continue

            try:
                resolved_path.unlink()
            except OSError:
                pass

    def resolve_managed_evidence_path(self, evidence_path):
        path = Path(str(evidence_path))
        storage_root = self.db_path.parent.parent
        candidate = path if path.is_absolute() else storage_root / path

        try:
            resolved_candidate = candidate.resolve(strict=False)
            evidence_dir = (self.db_path.parent / "evidence_frames").resolve(strict=False)
            resolved_candidate.relative_to(evidence_dir)
        except (OSError, ValueError):
            return None

        return resolved_candidate

    def sanitize_legacy_paths(self, connection):
        rows = connection.execute("SELECT id, file_path, file_name, technical_info FROM analysis_history").fetchall()
        for row in rows:
            safe_file_path = self.safe_file_name(row["file_path"])
            safe_file_name = self.safe_file_name(row["file_name"])
            technical_info = self.decode_json(row["technical_info"], {})
            technical_info.setdefault("source_path_policy", SOURCE_PATH_POLICY)

            if (
                safe_file_path != row["file_path"]
                or safe_file_name != row["file_name"]
                or technical_info != self.decode_json(row["technical_info"], {})
            ):
                connection.execute(
                    """
                    UPDATE analysis_history
                    SET file_path = ?, file_name = ?, technical_info = ?
                    WHERE id = ?
                    """,
                    (safe_file_path, safe_file_name, self.encode_json(technical_info, {}), row["id"]),
                )

    def vacuum(self):
        connection = sqlite3.connect(self.db_path)
        try:
            self.apply_connection_pragmas(connection)
            connection.execute("VACUUM")
        finally:
            connection.close()

    def apply_storage_permissions(self):
        self.chmod_if_possible(self.db_path.parent, 0o700)
        if self.db_path.exists():
            self.chmod_if_possible(self.db_path, 0o600)

    @staticmethod
    def apply_connection_pragmas(connection):
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA secure_delete = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")

    @staticmethod
    def chmod_if_possible(path, mode):
        try:
            os.chmod(path, mode)
        except (AttributeError, NotImplementedError, OSError):
            pass

    @staticmethod
    def safe_file_name(value):
        text = str(value or "").replace("\\", "/").rstrip("/")
        file_name = text.split("/")[-1]
        return file_name or "Без файла"

    @staticmethod
    def calculate_record_hash(
        created_at,
        file_name,
        media_type,
        file_size,
        status,
        verdict,
        confidence,
        sha256,
        raw_result,
        findings,
    ):
        import hashlib

        payload = {
            "created_at": created_at,
            "file_name": file_name,
            "media_type": media_type,
            "file_size": file_size,
            "status": status,
            "verdict": verdict,
            "confidence": confidence,
            "sha256": sha256,
            "raw_result": raw_result,
            "findings": findings or [],
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

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
