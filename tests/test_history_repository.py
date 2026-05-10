import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.core.constants import AUDIO, STATUS_COMPLETED, VERDICT_ORIGINAL
from app.core.models import AnalysisResult
from app.services.history_repository import (
    HISTORY_RETENTION_LIMIT,
    INTEGRITY_FAILED,
    INTEGRITY_PASSED,
    INTEGRITY_UNKNOWN,
    SOURCE_PATH_POLICY,
    HistoryRepository,
)


class HistoryRepositorySecurityTests(unittest.TestCase):
    def test_add_stores_filename_only_and_preserves_sha256(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            repository = HistoryRepository(Path(tmp_dir) / "data" / "app.sqlite3")
            repository.add(self.result(file_path=r"C:\secret\voice.wav", file_name="voice.wav"))

            records = repository.list_recent()

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].file_path, "voice.wav")
            self.assertEqual(records[0].file_name, "voice.wav")
            self.assertEqual(records[0].technical_info["sha256"], "abc123")
            self.assertEqual(records[0].technical_info["source_path_policy"], SOURCE_PATH_POLICY)
            self.assertEqual(records[0].technical_info["database_integrity_status"], INTEGRITY_PASSED)
            self.assertIn("database_record_hash", records[0].technical_info)

    def test_retention_keeps_latest_500_records(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            repository = HistoryRepository(Path(tmp_dir) / "data" / "app.sqlite3")

            for index in range(HISTORY_RETENTION_LIMIT + 1):
                repository.add(self.result(file_path=f"C:/media/file_{index}.wav", file_name=f"file_{index}.wav"))

            records = repository.list_recent(limit=HISTORY_RETENTION_LIMIT + 10)

            self.assertEqual(len(records), HISTORY_RETENTION_LIMIT)
            self.assertEqual(records[0].file_name, f"file_{HISTORY_RETENTION_LIMIT}.wav")
            self.assertNotIn("file_0.wav", {record.file_name for record in records})

    def test_retention_removes_evidence_file_for_pruned_record(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            evidence_dir = root / "data" / "evidence_frames"
            evidence_dir.mkdir(parents=True)
            repository = HistoryRepository(root / "data" / "app.sqlite3")

            for index in range(HISTORY_RETENTION_LIMIT + 1):
                evidence_path = evidence_dir / f"frame_{index}.jpg"
                evidence_path.write_bytes(b"jpeg")
                repository.add(
                    self.result(
                        file_path=f"C:/media/file_{index}.wav",
                        file_name=f"file_{index}.wav",
                        technical_info={
                            "sha256": "abc123",
                            "video_evidence_frame_path": f"data/evidence_frames/frame_{index}.jpg",
                        },
                    )
                )

            self.assertFalse((evidence_dir / "frame_0.jpg").exists())
            self.assertTrue((evidence_dir / f"frame_{HISTORY_RETENTION_LIMIT}.jpg").exists())

    def test_clear_removes_all_history(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            repository = HistoryRepository(Path(tmp_dir) / "data" / "app.sqlite3")
            repository.add(self.result())

            repository.clear()

            self.assertEqual(repository.list_recent(), [])

    def test_clear_removes_only_managed_evidence_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            evidence_dir = root / "data" / "evidence_frames"
            evidence_dir.mkdir(parents=True)
            managed_evidence = evidence_dir / "frame.jpg"
            managed_evidence.write_bytes(b"jpeg")
            managed_audio_evidence = evidence_dir / "audio_frame.jpg"
            managed_audio_evidence.write_bytes(b"audio-jpeg")
            outside_file = root / "outside.jpg"
            outside_file.write_bytes(b"outside")

            repository = HistoryRepository(root / "data" / "app.sqlite3")
            repository.add(
                self.result(
                    technical_info={
                        "sha256": "abc123",
                        "video_evidence_frame_path": "data/evidence_frames/frame.jpg",
                    },
                )
            )
            repository.add(
                self.result(
                    file_name="audio.wav",
                    technical_info={
                        "sha256": "audio123",
                        "audio_evidence_frame_path": "data/evidence_frames/audio_frame.jpg",
                    },
                )
            )
            repository.add(
                self.result(
                    file_name="outside.wav",
                    technical_info={
                        "sha256": "def456",
                        "video_evidence_frame_path": str(outside_file),
                    },
                )
            )

            repository.clear()

            self.assertFalse(managed_evidence.exists())
            self.assertFalse(managed_audio_evidence.exists())
            self.assertTrue(outside_file.exists())

    def test_initialize_sanitizes_legacy_full_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "data" / "app.sqlite3"
            db_path.parent.mkdir(parents=True)
            self.create_legacy_row(db_path, r"C:\private\old.wav")

            repository = HistoryRepository(db_path)
            records = repository.list_recent()

            self.assertEqual(records[0].file_path, "old.wav")
            self.assertEqual(records[0].file_name, "old.wav")
            self.assertEqual(records[0].technical_info["source_path_policy"], SOURCE_PATH_POLICY)
            self.assertEqual(records[0].technical_info["database_integrity_status"], INTEGRITY_UNKNOWN)

    def test_integrity_status_detects_tampered_row(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "data" / "app.sqlite3"
            repository = HistoryRepository(db_path)
            repository.add(self.result(raw_result="ORIGINAL (99.0%)"))

            connection = sqlite3.connect(db_path)
            try:
                connection.execute("UPDATE analysis_history SET raw_result = ? WHERE id = 1", ("DEEPFAKE (99.0%)",))
                connection.commit()
            finally:
                connection.close()

            records = repository.list_recent()

            self.assertEqual(records[0].technical_info["database_integrity_status"], INTEGRITY_FAILED)

    @staticmethod
    def result(
        file_path=r"C:\media\voice.wav",
        file_name="voice.wav",
        raw_result="ORIGINAL (99.0%)",
        technical_info=None,
    ):
        return AnalysisResult(
            file_path=file_path,
            file_name=file_name,
            media_type=AUDIO,
            file_size=128,
            status=STATUS_COMPLETED,
            verdict=VERDICT_ORIGINAL,
            confidence=99.0,
            raw_result=raw_result,
            duration=1.5,
            technical_info=technical_info or {"sha256": "abc123"},
            findings=["Критичные признаки подделки не обнаружены."],
        )

    @staticmethod
    def create_legacy_row(db_path, file_path):
        connection = sqlite3.connect(db_path)
        try:
            connection.execute(
                """
                CREATE TABLE analysis_history (
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
            connection.execute(
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
                    "2026-05-09T12:00:00",
                    file_path,
                    file_path,
                    AUDIO,
                    128,
                    STATUS_COMPLETED,
                    VERDICT_ORIGINAL,
                    99.0,
                    "ORIGINAL (99.0%)",
                    None,
                    1.5,
                    json.dumps({"sha256": "abc123"}),
                    json.dumps(["legacy"]),
                ),
            )
            connection.commit()
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
