import hashlib
import tempfile
import unittest
from pathlib import Path

from app.core.constants import AUDIO, VIDEO
from app.services.file_security import MAX_FILE_SIZE_BYTES
from app.services.validation import validate_media_file


class ValidateMediaFileTests(unittest.TestCase):
    def test_accepts_supported_signatures(self):
        cases = [
            ("sample.wav", AUDIO, b"RIFF\x24\x00\x00\x00WAVEfmt "),
            ("sample.mp3", AUDIO, b"ID3\x04\x00\x00\x00\x00\x00\x21"),
            ("sample.mp4", VIDEO, b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x01"),
            ("sample.mov", VIDEO, b"\x00\x00\x00\x14ftypqt  \x00\x00\x00\x00"),
            ("sample.avi", VIDEO, b"RIFF\x24\x00\x00\x00AVI LIST"),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            for file_name, media_type, content in cases:
                path = Path(tmp_dir) / file_name
                path.write_bytes(content)

                result = validate_media_file(path, media_type)

                self.assertTrue(result.is_valid, file_name)
                self.assertEqual(result.media_type, media_type)
                self.assertEqual(result.details["security_status"], "Пройдено")
                self.assertEqual(result.details["sha256"], hashlib.sha256(content).hexdigest())

    def test_rejects_wrong_signature(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "fake.mp4"
            path.write_bytes(b"not a real mp4")

            result = validate_media_file(path, VIDEO)

            self.assertFalse(result.is_valid)
            self.assertIn("Сигнатура", result.message)
            self.assertEqual(result.details["security_status"], "Отклонено")

    def test_rejects_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "empty.wav"
            path.write_bytes(b"")

            result = validate_media_file(path, AUDIO)

            self.assertFalse(result.is_valid)
            self.assertIn("пустой", result.message)

    def test_rejects_file_over_size_limit_before_hashing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "large.mp3"
            with path.open("wb") as file:
                file.truncate(MAX_FILE_SIZE_BYTES + 1)

            result = validate_media_file(path, AUDIO)

            self.assertFalse(result.is_valid)
            self.assertIn("слишком большой", result.message)
            self.assertNotIn("sha256", result.details)

    def test_missing_file_returns_validation_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "missing.wav"

            result = validate_media_file(path, AUDIO)

            self.assertFalse(result.is_valid)
            self.assertIn("не найден", result.message)


if __name__ == "__main__":
    unittest.main()
