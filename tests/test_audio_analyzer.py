import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from audio_detection.audio_analyzer import (
    AudioAnalysisOutput,
    analyze_audio,
    interpret_audio_metrics,
    save_audio_evidence_image,
)


class AudioAnalyzerExplanationTests(unittest.TestCase):
    def test_interpret_audio_metrics_detects_noise_timbre_jumps_and_clipping(self):
        details = interpret_audio_metrics(
            noise_score=0.8,
            spectral_flatness_score=0.7,
            timbre_instability_score=0.7,
            spectral_jump_score=0.7,
            clipping_score=0.5,
            suspicious_time=1.2,
        )

        self.assertEqual(
            details["audio_issue_categories"],
            [
                "шумовая обработка",
                "спектральная плоскость",
                "нестабильность тембра",
                "спектральные скачки",
                "перегрузка сигнала",
            ],
        )
        findings = " ".join(details["audio_issue_findings"])
        self.assertIn("шумовой обработки", findings)
        self.assertIn("нестабильность тембра", findings)
        self.assertIn("спектральные переходы", findings)
        self.assertIn("перегрузка", findings)

    def test_interpret_audio_metrics_returns_neutral_finding_without_issues(self):
        details = interpret_audio_metrics(
            noise_score=0.01,
            spectral_flatness_score=0.01,
            timbre_instability_score=0.01,
            spectral_jump_score=0.01,
            clipping_score=0.0,
        )

        self.assertEqual(details["audio_issue_categories"], [])
        self.assertEqual(
            details["audio_issue_findings"],
            ["Дополнительные аудио-признаки явных аномалий шума, тембра или спектра не выявили."],
        )

    def test_analyze_audio_keeps_string_wrapper_api(self):
        output = AudioAnalysisOutput(
            raw_result="ORIGINAL (Вероятность: 90.0%)",
            technical_info={"audio_noise_score": 0.1},
        )

        with patch("audio_detection.audio_analyzer.analyze_audio_with_details", return_value=output):
            result = analyze_audio("sample.wav")

        self.assertEqual(result, "ORIGINAL (Вероятность: 90.0%)")

    def test_save_audio_evidence_image_creates_relative_jpeg_with_hash(self):
        sr = 16000
        y = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr, endpoint=False)).astype(np.float32) * 0.2

        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            evidence_dir = project_root / "data" / "evidence_frames"
            technical_info = {
                "audio_suspicious_time": 0.25,
                "audio_issue_categories": ["спектральные скачки"],
            }

            with (
                patch("audio_detection.audio_analyzer.PROJECT_ROOT", project_root),
                patch("audio_detection.audio_analyzer.EVIDENCE_FRAMES_DIR", evidence_dir),
            ):
                metadata = save_audio_evidence_image(y, sr, technical_info)

            saved_path = project_root / metadata["audio_evidence_frame_path"]
            self.assertTrue(saved_path.is_file())
            self.assertEqual(metadata["audio_evidence_frame_path"].replace("\\", "/").split("/")[:2], ["data", "evidence_frames"])
            self.assertEqual(metadata["audio_evidence_frame_time"], 0.25)
            self.assertEqual(len(metadata["audio_evidence_frame_sha256"]), 64)
            self.assertIn("спектральные скачки", metadata["audio_evidence_frame_label"])


if __name__ == "__main__":
    unittest.main()
