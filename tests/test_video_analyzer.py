import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from video_detection.video_analyzer import (
    VideoAnalysisOutput,
    analyze_video,
    save_evidence_frame,
    select_evidence_candidate,
    interpret_video_metrics,
)


class VideoAnalyzerExplanationTests(unittest.TestCase):
    def test_interpret_video_metrics_detects_face_lip_and_artifact_issues(self):
        details = interpret_video_metrics(
            faces_analyzed=12,
            score_variance=0.08,
            face_motion_score=0.15,
            lip_motion_score=0.16,
            artifact_score=0.7,
        )

        self.assertEqual(
            details["video_issue_categories"],
            ["подмена лица", "движение губ", "визуальные артефакты"],
        )
        findings = " ".join(details["video_issue_findings"])
        self.assertIn("подмена лица", findings)
        self.assertIn("несогласованность губ", findings)
        self.assertIn("визуальные артефакты", findings)

    def test_interpret_video_metrics_returns_neutral_finding_without_issues(self):
        details = interpret_video_metrics(
            faces_analyzed=12,
            score_variance=0.01,
            face_motion_score=0.02,
            lip_motion_score=0.03,
            artifact_score=0.04,
        )

        self.assertEqual(details["video_issue_categories"], [])
        self.assertEqual(
            details["video_issue_findings"],
            ["Дополнительные видео-признаки явных аномалий лица, губ или артефактов не выявили."],
        )

    def test_analyze_video_keeps_string_wrapper_api(self):
        output = VideoAnalysisOutput(
            raw_result="ORIGINAL (90.0%)",
            technical_info={"video_faces_analyzed": 1},
        )

        with patch("video_detection.video_analyzer.analyze_video_with_details", return_value=output):
            result = analyze_video("sample.mp4")

        self.assertEqual(result, "ORIGINAL (90.0%)")

    def test_select_evidence_frame_candidate_keeps_highest_score(self):
        low_score_candidate = {"combined_score": 0.2, "frame_index": 1}
        high_score_candidate = {"combined_score": 0.9, "frame_index": 2}

        selected = select_evidence_candidate(low_score_candidate, high_score_candidate)

        self.assertIs(selected, high_score_candidate)

    def test_save_evidence_frame_creates_relative_jpeg_with_hash(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            evidence_dir = project_root / "data" / "evidence_frames"
            candidate = {
                "frame": np.zeros((80, 120, 3), dtype=np.uint8),
                "box": (10, 10, 40, 40),
                "frame_index": 7,
                "frame_time": 0.24,
                "combined_score": 0.8,
            }

            with (
                patch("video_detection.video_analyzer.PROJECT_ROOT", project_root),
                patch("video_detection.video_analyzer.EVIDENCE_FRAMES_DIR", evidence_dir),
            ):
                metadata = save_evidence_frame(candidate, ["подмена лица"])

            saved_path = project_root / metadata["video_evidence_frame_path"]
            self.assertTrue(saved_path.is_file())
            self.assertEqual(metadata["video_evidence_frame_path"].replace("\\", "/").split("/")[:2], ["data", "evidence_frames"])
            self.assertEqual(metadata["video_evidence_frame_index"], 7)
            self.assertEqual(metadata["video_evidence_frame_time"], 0.24)
            self.assertEqual(len(metadata["video_evidence_frame_sha256"]), 64)
            self.assertIn("подмена лица", metadata["video_evidence_frame_label"])


if __name__ == "__main__":
    unittest.main()
