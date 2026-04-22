#!/usr/bin/env python3
"""
=============================================================================
  Unit Tests for Drowsiness Detection System
  Run:  python test_functions.py
=============================================================================
"""
import unittest
import time
import os
import sys
import numpy as np
import cv2

# Import modules under test
from core_engine import (
    Config, GPIOController, EventLogger, MockCamera,
    BlinkDetector, DrowsinessScorer, compute_ear,
)
from drowsiness_detector import FaceEyeDetector, UIRenderer, open_camera


class TestConfig(unittest.TestCase):
    """Tests for configuration loading and defaults."""

    def test_defaults_loaded(self):
        """Config should have all default values when no file exists."""
        cfg = Config("nonexistent_file.json")
        self.assertEqual(cfg.ear_threshold, 0.22)
        self.assertEqual(cfg.degradation_weight, 0.60)
        self.assertEqual(cfg.calibration_blinks, 5)
        self.assertTrue(cfg.log_enabled)

    def test_load_real_config(self):
        """Config should load from the actual config file if present."""
        path = "drowsiness_detector_config.json"
        if os.path.isfile(path):
            cfg = Config(path)
            self.assertIsInstance(cfg.ear_threshold, float)
            self.assertGreater(cfg.ear_threshold, 0)

    def test_missing_key_raises(self):
        """Accessing a nonexistent key should raise AttributeError."""
        cfg = Config("nonexistent.json")
        with self.assertRaises(AttributeError):
            _ = cfg.totally_fake_key

    def test_config_types(self):
        """All critical config values should have correct types."""
        cfg = Config("nonexistent.json")
        self.assertIsInstance(cfg.ear_threshold, float)
        self.assertIsInstance(cfg.ear_consec_frames, int)
        self.assertIsInstance(cfg.calibration_blinks, int)
        self.assertIsInstance(cfg.show_video, bool)
        self.assertIsInstance(cfg.haar_min_face, list)


class TestEARComputation(unittest.TestCase):
    """Tests for the landmark-based EAR calculation."""

    def _make_eye_points(self, open_pct: float = 1.0) -> np.ndarray:
        """Create synthetic eye landmarks (6 points). open_pct: 1.0=fully open, 0.0=closed."""
        # 6 points in clockwise order: left, top-left, top-right, right, bottom-right, bottom-left
        # Width = 20, Max height = 10
        width = 20
        height = max(int(10 * open_pct), 0)
        
        return np.array([
            [0, 0],                     # P0: left corner
            [width*0.3, height],        # P1: top-left
            [width*0.7, height],        # P2: top-right
            [width, 0],                 # P3: right corner
            [width*0.7, -height],       # P4: bottom-right
            [width*0.3, -height]        # P5: bottom-left
        ], dtype=np.float64)

    def test_open_eye_returns_positive(self):
        """Fully open eye should return EAR > 0."""
        pts = self._make_eye_points(1.0)
        ear = compute_ear(pts)
        self.assertGreater(ear, 0.0)

    def test_closed_eye_returns_low(self):
        """Closed eye should return lower EAR than open eye."""
        open_ear = compute_ear(self._make_eye_points(1.0))
        closed_ear = compute_ear(self._make_eye_points(0.1))
        self.assertLess(closed_ear, open_ear)

    def test_empty_roi_returns_zero(self):
        """Empty or invalid points should return 0.0."""
        self.assertEqual(compute_ear(None), 0.0)
        self.assertEqual(compute_ear(np.array([])), 0.0)
        self.assertEqual(compute_ear(np.array([[0,0], [1,1]])), 0.0) # not 6 points

    def test_ear_in_valid_range(self):
        """EAR should always be reasonable (usually 0 to ~1.0)."""
        for pct in [0.0, 0.25, 0.5, 0.75, 1.0]:
            pts = self._make_eye_points(pct)
            ear = compute_ear(pts)
            self.assertGreaterEqual(ear, 0.0)
            self.assertLessEqual(ear, 1.5)


class TestBlinkDetector(unittest.TestCase):
    """Tests for blink detection and degradation index."""

    def _make_cfg(self) -> Config:
        cfg = Config("nonexistent.json")
        cfg._d["console_debug"] = False
        cfg._d["calibration_timeout_sec"] = 999
        return cfg

    def test_blink_detection_basic(self):
        """A drop-then-rise in EAR should register as a blink."""
        cfg = self._make_cfg()
        cfg._d["ear_consec_frames"] = 2
        det = BlinkDetector(cfg)

        # Open eyes
        for _ in range(10):
            det.update(0.30)

        # Close eyes for 3 frames then open
        for _ in range(3):
            det.update(0.10)
        blink_ended, _ = det.update(0.30)

        self.assertTrue(blink_ended, "Blink should be detected on reopening")
        self.assertEqual(det.blink_count, 1)

    def test_no_blink_if_too_short(self):
        """A single frame below threshold should NOT count as a blink."""
        cfg = self._make_cfg()
        cfg._d["ear_consec_frames"] = 3
        det = BlinkDetector(cfg)

        for _ in range(10):
            det.update(0.30)
        det.update(0.10)  # only 1 frame below
        blink_ended, _ = det.update(0.30)

        self.assertFalse(blink_ended)
        self.assertEqual(det.blink_count, 0)

    def test_calibration_completes(self):
        """After N blinks, calibration should complete."""
        cfg = self._make_cfg()
        cfg._d["calibration_blinks"] = 3
        cfg._d["ear_consec_frames"] = 2
        det = BlinkDetector(cfg)

        for _ in range(3):
            for _ in range(5):
                det.update(0.30)
            for _ in range(3):
                det.update(0.10)
            det.update(0.30)

        self.assertTrue(det.calibrated)
        self.assertIsNotNone(det.baseline_speed)
        print(f"  Baseline speed: {det.baseline_speed:.1f} frames/blink")

    def test_degradation_increases_with_slow_blinks(self):
        """Slower blinks after calibration should increase degradation."""
        cfg = self._make_cfg()
        cfg._d["calibration_blinks"] = 3
        cfg._d["blink_history_size"] = 3
        cfg._d["ear_consec_frames"] = 2
        det = BlinkDetector(cfg)

        # Calibrate with fast blinks (3 frames each)
        for _ in range(3):
            for _ in range(5):
                det.update(0.30)
            for _ in range(3):
                det.update(0.10)
            det.update(0.30)

        baseline = det.baseline_speed
        self.assertTrue(det.calibrated)

        # Now do slow blinks (8 frames each)
        for _ in range(5):
            for _ in range(5):
                det.update(0.30)
            for _ in range(8):
                det.update(0.10)
            _, deg = det.update(0.30)

        self.assertGreater(deg, 0.0,
                           f"Degradation should be > 0 for slow blinks, got {deg}")
        print(f"  Degradation after slow blinks: {deg * 100:.1f}%")

    def test_calibration_timeout(self):
        """If no blinks happen, calibration should timeout."""
        cfg = self._make_cfg()
        cfg._d["calibration_timeout_sec"] = 0.1  # very fast timeout
        det = BlinkDetector(cfg)

        time.sleep(0.2)
        det.update(0.30)

        self.assertTrue(det.calibrated)
        self.assertEqual(det.baseline_speed, 4.0)


class TestDrowsinessScorer(unittest.TestCase):
    """Tests for the drowsiness scoring formula."""

    def _make_cfg(self) -> Config:
        return Config("nonexistent.json")

    def test_alert_state_low_score(self):
        """High EAR + zero degradation should give low drowsiness."""
        cfg = self._make_cfg()
        scorer = DrowsinessScorer(cfg)
        for _ in range(10):
            score = scorer.compute(ear=0.30, degradation=0.0)
        self.assertLess(score, cfg.warning_threshold)

    def test_drowsy_state_high_score(self):
        """Low EAR + high degradation should give high drowsiness."""
        cfg = self._make_cfg()
        scorer = DrowsinessScorer(cfg)
        for i in range(20):
            score = scorer.compute(ear=0.05, degradation=0.8,
                                   sustained_closed=i + 10)
        self.assertGreater(score, cfg.warning_threshold)

    def test_level_classification(self):
        """get_level should return correct labels."""
        cfg = self._make_cfg()
        scorer = DrowsinessScorer(cfg)
        self.assertEqual(scorer.get_level(0.10), "normal")
        self.assertEqual(scorer.get_level(0.50), "warning")
        self.assertEqual(scorer.get_level(0.80), "critical")

    def test_score_bounds(self):
        """Score should always be in [0, 1]."""
        cfg = self._make_cfg()
        scorer = DrowsinessScorer(cfg)
        for ear in [0.0, 0.1, 0.3, 0.5]:
            for deg in [0.0, 0.5, 1.0, 2.0]:
                score = scorer.compute(ear, deg)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)


class TestMockCamera(unittest.TestCase):
    """Tests for the mock camera."""

    def test_returns_frames(self):
        cam = MockCamera(320, 240)
        self.assertTrue(cam.isOpened())
        ret, frame = cam.read()
        self.assertTrue(ret)
        self.assertEqual(frame.shape, (240, 320, 3))

    def test_release(self):
        cam = MockCamera()
        cam.release()
        self.assertFalse(cam.isOpened())
        ret, _ = cam.read()
        self.assertFalse(ret)

    def test_multiple_frames(self):
        """Should generate many frames without error."""
        cam = MockCamera(320, 240)
        for _ in range(100):
            ret, frame = cam.read()
            self.assertTrue(ret)
            self.assertIsNotNone(frame)
        cam.release()


class TestEventLogger(unittest.TestCase):
    """Tests for CSV logging."""

    def test_log_and_flush(self):
        cfg = Config("nonexistent.json")
        cfg._d["log_file"] = "_test_log.csv"
        cfg._d["log_max_events"] = 100
        logger = EventLogger(cfg)

        for i in range(10):
            logger.log(i, 0.25, 0.1, 0.3, False)

        logger.flush()
        self.assertTrue(os.path.isfile("_test_log.csv"))

        # Verify content
        with open("_test_log.csv", "r") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 11)  # header + 10 rows
        os.remove("_test_log.csv")

    def test_circular_buffer(self):
        cfg = Config("nonexistent.json")
        cfg._d["log_max_events"] = 5
        logger = EventLogger(cfg)
        for i in range(20):
            logger.log(i, 0.2, 0.1, 0.3, False)
        self.assertEqual(len(logger.buffer), 5)


class TestFaceEyeDetector(unittest.TestCase):
    """Tests for Haar cascade face/eye detection."""

    def test_no_face_in_blank_frame(self):
        cfg = Config("nonexistent.json")
        det = FaceEyeDetector(cfg)
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(blank)
        self.assertNotIn("face_rect", result)

    def test_detect_synthetic_face(self):
        """MockCamera generates a face-like shape; MediaPipe won't detect it, but it shouldn't crash."""
        cfg = Config("nonexistent.json")
        det = FaceEyeDetector(cfg)
        cam = MockCamera(640, 480)
        _, frame = cam.read()
        result = det.detect(frame)
        self.assertIsInstance(result, dict)
        cam.release()


class TestUIRenderer(unittest.TestCase):
    """Tests for UI overlay rendering."""

    def test_draw_no_crash(self):
        """All draw methods should execute without error."""
        cfg = Config("nonexistent.json")
        renderer = UIRenderer(cfg)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        renderer.draw_face_rect(frame, (100, 100, 200, 200), (0, 255, 0))
        pts = np.array([[10,10], [20,10], [15,20]])
        renderer.draw_eye_points(frame, pts, (255, 0, 0))
        renderer.draw_no_face(frame)
        renderer.draw_alert_flash(frame)
        renderer.draw_status_bar(
            frame, fps=25.0, ear=0.25, blink_count=10,
            degradation=0.15, drowsiness=0.40, level="warning",
            calibrated=True, calib_progress=5,
        )
        # If we get here, no crash
        self.assertTrue(True)

    def test_calibration_display(self):
        """Calibration overlay should render without error."""
        cfg = Config("nonexistent.json")
        renderer = UIRenderer(cfg)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        renderer.draw_status_bar(
            frame, fps=30.0, ear=0.30, blink_count=2,
            degradation=0.0, drowsiness=0.0, level="normal",
            calibrated=False, calib_progress=2,
        )
        self.assertTrue(True)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""

    def test_ear_with_uniform_image(self):
        """Test with empty array."""
        ear = compute_ear(np.array([]))
        self.assertEqual(ear, 0.0)

    def test_ear_with_noise(self):
        """Random points should not crash."""
        pts = np.random.randint(0, 100, (6, 2), dtype=np.int32)
        ear = compute_ear(pts)
        self.assertGreaterEqual(ear, 0.0)

    def test_rapid_blink_sequence(self):
        """Rapid alternating EAR should count blinks correctly."""
        cfg = Config("nonexistent.json")
        cfg._d["ear_consec_frames"] = 2
        cfg._d["min_blink_speed"] = 3
        cfg._d["console_debug"] = False
        det = BlinkDetector(cfg)

        blinks_detected = 0
        for _ in range(20):
            for _ in range(3):
                det.update(0.30)
            # Use 4 frames below threshold (above min_blink_speed=3)
            for _ in range(4):
                det.update(0.10)
            ended, _ = det.update(0.30)
            if ended:
                blinks_detected += 1

        self.assertGreater(blinks_detected, 0)
        self.assertEqual(blinks_detected, det.blink_count)

    def test_gpio_without_hardware(self):
        """GPIO controller should not crash on non-Pi platform."""
        cfg = Config("nonexistent.json")
        gpio = GPIOController(cfg)
        gpio.set_alert_state("normal")
        gpio.set_alert_state("warning")
        gpio.set_alert_state("critical")
        gpio.buzzer_off()
        gpio.cleanup()
        self.assertTrue(True)  # no crash

    def test_continuous_operation_memory(self):
        """Simulate 500 frames – deques should not grow unbounded."""
        cfg = Config("nonexistent.json")
        cfg._d["ear_consec_frames"] = 2
        cfg._d["blink_history_size"] = 5
        cfg._d["log_max_events"] = 100
        cfg._d["console_debug"] = False
        det = BlinkDetector(cfg)
        logger = EventLogger(cfg)

        for i in range(500):
            ear = 0.30 if i % 10 != 0 else 0.10
            _, deg = det.update(ear)
            logger.log(i, ear, deg, 0.2, False)

        self.assertLessEqual(len(det.speed_history), 5)
        self.assertLessEqual(len(logger.buffer), 100)


# ===========================================================================
#  INTEGRATION TEST (uses mock camera, runs briefly)
# ===========================================================================
class TestIntegration(unittest.TestCase):
    """Quick integration smoke test using mock camera."""

    def test_full_pipeline_mock(self):
        """Run the full pipeline for 50 frames with mock camera."""
        cfg = Config("nonexistent.json")
        cfg._d["mock_mode"] = True
        cfg._d["show_video"] = False
        cfg._d["console_debug"] = False
        cfg._d["log_file"] = "_test_integration_log.csv"

        detector_obj = FaceEyeDetector(cfg)
        blink_det = BlinkDetector(cfg)
        scorer = DrowsinessScorer(cfg)
        logger = EventLogger(cfg)
        cam = MockCamera(320, 240)

        for i in range(50):
            ret, frame = cam.read()
            self.assertTrue(ret)
            detections = detector_obj.detect(frame)

            if "left_eye_pts" in detections:
                ear = compute_ear(detections["left_eye_pts"])
            else:
                ear = 0.3  # default open

            _, deg = blink_det.update(ear)
            score = scorer.compute(ear, deg)
            level = scorer.get_level(score)
            logger.log(i, ear, deg, score, level == "critical")

        cam.release()
        logger.flush()

        if os.path.isfile("_test_integration_log.csv"):
            os.remove("_test_integration_log.csv")

        print(f"  Integration: {blink_det.blink_count} blinks detected "
              f"in 50 frames")


if __name__ == "__main__":
    print("=" * 60)
    print("  Drowsiness Detection System – Unit Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
