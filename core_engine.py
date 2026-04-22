"""
Core Drowsiness Detection Engine
Implements: EAR calculation, Blink Quality Degradation Index, calibration, alerting.
"""
import time
import math
import json
import csv
import os
import platform
from collections import deque
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Platform detection & optional GPIO
# ---------------------------------------------------------------------------
IS_PI = platform.machine().startswith("arm") or platform.machine().startswith("aarch")

GPIO = None
if IS_PI:
    try:
        import RPi.GPIO as _GPIO
        GPIO = _GPIO
    except ImportError:
        pass


# ============================= CONFIGURATION ================================
class Config:
    """Loads JSON config with graceful fallback to defaults."""

    DEFAULTS: Dict[str, Any] = {
        "ear_threshold": 0.22,
        "ear_consec_frames": 2,
        "min_blink_speed": 3,          # reject blinks shorter than this (noise filter)
        "degradation_weight": 0.60,
        "calibration_blinks": 5,
        "calibration_timeout_sec": 30,
        "blink_history_size": 5,
        "warning_threshold": 0.35,
        "critical_threshold": 0.65,
        "sustained_closure_frames": 15, # frames of closed eyes to trigger direct alert
        "alert_cooldown_sec": 3.0,
        "gpio_buzzer": 17,
        "gpio_red_led": 27,
        "gpio_green_led": 22,
        "cam_width": 640,
        "cam_height": 480,
        "cam_fps": 30,
        "cam_index": 0,
        "show_video": True,
        "log_enabled": True,
        "log_max_events": 10000,
        "log_file": "drowsiness_log.csv",
        "console_debug": False,
        "haar_scale": 1.3,
        "haar_neighbors": 5,
        "haar_min_face": [80, 80],
        "mock_mode": False,
    }

    def __init__(self, path: str = "drowsiness_detector_config.json"):
        self._d = dict(self.DEFAULTS)
        self._load(path)

    def _load(self, path: str) -> None:
        if not os.path.isfile(path):
            print(f"[CONFIG] '{path}' not found – using defaults.")
            return
        try:
            with open(path, "r") as f:
                raw = json.load(f)
            # Flatten nested structure into simple keys
            mapping = {
                "ear_threshold": ("ear_threshold", "value"),
                "ear_consec_frames": ("ear_consec_frames", "value"),
                "degradation_weight": ("degradation_weight", "value"),
                "calibration_blinks": ("calibration_blinks", "value"),
                "calibration_timeout_sec": ("calibration_timeout_sec", "value"),
                "blink_history_size": ("blink_history_size", "value"),
                "warning_threshold": ("drowsiness_thresholds", "warning"),
                "critical_threshold": ("drowsiness_thresholds", "critical"),
                "alert_cooldown_sec": ("alert_cooldown_sec", "value"),
                "gpio_buzzer": ("gpio_pins", "buzzer"),
                "gpio_red_led": ("gpio_pins", "red_led"),
                "gpio_green_led": ("gpio_pins", "green_led"),
                "cam_width": ("camera", "resolution_width"),
                "cam_height": ("camera", "resolution_height"),
                "cam_fps": ("camera", "fps_target"),
                "cam_index": ("camera", "camera_index"),
                "show_video": ("display", "show_video"),
                "log_enabled": ("logging", "enabled"),
                "log_max_events": ("logging", "max_events"),
                "log_file": ("logging", "log_file"),
                "console_debug": ("logging", "console_debug"),
                "haar_scale": ("performance", "haar_scale_factor"),
                "haar_neighbors": ("performance", "haar_min_neighbors"),
                "haar_min_face": ("performance", "haar_min_face_size"),
                "mock_mode": ("mock_mode", "enabled"),
            }
            for key, path_keys in mapping.items():
                obj = raw
                try:
                    for pk in path_keys:
                        obj = obj[pk]
                    self._d[key] = obj
                except (KeyError, TypeError):
                    pass
            print("[CONFIG] Loaded configuration file.")
        except (json.JSONDecodeError, IOError) as e:
            print(f"[CONFIG] Error reading config: {e} – using defaults.")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        try:
            return self._d[name]
        except KeyError:
            raise AttributeError(f"No config key '{name}'")


# ============================= GPIO CONTROLLER ==============================
class GPIOController:
    """Controls buzzer and LEDs via GPIO with graceful degradation."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.available = False
        if GPIO is not None:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                GPIO.setup(cfg.gpio_buzzer, GPIO.OUT, initial=GPIO.LOW)
                GPIO.setup(cfg.gpio_red_led, GPIO.OUT, initial=GPIO.LOW)
                GPIO.setup(cfg.gpio_green_led, GPIO.OUT, initial=GPIO.HIGH)
                self.available = True
                print("[GPIO] Initialized successfully.")
            except Exception as e:
                print(f"[GPIO] Init failed: {e} – running without hardware.")
        else:
            print("[GPIO] Not available on this platform – simulating.")

    def set_alert_state(self, level: str) -> None:
        """Set LEDs/buzzer: level is 'normal', 'warning', or 'critical'."""
        if not self.available:
            return
        try:
            if level == "normal":
                GPIO.output(self.cfg.gpio_green_led, GPIO.HIGH)
                GPIO.output(self.cfg.gpio_red_led, GPIO.LOW)
                GPIO.output(self.cfg.gpio_buzzer, GPIO.LOW)
            elif level == "warning":
                GPIO.output(self.cfg.gpio_green_led, GPIO.LOW)
                GPIO.output(self.cfg.gpio_red_led, GPIO.HIGH)
                GPIO.output(self.cfg.gpio_buzzer, GPIO.LOW)
            elif level == "critical":
                GPIO.output(self.cfg.gpio_green_led, GPIO.LOW)
                GPIO.output(self.cfg.gpio_red_led, GPIO.HIGH)
                GPIO.output(self.cfg.gpio_buzzer, GPIO.HIGH)
        except Exception as e:
            print(f"[GPIO] Error setting state: {e}")

    def buzzer_off(self) -> None:
        if self.available:
            try:
                GPIO.output(self.cfg.gpio_buzzer, GPIO.LOW)
            except Exception:
                pass

    def cleanup(self) -> None:
        if self.available:
            try:
                GPIO.cleanup()
                print("[GPIO] Cleaned up.")
            except Exception:
                pass


# ============================= EVENT LOGGER =================================
class EventLogger:
    """CSV circular-buffer logger for drowsiness events."""

    HEADER = ["frame_num", "timestamp", "ear", "degradation",
              "drowsiness_level", "alert_triggered"]

    def __init__(self, cfg: Config):
        self.enabled = cfg.log_enabled
        self.path = cfg.log_file
        self.max_events = cfg.log_max_events
        self.buffer: deque = deque(maxlen=self.max_events)

    def log(self, frame_num: int, ear: float, degradation: float,
            drowsiness: float, alert: bool) -> None:
        if not self.enabled:
            return
        self.buffer.append([
            frame_num,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            round(ear, 4),
            round(degradation, 4),
            round(drowsiness, 4),
            int(alert),
        ])

    def flush(self) -> None:
        """Write buffer to CSV."""
        if not self.enabled or len(self.buffer) == 0:
            return
        try:
            with open(self.path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(self.HEADER)
                w.writerows(self.buffer)
            print(f"[LOG] Saved {len(self.buffer)} events to {self.path}")
        except IOError as e:
            print(f"[LOG] Write error: {e}")


# ========================== EYE ASPECT RATIO ================================
def compute_ear(eye_points: np.ndarray) -> float:
    """
    Compute Eye Aspect Ratio from 6 facial landmarks for an eye.
    The points are expected in clockwise order starting from the outer corner.
    
    Returns a float representing the EAR.
    """
    if eye_points is None or len(eye_points) != 6:
        return 0.0
        
    # Vertical distances across the eye
    vert_dist1 = np.linalg.norm(eye_points[1] - eye_points[5])
    vert_dist2 = np.linalg.norm(eye_points[2] - eye_points[4])
    
    # Horizontal distance from corner to corner
    horiz_dist = np.linalg.norm(eye_points[0] - eye_points[3])
    
    if horiz_dist == 0:
        return 0.0
        
    ear = (vert_dist1 + vert_dist2) / (2.0 * horiz_dist)
    return ear


# ========================== BLINK DETECTOR ==================================
class BlinkDetector:
    """
    Detects blinks and computes the Blink Quality Degradation Index.

    Blink detection: EAR drops below threshold for N consecutive frames,
    then rises back above threshold.

    Blink speed: number of frames the eye stays closed during a blink.
    Baseline speed is established during calibration (first K blinks).

    Degradation Index = (current_avg_speed - baseline) / baseline
    A positive value means blinks are getting slower → drowsiness.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.frame_counter: int = 0       # frames below threshold
        self.blink_count: int = 0
        self.in_blink: bool = False        # currently in a blink?
        self.blink_start_frame: int = 0
        self.global_frame: int = 0

        # Sustained eye closure tracking (for direct drowsiness boost)
        self.sustained_closed: int = 0     # consecutive frames eyes are closed

        # Calibration state
        self.calibrated: bool = False
        self.baseline_speed: Optional[float] = None
        self.calibration_speeds: List[int] = []
        self.calibration_start_time: float = time.time()

        # Rolling history of recent VALID blink speeds (frames per blink)
        self.speed_history: deque = deque(maxlen=cfg.blink_history_size)

        # Previous EAR for edge detection
        self.prev_ear: float = 0.3

    def update(self, ear: float) -> Tuple[bool, float]:
        """
        Process a new EAR value. Returns (blink_just_ended, degradation_index).
        """
        self.global_frame += 1
        blink_ended = False
        threshold = self.cfg.ear_threshold
        min_speed = self.cfg.min_blink_speed  # noise filter

        if ear < threshold:
            # Eye is closed
            self.sustained_closed += 1
            if not self.in_blink:
                self.in_blink = True
                self.blink_start_frame = self.global_frame
            self.frame_counter += 1
        else:
            # Eye is open
            self.sustained_closed = 0  # reset sustained closure
            if self.in_blink and self.frame_counter >= self.cfg.ear_consec_frames:
                blink_speed = self.global_frame - self.blink_start_frame
                # FILTER: reject micro-blinks (Haar cascade noise)
                if blink_speed >= min_speed:
                    self._record_blink(blink_speed)
                    blink_ended = True
                    self.blink_count += 1
                    if self.cfg.console_debug:
                        print(f"[BLINK] #{self.blink_count} speed={blink_speed}f")
                else:
                    if self.cfg.console_debug:
                        print(f"[NOISE] Rejected micro-blink speed={blink_speed}f")

            self.in_blink = False
            self.frame_counter = 0

        self.prev_ear = ear
        degradation = self._compute_degradation()

        # Handle calibration timeout
        if not self.calibrated:
            elapsed = time.time() - self.calibration_start_time
            if elapsed > self.cfg.calibration_timeout_sec:
                self.baseline_speed = 4.0  # sensible default
                self.calibrated = True
                print(f"[CALIB] Timeout – using default baseline: "
                      f"{self.baseline_speed:.1f} frames/blink")

        return blink_ended, degradation

    def _record_blink(self, speed: int) -> None:
        """Record a VALID blink speed for calibration or history."""
        if not self.calibrated:
            self.calibration_speeds.append(speed)
            print(f"[CALIB] Blink {len(self.calibration_speeds)}/"
                  f"{self.cfg.calibration_blinks} (speed={speed}f)")
            if len(self.calibration_speeds) >= self.cfg.calibration_blinks:
                self.baseline_speed = (
                    sum(self.calibration_speeds) / len(self.calibration_speeds)
                )
                self.calibrated = True
                print(f"[CALIB] Complete! Baseline: "
                      f"{self.baseline_speed:.1f} frames/blink")
        self.speed_history.append(speed)

    def _compute_degradation(self) -> float:
        """
        Compute Blink Quality Degradation Index.
        Returns 0.0 if not enough data, otherwise a float >= 0.
        Also factors in sustained eye closure as an extreme degradation signal.
        """
        if self.baseline_speed is None:
            return 0.0

        # If eyes are closed for a long time, that IS extreme degradation
        sustained_limit = self.cfg.sustained_closure_frames
        if self.sustained_closed >= sustained_limit:
            # Scale from 0.5 to 1.0 based on how long eyes stayed closed
            ratio = min(self.sustained_closed / (sustained_limit * 3.0), 1.0)
            return 0.5 + 0.5 * ratio

        if len(self.speed_history) < 2:
            return 0.0

        avg_speed = sum(self.speed_history) / len(self.speed_history)
        degradation = (avg_speed - self.baseline_speed) / self.baseline_speed
        return max(degradation, 0.0)  # clamp to non-negative

    @property
    def calibration_progress(self) -> int:
        return len(self.calibration_speeds)


# ========================= DROWSINESS SCORER ================================
class DrowsinessScorer:
    """
    Combines EAR-based score and Degradation Index into a single
    drowsiness score in [0, 1].

    Formula (improved):
      ear_score = 1 - clamp(ear / ear_threshold, 0, 1)
      deg_score = clamp(degradation, 0, 1)
      base = (1 - w) * ear_score + w * deg_score

      If eyes are fully closed (ear_score ~1.0), the score is boosted
      so sustained closure CAN reach critical even without degradation data.

    where w = degradation_weight (default 0.6).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Smoothing: exponential moving average
        self._smoothed: float = 0.0
        self._alpha: float = 0.3  # smoothing factor

    def compute(self, ear: float, degradation: float,
                sustained_closed: int = 0) -> float:
        """Return smoothed drowsiness score in [0, 1].

        Args:
            ear: current Eye Aspect Ratio
            degradation: Blink Quality Degradation Index
            sustained_closed: consecutive frames eyes have been closed
        """
        # EAR-based component: lower EAR → higher score
        if self.cfg.ear_threshold > 0:
            ear_norm = ear / self.cfg.ear_threshold
        else:
            ear_norm = 1.0
        ear_score = 1.0 - min(max(ear_norm, 0.0), 1.0)

        # Degradation component: higher degradation → higher score
        deg_score = min(max(degradation, 0.0), 1.0)

        w = self.cfg.degradation_weight
        raw = (1.0 - w) * ear_score + w * deg_score

        # CRITICAL FIX: Boost score when eyes are sustained-closed.
        # Without this, max score with degradation=0 is only 40%.
        # If eyes are closed for many frames, directly push score up.
        sustained_limit = self.cfg.sustained_closure_frames
        if sustained_closed >= sustained_limit:
            # Ramp from current raw up toward 1.0 over 3x the limit
            closure_ratio = min(sustained_closed / (sustained_limit * 2.0), 1.0)
            # Blend: at least the raw score, ramping up to 0.95
            raw = max(raw, 0.4 + 0.55 * closure_ratio)

        # Smooth (use faster alpha when score is rising for responsiveness)
        alpha = self._alpha if raw <= self._smoothed else 0.5
        self._smoothed = alpha * raw + (1.0 - alpha) * self._smoothed
        return min(max(self._smoothed, 0.0), 1.0)

    def get_level(self, score: float) -> str:
        """Return 'normal', 'warning', or 'critical'."""
        if score >= self.cfg.critical_threshold:
            return "critical"
        elif score >= self.cfg.warning_threshold:
            return "warning"
        return "normal"


# ========================== MOCK CAMERA =====================================
class MockCamera:
    """Generates synthetic frames with simulated blink patterns for testing."""

    def __init__(self, width: int = 640, height: int = 480):
        self.w, self.h = width, height
        self.frame_num = 0
        self._open = True

    def isOpened(self) -> bool:
        return self._open

    def read(self) -> Tuple[bool, np.ndarray]:
        if not self._open:
            return False, None
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        frame[:] = (40, 40, 40)  # dark gray background

        self.frame_num += 1
        # Draw a synthetic face (circle)
        cx, cy = self.w // 2, self.h // 2
        cv2.circle(frame, (cx, cy), 120, (180, 160, 140), -1)

        # Simulate blink cycle: every 90 frames, close eyes for some frames
        cycle = self.frame_num % 90
        # Gradually increase blink duration to simulate drowsiness
        drowsy_factor = min(self.frame_num / 1500.0, 1.0)
        blink_dur = int(4 + 8 * drowsy_factor)  # 4-12 frames

        eye_open = cycle >= blink_dur
        eye_h = 12 if eye_open else 2

        # Left eye
        le_x, le_y = cx - 40, cy - 20
        cv2.ellipse(frame, (le_x, le_y), (18, eye_h), 0, 0, 360,
                     (255, 255, 255), -1)
        cv2.circle(frame, (le_x, le_y), 6 if eye_open else 1,
                   (50, 50, 50), -1)

        # Right eye
        re_x, re_y = cx + 40, cy - 20
        cv2.ellipse(frame, (re_x, re_y), (18, eye_h), 0, 0, 360,
                     (255, 255, 255), -1)
        cv2.circle(frame, (re_x, re_y), 6 if eye_open else 1,
                   (50, 50, 50), -1)

        # Mouth
        cv2.ellipse(frame, (cx, cy + 40), (25, 10), 0, 0, 180,
                     (100, 100, 150), 2)

        return True, frame

    def set(self, prop, val):
        pass

    def release(self):
        self._open = False

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        return 0.0
