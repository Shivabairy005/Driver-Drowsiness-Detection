#!/usr/bin/env python3
"""
=============================================================================
  Drowsiness Detection System  –  Production-Ready
  Novel: Blink Quality Degradation Index (early warning, 3-5s before closure)
  Runs on: Raspberry Pi 4B (15 FPS) | Laptop (30 FPS)
  Launch:  python drowsiness_detector.py [--mock] [--no-display] [--debug]
=============================================================================
"""
import sys
import os
import time
import signal
import argparse
from typing import Optional, Tuple
import threading

import numpy as np
import cv2
import mediapipe as mp

try:
    from flask import Flask, Response
except ImportError:
    Flask, Response = None, None

# Local modules
from core_engine import (
    Config, GPIOController, EventLogger, MockCamera,
    BlinkDetector, DrowsinessScorer, compute_ear,
)

# ===========================================================================
#  CONSTANTS
# ===========================================================================
COLORS = {
    "green":   (0, 220, 80),
    "yellow":  (0, 220, 255),
    "red":     (0, 0, 255),
    "white":   (255, 255, 255),
    "cyan":    (255, 200, 0),
    "bg_dark": (30, 30, 30),
}

LEVEL_COLOR = {
    "normal":   COLORS["green"],
    "warning":  COLORS["yellow"],
    "critical": COLORS["red"],
}


# ===========================================================================
#  FACE / EYE DETECTOR  (MediaPipe Face Mesh – 468 Landmarks)
# ===========================================================================
class FaceEyeDetector:
    """Uses Google MediaPipe for highly accurate facial landmarks."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Standard MediaPipe eye indices (clockwise from outer corner)
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    def detect(self, frame: np.ndarray) -> dict:
        """
        Detect face and eye landmarks. Returns dict with keys:
          face_rect: (x, y, w, h)
          left_eye_pts: numpy array of 6 (x,y) points
          right_eye_pts: numpy array of 6 (x,y) points
        """
        result = {}
        h, w = frame.shape[:2]

        # MediaPipe needs RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Extract precise 2D eye points
            left_eye_pts = np.array([
                [int(landmarks[i].x * w), int(landmarks[i].y * h)]
                for i in self.LEFT_EYE
            ])
            right_eye_pts = np.array([
                [int(landmarks[i].x * w), int(landmarks[i].y * h)]
                for i in self.RIGHT_EYE
            ])
            
            result["left_eye_pts"] = left_eye_pts
            result["right_eye_pts"] = right_eye_pts
            
            # Compute a rough bounding box for the face to draw the rectangle
            x_coords = [int(lm.x * w) for lm in landmarks]
            y_coords = [int(lm.y * h) for lm in landmarks]
            if x_coords and y_coords:
                min_x, max_x = max(0, min(x_coords)), min(w, max(x_coords))
                min_y, max_y = max(0, min(y_coords)), min(h, max(y_coords))
                result["face_rect"] = (min_x, min_y, max_x - min_x, max_y - min_y)

        return result


# ===========================================================================
#  UI OVERLAY RENDERER
# ===========================================================================
class UIRenderer:
    """Draws all overlays, annotations, and metrics on the video frame."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_small = cv2.FONT_HERSHEY_PLAIN

    def draw_face_rect(self, frame: np.ndarray, rect: tuple,
                       color: tuple) -> None:
        x, y, w, h = rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    def draw_eye_points(self, frame: np.ndarray, pts: np.ndarray, color: tuple) -> None:
        """Draw landmarks over eyes."""
        hull = cv2.convexHull(pts)
        cv2.drawContours(frame, [hull], -1, color, 1)

    def draw_status_bar(self, frame: np.ndarray, fps: float, ear: float,
                        blink_count: int, degradation: float,
                        drowsiness: float, level: str,
                        calibrated: bool, calib_progress: int) -> None:
        """Draw metrics panel at top of frame."""
        h, w = frame.shape[:2]
        color = LEVEL_COLOR.get(level, COLORS["white"])

        # Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), COLORS["bg_dark"], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Row 1: Status + FPS
        if not calibrated:
            status = f"CALIBRATING {calib_progress}/{self.cfg.calibration_blinks}"
            cv2.putText(frame, status, (10, 25), self.font, 0.7,
                        COLORS["cyan"], 2)
        else:
            status = f"Drowsiness: {drowsiness * 100:.0f}%"
            cv2.putText(frame, status, (10, 25), self.font, 0.7, color, 2)

            # Drowsiness bar
            bar_x, bar_y, bar_w, bar_h = w - 220, 8, 200, 18
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h),
                          (80, 80, 80), -1)
            fill = int(bar_w * min(drowsiness, 1.0))
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + fill, bar_y + bar_h), color, -1)
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h),
                          COLORS["white"], 1)

        # Row 2: Metrics
        fps_txt = f"FPS:{fps:.0f}"
        ear_txt = f"EAR:{ear:.3f}"
        blink_txt = f"Blinks:{blink_count}"
        deg_txt = f"Degrad:{degradation * 100:.1f}%"
        y2 = 55
        cv2.putText(frame, fps_txt, (10, y2), self.font_small, 1.2,
                    COLORS["white"], 1)
        cv2.putText(frame, ear_txt, (120, y2), self.font_small, 1.2,
                    COLORS["white"], 1)
        cv2.putText(frame, blink_txt, (260, y2), self.font_small, 1.2,
                    COLORS["white"], 1)
        cv2.putText(frame, deg_txt, (400, y2), self.font_small, 1.2,
                    COLORS["cyan"], 1)

        # Row 3: Alert level text
        level_txt = f"[{level.upper()}]"
        cv2.putText(frame, level_txt, (10, 80), self.font, 0.6, color, 2)

    def draw_no_face(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        cv2.putText(frame, "NO FACE DETECTED", (w // 2 - 140, h // 2),
                    self.font, 0.8, COLORS["yellow"], 2)

    def draw_alert_flash(self, frame: np.ndarray) -> None:
        """Red border flash for critical alerts."""
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLORS["red"], 6)
        cv2.putText(frame, "!! WAKE UP !!", (w // 2 - 100, h - 30),
                    self.font, 1.0, COLORS["red"], 3)


# ===========================================================================
#  CAMERA INITIALIZER
# ===========================================================================
def open_camera(cfg: Config) -> object:
    """Try camera indices 0, 1, 2 in sequence. Returns cap or None."""
    indices = [cfg.cam_index, 0, 1, 2]
    seen = set()
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        print(f"[CAM] Trying camera index {idx}...")
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_height)
            cap.set(cv2.CAP_PROP_FPS, cfg.cam_fps)
            print(f"[CAM] Opened camera {idx} "
                  f"({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                  f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
                  f"@ {cap.get(cv2.CAP_PROP_FPS):.0f}fps)")
            return cap
        cap.release()
    print("[CAM] ERROR: No camera found!")
    return None


# ===========================================================================
#  MAIN LOOP
# ===========================================================================
class DrowsinessDetector:
    """Main application controller orchestrating all components."""

    def __init__(self, cfg: Config, use_mock: bool = False,
                 show_display: bool = True):
        self.cfg = cfg
        self.show_display = show_display and cfg.show_video

        # Components
        self.detector = FaceEyeDetector(cfg)
        self.blink_det = BlinkDetector(cfg)
        self.scorer = DrowsinessScorer(cfg)
        self.gpio = GPIOController(cfg)
        self.logger = EventLogger(cfg)
        self.renderer = UIRenderer(cfg)

        # Camera
        if use_mock or cfg.mock_mode:
            print("[CAM] Using MOCK camera (synthetic frames)")
            self.cap = MockCamera(cfg.cam_width, cfg.cam_height)
        else:
            self.cap = open_camera(cfg)
            if self.cap is None:
                print("[CAM] Falling back to mock camera.")
                self.cap = MockCamera(cfg.cam_width, cfg.cam_height)

        # State
        self.running = True
        self.frame_num = 0
        self.last_alert_time = 0.0
        self.fps = 0.0
        self._fps_frames = 0
        self._fps_start = time.time()

        # Web Stream State
        self.latest_frame = None
        self.lock = threading.Lock()
        self.web_stream = cfg._d.get("web_stream", False)
        if self.web_stream:
            self._start_flask_server()

        # EAR smoothing: carry forward last valid reading when cascade flickers
        self.last_valid_ear = 0.30  # assume eyes open at start
        self.no_eye_frames = 0      # consecutive frames without eye detection
        self.ear_smooth = 0.30      # exponential moving average of EAR

    def _start_flask_server(self) -> None:
        if Flask is None:
            print("[WEB] ERROR: Flask is not installed. Run 'pip install flask' to use --web-stream")
            self.web_stream = False
            return

        app = Flask(__name__)

        @app.route('/')
        def index():
            return "<html><body style='background-color:#1e1e1e; color:white; text-align:center; font-family:sans-serif;'><h2>Drowsiness Detector Live Stream</h2><img src='/video_feed' width='640' height='480' style='border:2px solid #333; border-radius:8px;'></body></html>"

        def generate_frames():
            while self.running:
                with self.lock:
                    if self.latest_frame is None:
                        time.sleep(0.05)
                        continue
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', self.latest_frame)
                    frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.03) # Cap at ~30 FPS to save bandwidth

        @app.route('/video_feed')
        def video_feed():
            return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

        # Suppress Flask default HTTP logging for cleaner console output
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        # Run Flask in a background daemon thread
        flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False))
        flask_thread.daemon = True
        flask_thread.start()
        print("[WEB] Flask server streaming at http://0.0.0.0:5000")

    def run(self) -> None:
        """Main detection loop. Press 'q' or Ctrl+C to exit."""
        print("=" * 60)
        print("  DROWSINESS DETECTION SYSTEM – RUNNING")
        print("  Press 'q' to quit")
        print("  Press 'd' to toggle DEMO MODE (simulate drowsiness)")
        print("  Press 'r' to reset calibration")
        print("=" * 60)

        demo_mode = False  # toggle with 'd' key for professor demo

        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("[CAM] Frame read failed – retrying...")
                    time.sleep(0.1)
                    continue

                self.frame_num += 1
                self._update_fps()

                # --- Detect face and eyes ---
                detections = self.detector.detect(frame)

                if "face_rect" not in detections:
                    # No face – draw message, pause detection
                    ear, degradation, drowsiness, level = 0.0, 0.0, 0.0, "normal"
                    self.gpio.set_alert_state("normal")
                    if self.show_display or self.web_stream:
                        self.renderer.draw_no_face(frame)
                        self.renderer.draw_status_bar(
                            frame, self.fps, ear,
                            self.blink_det.blink_count, degradation,
                            drowsiness, level,
                            self.blink_det.calibrated,
                            self.blink_det.calibration_progress,
                        )
                else:
                    # Face found – process eyes
                    face_rect = detections["face_rect"]
                    left_pts = detections.get("left_eye_pts")
                    right_pts = detections.get("right_eye_pts")

                    # Compute EAR from each eye, average
                    ears = []
                    if left_pts is not None:
                        ears.append(compute_ear(left_pts))
                    if right_pts is not None:
                        ears.append(compute_ear(right_pts))

                    if ears:
                        # We got actual eye measurements
                        raw_ear = sum(ears) / len(ears)
                        self.last_valid_ear = raw_ear
                        self.no_eye_frames = 0
                    else:
                        # Eyes NOT detected — cascade flickered.
                        # Carry forward last valid EAR (don't assume closed!)
                        self.no_eye_frames += 1
                        if self.no_eye_frames > 20:
                            # Eyes missing for 20+ frames — likely actually closed
                            raw_ear = 0.05
                        else:
                            raw_ear = self.last_valid_ear
                        if self.cfg.console_debug and self.no_eye_frames == 1:
                            print(f"[EYE] Cascade miss, using last EAR={raw_ear:.3f}")

                    # Smooth EAR to prevent sudden jumps from cascade noise
                    smooth_alpha = 0.4  # higher = more responsive
                    self.ear_smooth = (smooth_alpha * raw_ear +
                                       (1 - smooth_alpha) * self.ear_smooth)
                    ear = self.ear_smooth

                    # DEMO MODE: override EAR to simulate drowsiness
                    if demo_mode:
                        ear = 0.05
                        self.ear_smooth = 0.05

                    # Validate EAR
                    ear = max(0.0, min(ear, 1.0))

                    # Update blink detector
                    blink_ended, degradation = self.blink_det.update(ear)

                    # Compute drowsiness score (pass sustained closure count)
                    drowsiness = self.scorer.compute(
                        ear, degradation,
                        sustained_closed=self.blink_det.sustained_closed
                    )
                    level = self.scorer.get_level(drowsiness)

                    # Trigger alerts
                    alert_triggered = False
                    now = time.time()
                    if level == "critical":
                        if now - self.last_alert_time > self.cfg.alert_cooldown_sec:
                            self.gpio.set_alert_state("critical")
                            self.last_alert_time = now
                            alert_triggered = True
                            if self.cfg.console_debug:
                                print(f"[ALERT] CRITICAL – drowsiness "
                                      f"{drowsiness * 100:.0f}%")
                    elif level == "warning":
                        self.gpio.set_alert_state("warning")
                    else:
                        self.gpio.set_alert_state("normal")

                    # Log event
                    self.logger.log(self.frame_num, ear, degradation,
                                    drowsiness, alert_triggered)

                    # Console debug output
                    if self.cfg.console_debug and self.frame_num % 30 == 0:
                        print(f"[F{self.frame_num}] EAR={ear:.3f} "
                              f"Deg={degradation * 100:.1f}% "
                              f"Drowsy={drowsiness * 100:.0f}% [{level}]")

                    # --- Draw UI ---
                    if self.show_display or self.web_stream:
                        color = LEVEL_COLOR.get(level, COLORS["white"])
                        self.renderer.draw_face_rect(frame, face_rect, color)

                        for key in ("left_eye_pts", "right_eye_pts"):
                            if key in detections:
                                self.renderer.draw_eye_points(
                                    frame, detections[key], COLORS["cyan"])

                        self.renderer.draw_status_bar(
                            frame, self.fps, ear,
                            self.blink_det.blink_count, degradation,
                            drowsiness, level,
                            self.blink_det.calibrated,
                            self.blink_det.calibration_progress,
                        )

                        if level == "critical":
                            self.renderer.draw_alert_flash(frame)

                # Store latest frame for web stream
                if getattr(self, "web_stream", False):
                    with self.lock:
                        self.latest_frame = frame.copy()

                # Show frame
                if self.show_display:
                    cv2.imshow(self.cfg._d.get("window_name",
                               "Drowsiness Detector"), frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("\n[QUIT] User pressed 'q'.")
                        break
                    elif key == ord("d"):
                        demo_mode = not demo_mode
                        state = "ON" if demo_mode else "OFF"
                        print(f"\n[DEMO] Demo mode {state}")
                    elif key == ord("r"):
                        self.blink_det = BlinkDetector(self.cfg)
                        self.scorer = DrowsinessScorer(self.cfg)
                        print("\n[RESET] Calibration reset!")

        except KeyboardInterrupt:
            print("\n[QUIT] Ctrl+C received.")
        finally:
            self.shutdown()

    def _update_fps(self) -> None:
        self._fps_frames += 1
        elapsed = time.time() - self._fps_start
        if elapsed >= 1.0:
            self.fps = self._fps_frames / elapsed
            self._fps_frames = 0
            self._fps_start = time.time()

    def shutdown(self) -> None:
        """Graceful cleanup of all resources."""
        print("[SHUTDOWN] Cleaning up...")
        self.running = False
        self.gpio.buzzer_off()
        self.gpio.cleanup()
        self.logger.flush()
        if hasattr(self.cap, "release"):
            self.cap.release()
        cv2.destroyAllWindows()
        print("[SHUTDOWN] Done. Goodbye!")


# ===========================================================================
#  ENTRY POINT
# ===========================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drowsiness Detection System with Blink Degradation Index"
    )
    parser.add_argument("--mock", action="store_true",
                        help="Use mock camera (synthetic frames)")
    parser.add_argument("--no-display", action="store_true",
                        help="Run headless (no video window)")
    parser.add_argument("--web-stream", action="store_true",
                        help="Start Flask server to stream video over network")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose console output")
    parser.add_argument("--config", type=str,
                        default="drowsiness_detector_config.json",
                        help="Path to config JSON file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    cfg = Config(args.config)

    # CLI overrides
    if args.mock:
        cfg._d["mock_mode"] = True
    if args.no_display:
        cfg._d["show_video"] = False
    if args.web_stream:
        cfg._d["web_stream"] = True
    if args.debug:
        cfg._d["console_debug"] = True

    # Register SIGINT for clean shutdown
    detector = DrowsinessDetector(
        cfg,
        use_mock=cfg.mock_mode,
        show_display=cfg.show_video,
    )

    def sig_handler(sig, frame):
        detector.running = False

    signal.signal(signal.SIGINT, sig_handler)

    detector.run()


if __name__ == "__main__":
    main()
