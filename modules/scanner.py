"""
CIS v2 - Scanner Orchestrator
Manages the 3-scan pipeline, coordinates all analysis modules,
emits progress callbacks for the UI, and returns the final FusionResult.
"""
import cv2
import numpy as np
import threading
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import WEBCAM_SCAN_FRAMES, WEBCAM_TOTAL_SCANS, SCAN_INTERVAL_SEC
from modules.face_analyzer import FaceAnalyzer, FaceFeatures
from modules.gait_analyzer import GaitAnalyzer, GaitFeatures
from modules.behavior_analyzer import BehaviorAnalyzer, BehaviorFeatures
from modules.fusion_engine import FusionEngine, FusionResult
from database.db_manager import log_detection_event


class ScanResult:
    """Full output of one complete scanning cycle."""
    def __init__(self):
        self.fusion      : FusionResult = None
        self.face        : FaceFeatures = None
        self.gait        : GaitFeatures = None
        self.behavior    : BehaviorFeatures = None
        self.scan_frames : list  = []    # (frame, label) tuples for display
        self.mode        : str   = ""
        self.input_source: str   = ""
        self.timestamp   : str   = ""
        self.success     : bool  = False
        self.error       : str   = ""

    @property
    def verdict(self) -> str:
        return self.fusion.verdict if self.fusion else "CLEAR"

    @property
    def risk_level(self) -> str:
        return self.fusion.risk_level if self.fusion else "LOW"

    @property
    def fusion_conf(self) -> float:
        return self.fusion.fusion_conf if self.fusion else 0.0


class Scanner:
    """
    Orchestrates the multi-scan identification pipeline.

    on_progress callback signature:
        callback(scan_num: int, total_scans: int, label: str, frame: np.ndarray)
    on_complete callback signature:
        callback(scan_result: ScanResult)
    """

    def __init__(self, on_progress=None, on_complete=None, on_frame=None):
        self.on_progress = on_progress   # Progress callback
        self.on_complete = on_complete   # Final result callback
        self.on_frame    = on_frame      # Per-frame display callback

        self.face_analyzer     = FaceAnalyzer()
        self.gait_analyzer     = GaitAnalyzer()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.fusion_engine     = FusionEngine()

        self._stop_flag = threading.Event()

    def stop(self):
        """Signal the scanner to stop."""
        self._stop_flag.set()

    def reset(self):
        """Reset all modules for a fresh scan."""
        self._stop_flag.clear()
        self.gait_analyzer.reset()
        self.behavior_analyzer.reset()
        self.fusion_engine.reset()

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry points
    # ──────────────────────────────────────────────────────────────────────────

    def scan_image(self, image_path: str) -> ScanResult:
        """Analyze a single image file."""
        from datetime import datetime
        self.reset()
        sr = ScanResult()
        sr.mode         = "image"
        sr.input_source = image_path
        sr.timestamp    = datetime.now().isoformat()

        try:
            frame = cv2.imread(image_path)
            if frame is None:
                sr.error = f"Could not read image: {image_path}"
                return sr

            self._emit_progress(1, 1, "Analyzing image...", frame)

            # Run all modules on the single frame
            ff = self.face_analyzer.analyze(frame)
            gf = self.gait_analyzer.analyze_frame(frame)
            bf = self.behavior_analyzer.analyze_frame(frame, face_bbox=ff.bbox)

            fr = self.fusion_engine.fuse(
                face_conf      = ff.confidence,
                gait_conf      = gf.confidence,
                behavior_conf  = bf.confidence,
                face_features  = ff.feature_vector,
                face_available = ff.detected,
                gait_available = gf.detected,
                behavior_available = bf.detected,
                scan_num       = 1,
            )

            sr.fusion   = fr
            sr.face     = ff
            sr.gait     = gf
            sr.behavior = bf
            sr.success  = True

            # Log to DB
            self._log(sr)
            self._emit_complete(sr)

        except Exception as e:
            sr.error = str(e)
            import traceback
            traceback.print_exc()

        return sr

    def scan_video(self, video_path: str, on_frame_result=None) -> ScanResult:
        """Analyze a video file frame by frame."""
        from datetime import datetime
        from config.settings import VIDEO_SAMPLE_EVERY_N
        self.reset()
        sr = ScanResult()
        sr.mode         = "video"
        sr.input_source = video_path
        sr.timestamp    = datetime.now().isoformat()

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                sr.error = f"Cannot open video: {video_path}"
                return sr

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx    = 0
            analyzed     = 0
            ff_last = FaceFeatures(); gf_last = GaitFeatures(); bf_last = BehaviorFeatures()
            fr_last = FusionResult()
            face_features_best = None

            while not self._stop_flag.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                if frame_idx % VIDEO_SAMPLE_EVERY_N != 0:
                    # Still display the frame for smooth playback
                    if self.on_frame:
                        self.on_frame(frame, fr_last)
                    continue

                analyzed += 1
                progress = frame_idx / max(total_frames, 1)
                self._emit_progress(frame_idx, total_frames, f"Frame {frame_idx}/{total_frames}", frame)

                ff = self.face_analyzer.analyze(frame)
                gf = self.gait_analyzer.analyze_frame(frame)
                bf = self.behavior_analyzer.analyze_frame(frame, face_bbox=ff.bbox)

                fr = self.fusion_engine.fuse(
                    face_conf      = ff.confidence,
                    gait_conf      = gf.confidence,
                    behavior_conf  = bf.confidence,
                    face_features  = ff.feature_vector if ff.detected else None,
                    face_available = ff.detected,
                    gait_available = gf.detected,
                    behavior_available = bf.detected,
                    scan_num       = analyzed,
                )

                ff_last = ff; gf_last = gf; bf_last = bf; fr_last = fr
                if ff.detected and ff.quality > (face_features_best[1] if face_features_best else 0):
                    face_features_best = (ff.feature_vector, ff.quality)

                # Overlay and send frame to UI
                annotated = self.face_analyzer.draw_overlay(frame, ff, verdict=fr.verdict)
                self._add_hud(annotated, fr, frame_idx, total_frames)
                if self.on_frame:
                    self.on_frame(annotated, fr)

                if on_frame_result:
                    on_frame_result(fr)

            cap.release()

            # Aggregate all scan passes
            final_fr = self.fusion_engine.aggregate_scans()
            sr.fusion   = final_fr
            sr.face     = ff_last
            sr.gait     = gf_last
            sr.behavior = bf_last
            sr.success  = True
            self._log(sr)
            self._emit_complete(sr)

        except Exception as e:
            sr.error = str(e)
            import traceback
            traceback.print_exc()

        return sr

    @staticmethod
    def _open_camera(preferred_index: int = 0):
        """
        Try to open the first working camera with the best backend for this OS.
        - On Windows: tries DirectShow (CAP_DSHOW) first for laptop webcams, then ANY.
        - preferred_index=-1  → auto-scan all indices 0..4.
        - Uses MJPEG codec hint for faster laptop camera startup on Windows.
        - Retries frame read up to 10 times for slow-starting cameras.
        Returns (cap, actual_index) or (None, -1).
        """
        import platform
        is_windows = platform.system() == "Windows"

        if preferred_index < 0:
            indices_to_try = list(range(5))
        else:
            indices_to_try = [preferred_index] + [i for i in range(5) if i != preferred_index]

        backends = ([cv2.CAP_DSHOW, cv2.CAP_ANY] if is_windows else [cv2.CAP_ANY])

        for idx in indices_to_try:
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(idx, backend)
                    if not cap.isOpened():
                        cap.release()
                        continue

                    # Set resolution and framerate
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    # Prefer MJPEG for faster laptop-cam startup
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

                    # Retry up to 10 frames — some cameras take time to warm up
                    for _ in range(10):
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            bname = 'DSHOW' if backend == cv2.CAP_DSHOW else 'ANY'
                            print(f"[Camera] Opened index={idx} backend={bname} "
                                  f"res={frame.shape[1]}x{frame.shape[0]}")
                            return cap, idx
                    cap.release()
                except Exception as exc:
                    print(f"[Camera] Error trying index={idx}: {exc}")
        return None, -1

    def scan_webcam(self, camera_index: int = 0) -> ScanResult:
        """
        Perform the 3-scan webcam cycle.
        Each scan = WEBCAM_SCAN_FRAMES frames; pauses SCAN_INTERVAL_SEC between scans.
        """
        from datetime import datetime
        self.reset()
        sr = ScanResult()
        sr.mode         = "webcam"
        sr.input_source = f"CAM:{camera_index}"
        sr.timestamp    = datetime.now().isoformat()

        try:
            cap, actual_index = self._open_camera(camera_index)
            if cap is None:
                sr.error = ("No camera found. Please ensure your webcam is connected "
                            "and not being used by another app (e.g. Teams, Zoom, Skype).")
                self._emit_complete(sr)
                return sr
            sr.input_source = f"CAM:{actual_index}"

            # Send a warm-up frame immediately so the UI shows the camera
            for _ in range(5):
                ret, warm_frame = cap.read()
                if ret and self.on_frame:
                    self.on_frame(warm_frame, None)

            # 3 scan passes
            for scan_num in range(1, WEBCAM_TOTAL_SCANS + 1):
                if self._stop_flag.is_set():
                    break

                scan_label = f"SCAN {scan_num}/{WEBCAM_TOTAL_SCANS}"
                self._emit_progress(scan_num, WEBCAM_TOTAL_SCANS, scan_label, None)

                ff_best = None
                gf_last = GaitFeatures()
                bf_last = BehaviorFeatures()

                for f_idx in range(WEBCAM_SCAN_FRAMES):
                    if self._stop_flag.is_set():
                        break

                    ret, frame = cap.read()
                    if not ret:
                        break

                    ff = self.face_analyzer.analyze(frame)
                    gf = self.gait_analyzer.analyze_frame(frame)
                    bf = self.behavior_analyzer.analyze_frame(frame, face_bbox=ff.bbox)

                    # Keep best-quality face
                    if ff.detected and (ff_best is None or ff.quality > ff_best.quality):
                        ff_best = ff

                    gf_last = gf
                    bf_last = bf

                    # Annotate frame for display
                    annotated = self.face_analyzer.draw_overlay(frame, ff, scan_label=scan_label)
                    self._add_scan_hud(annotated, scan_num, f_idx, WEBCAM_SCAN_FRAMES)
                    if self.on_frame:
                        self.on_frame(annotated, None)

                # Run fusion for this scan pass
                ff_use = ff_best if ff_best else FaceFeatures()
                fr = self.fusion_engine.fuse(
                    face_conf      = ff_use.confidence,
                    gait_conf      = gf_last.confidence,
                    behavior_conf  = bf_last.confidence,
                    face_features  = ff_use.feature_vector if ff_use.detected else None,
                    face_available = ff_use.detected,
                    gait_available = gf_last.detected,
                    behavior_available = bf_last.detected,
                    scan_num       = scan_num,
                )

                # Brief pause between scans
                if scan_num < WEBCAM_TOTAL_SCANS:
                    pause_end = time.time() + SCAN_INTERVAL_SEC
                    while time.time() < pause_end and not self._stop_flag.is_set():
                        ret, frame = cap.read()
                        if ret:
                            # Show "PAUSE" screen between scans
                            _f = frame.copy()
                            cv2.rectangle(_f, (0, 0), (_f.shape[1], 60), (30, 30, 30), -1)
                            cv2.putText(_f, f"Scan {scan_num} complete. Preparing scan {scan_num+1}...",
                                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
                            if self.on_frame:
                                self.on_frame(_f, fr)
                        time.sleep(0.05)

            cap.release()

            # Final aggregated result
            final_fr = self.fusion_engine.aggregate_scans()
            sr.fusion   = final_fr
            sr.face     = ff_use if 'ff_use' in dir() else FaceFeatures()
            sr.gait     = gf_last if 'gf_last' in dir() else GaitFeatures()
            sr.behavior = bf_last if 'bf_last' in dir() else BehaviorFeatures()
            sr.success  = True
            self._log(sr)
            self._emit_complete(sr)

        except Exception as e:
            sr.error = str(e)
            import traceback
            traceback.print_exc()

        return sr

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _emit_progress(self, current, total, label, frame):
        if self.on_progress:
            try:
                self.on_progress(current, total, label, frame)
            except Exception:
                pass

    def _emit_complete(self, sr: ScanResult):
        if self.on_complete:
            try:
                self.on_complete(sr)
            except Exception:
                pass

    def _log(self, sr: ScanResult):
        """Log this scan to the database."""
        try:
            fr = sr.fusion
            event = {
                "criminal_id":  fr.suspect_id,
                "mode":         sr.mode,
                "face_conf":    fr.face_conf,
                "gait_conf":    fr.gait_conf,
                "behavior_conf":fr.behavior_conf,
                "fusion_conf":  fr.fusion_conf,
                "verdict":      fr.verdict,
                "risk_level":   fr.risk_level,
                "input_source": sr.input_source,
                "notes":        f"Match: {fr.suspect_name or 'None'} | DB sim: {fr.db_similarity:.2%}",
            }
            log_detection_event(event)
        except Exception:
            pass

    def _add_hud(self, frame: np.ndarray, fr: FusionResult, frame_num: int, total: int):
        """Add HUD overlay for video mode."""
        h, w = frame.shape[:2]
        # Progress bar
        prog = frame_num / max(total, 1)
        bar_w = int(w * prog)
        cv2.rectangle(frame, (0, h - 6), (w, h), (30, 30, 30), -1)
        color = (0, 0, 255) if fr.verdict == "CRIMINAL" else (0, 165, 255) if fr.verdict == "WATCH LIST" else (0, 200, 60)
        cv2.rectangle(frame, (0, h - 6), (bar_w, h), color, -1)
        # Top label
        cv2.rectangle(frame, (0, 0), (340, 38), (0, 0, 0), -1)
        label = f"{fr.verdict}  {fr.fusion_conf:.0%}"
        cv2.putText(frame, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    def _add_scan_hud(self, frame: np.ndarray, scan_num: int, frame_idx: int, total_frames: int):
        """Add HUD overlay for webcam scan mode."""
        h, w = frame.shape[:2]
        progress = frame_idx / max(total_frames, 1)
        bar_w = int(w * progress)
        # Scanning progress bar (orange)
        cv2.rectangle(frame, (0, h - 8), (w, h), (20, 20, 20), -1)
        cv2.rectangle(frame, (0, h - 8), (bar_w, h), (0, 165, 255), -1)
        # Scan counter top-right
        cv2.rectangle(frame, (w - 180, 0), (w, 45), (0, 0, 0), -1)
        cv2.putText(frame, f"SCAN {scan_num}/{WEBCAM_TOTAL_SCANS}", (w - 170, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
