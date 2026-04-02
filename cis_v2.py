"""
CIS v2 — Criminal Identification System (Enhanced)
Main 4-tab GUI Application
Team: BioFuse | Theme: AI for Public Safety

Tabs:
  1. Image Upload   — Analyze static images
  2. Video Upload   — Process video files frame-by-frame
  3. Live Webcam    — 3-scan webcam pipeline
  4. Database       — Criminal registry + detection log

Requires: opencv-python pillow numpy
Optional:  mediapipe (for gait analysis)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
import sys
import queue
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config.settings import *
from database.db_manager import (
    init_db, get_all_criminals, get_recent_detections,
    get_stats, get_simple_image_verdict,
    search_by_face_features, add_criminal_to_db, register_criminal_face
)
from modules.scanner import Scanner, ScanResult
from modules.fusion_engine import FusionResult


# ═════════════════════════════════════════════════════════════════════════════
#  Helper: Theme colours
# ═════════════════════════════════════════════════════════════════════════════

C = {
    "bg":       "#0d0d1a",       # Darkest background
    "bg2":      "#141428",       # Card background
    "bg3":      "#1e1e3a",       # Button / panel backgrounds
    "bg4":      "#252545",       # Slightly lighter panel
    "accent":   "#00d4ff",       # Cyan accent
    "red":      "#ff3355",       # CRIMINAL
    "orange":   "#ff9900",       # WATCH LIST
    "green":    "#00e676",       # CLEAR
    "yellow":   "#ffd600",
    "purple":   "#7c4dff",
    "white":    "#f0f0f0",
    "gray":     "#7a7a9a",
    "txt":      "#e0e0f0",
    "gold":     "#FFD700",       # Vivid gold for registry actions
}

VERDICT_COLORS = {
    "CRIMINAL":   C["red"],
    "SUSPECT":    C["orange"],
    "WATCH LIST": C["orange"],
    "INNOCENT":   C["green"],
    "CLEAR":      C["green"],
}


# ═════════════════════════════════════════════════════════════════════════════
#  Utility: PIL-based canvas drawing
# ═════════════════════════════════════════════════════════════════════════════

def make_placeholder_frame(w, h, text="NO SIGNAL", color=C["accent"]):
    """Generate a placeholder camera-off frame."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Grid pattern
    for i in range(0, h, 30):
        cv2.line(img, (0, i), (w, i), (20, 20, 40), 1)
    for j in range(0, w, 30):
        cv2.line(img, (j, 0), (j, h), (20, 20, 40), 1)
    # Centre text
    cv2.putText(img, text, (w // 2 - len(text) * 9, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 220), 2)
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), (30, 30, 60), 2)
    return img


def frame_to_photoimage(frame: np.ndarray, w: int, h: int) -> ImageTk.PhotoImage:
    """Convert BGR numpy frame to Tkinter PhotoImage scaled to (w, h)."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img   = Image.fromarray(frame_rgb).resize((w, h), Image.LANCZOS)
    return ImageTk.PhotoImage(image=pil_img)


def verdict_icon(verdict: str) -> str:
    return {
        "CRIMINAL":   "🚨",
        "SUSPECT":    "⚠️",
        "WATCH LIST": "⚠️",
        "INNOCENT":   "✅",
        "CLEAR":      "✅",
    }.get(verdict, "❓")


# ═════════════════════════════════════════════════════════════════════════════
#  Reusable UI Widgets
# ═════════════════════════════════════════════════════════════════════════════

class ConfidenceBar(tk.Frame):
    """Animated horizontal confidence bar with label."""

    def __init__(self, parent, label: str, color: str, **kwargs):
        super().__init__(parent, bg=C["bg2"], **kwargs)
        self._label  = label
        self._color  = color
        self._value  = 0.0
        self._target = 0.0

        tk.Label(self, text=label, bg=C["bg2"], fg=C["gray"],
                 font=("Segoe UI", 9)).pack(anchor="w")

        bar_row = tk.Frame(self, bg=C["bg2"])
        bar_row.pack(fill="x")

        self._bar_bg = tk.Frame(bar_row, bg=C["bg3"], height=14)
        self._bar_bg.pack(fill="x", side="left", expand=True, padx=(0, 8))

        self._bar_fill = tk.Frame(self._bar_bg, bg=color, height=14)
        self._bar_fill.place(relx=0, rely=0, relwidth=0, relheight=1)

        self._pct_lbl = tk.Label(bar_row, text="0%", bg=C["bg2"], fg=color,
                                  font=("Segoe UI", 9, "bold"), width=5)
        self._pct_lbl.pack(side="right")

    def set_value(self, value: float):
        """Animate bar to value (0–1)."""
        self._target = max(0.0, min(1.0, value))
        self._animate()

    def _animate(self):
        step = 0.03
        if abs(self._value - self._target) > step:
            self._value += step if self._target > self._value else -step
            self._bar_fill.place_configure(relwidth=self._value)
            self._pct_lbl.config(text=f"{self._value:.0%}")
            self.after(16, self._animate)
        else:
            self._value = self._target
            self._bar_fill.place_configure(relwidth=self._value)
            self._pct_lbl.config(text=f"{self._value:.0%}")


class VerdictPanel(tk.Frame):
    """Large verdict display panel with confidence breakdown."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=C["bg2"], **kwargs)
        self._build()

    def _build(self):
        # Title
        self._title = tk.Label(self, text="AWAITING SCAN", font=("Segoe UI", 18, "bold"),
                                bg=C["bg2"], fg=C["gray"])
        self._title.pack(pady=(18, 4))

        # Icon + Score
        row = tk.Frame(self, bg=C["bg2"])
        row.pack()
        self._icon  = tk.Label(row, text="❓", font=("Segoe UI", 40), bg=C["bg2"], fg=C["gray"])
        self._icon.pack(side="left", padx=10)
        self._score = tk.Label(row, text="—.—%", font=("Segoe UI", 38, "bold"),
                                bg=C["bg2"], fg=C["gray"])
        self._score.pack(side="left")

        # Suspect info
        self._suspect = tk.Label(self, text="", font=("Segoe UI", 11),
                                  bg=C["bg2"], fg=C["accent"])
        self._suspect.pack(pady=(4, 0))
        self._crime = tk.Label(self, text="", font=("Segoe UI", 9),
                                bg=C["bg2"], fg=C["gray"])
        self._crime.pack()

        # Separator
        tk.Frame(self, bg=C["bg3"], height=1).pack(fill="x", padx=20, pady=12)

        # Confidence bars
        bars_frame = tk.Frame(self, bg=C["bg2"])
        bars_frame.pack(fill="x", padx=20)

        self._bar_face  = ConfidenceBar(bars_frame, "Face Recognition", C["accent"])
        self._bar_face.pack(fill="x", pady=3)
        self._bar_gait  = ConfidenceBar(bars_frame, "Gait Analysis", C["purple"])
        self._bar_gait.pack(fill="x", pady=3)
        self._bar_behav = ConfidenceBar(bars_frame, "Behavioral Analysis", C["yellow"])
        self._bar_behav.pack(fill="x", pady=3)
        self._bar_fused = ConfidenceBar(bars_frame, "FUSION SCORE", C["white"])
        self._bar_fused.pack(fill="x", pady=(10, 3))

        # Reasoning log
        tk.Frame(self, bg=C["bg3"], height=1).pack(fill="x", padx=20, pady=8)
        tk.Label(self, text="ANALYSIS LOG", font=("Segoe UI", 8, "bold"),
                 bg=C["bg2"], fg=C["gray"]).pack(anchor="w", padx=20)
        self._log = tk.Text(self, bg=C["bg"], fg=C["gray"], font=("Consolas", 8),
                             height=6, wrap="word", relief="flat", state="disabled")
        self._log.pack(fill="x", padx=20, pady=(4, 16))

    def update_result(self, fr: FusionResult):
        verdict = fr.verdict
        color   = VERDICT_COLORS.get(verdict, C["gray"])
        icon    = verdict_icon(verdict)

        self._title.config(text=verdict, fg=color)
        self._icon.config(text=icon, fg=color)
        self._score.config(text=f"{fr.fusion_conf:.0%}", fg=color)

        if fr.suspect_name:
            alias = f" ({fr.suspect_alias})" if fr.suspect_alias else ""
            self._suspect.config(text=f"⚠ Suspect: {fr.suspect_name}{alias}")
            self._crime.config(text=f"Crime: {fr.crime_type or 'Unknown'}  |  Risk: {fr.risk_level}")
        else:
            self._suspect.config(text="No database match")
            self._crime.config(text="")

        self._bar_face.set_value(fr.face_conf)
        self._bar_gait.set_value(fr.gait_conf)
        self._bar_behav.set_value(fr.behavior_conf)
        self._bar_fused.set_value(fr.fusion_conf)

        # Reasoning log
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        for line in fr.reasoning:
            self._log.insert("end", f"• {line}\n")
        self._log.config(state="disabled")

    def set_scanning(self, scan_label: str = "SCANNING..."):
        self._title.config(text=scan_label, fg=C["accent"])
        self._icon.config(text="🔍", fg=C["accent"])
        self._score.config(text="—", fg=C["accent"])
        self._suspect.config(text="")
        self._crime.config(text="")
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")


class LogPanel(tk.Frame):
    """Scrollable activity log."""

    def __init__(self, parent, title="ACTIVITY LOG", **kwargs):
        super().__init__(parent, bg=C["bg2"], **kwargs)
        tk.Label(self, text=title, bg=C["bg2"], fg=C["gray"],
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=10, pady=(8, 2))
        self._txt = tk.Text(self, bg=C["bg"], fg=C["txt"], font=("Consolas", 8),
                             wrap="word", relief="flat", state="disabled")
        self._txt.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        sb = ttk.Scrollbar(self._txt, command=self._txt.yview)
        self._txt.config(yscrollcommand=sb.set)

        # Tag colours
        self._txt.tag_config("criminal", foreground=C["red"])
        self._txt.tag_config("suspect",  foreground=C["orange"])
        self._txt.tag_config("watchlist", foreground=C["orange"])
        self._txt.tag_config("innocent", foreground=C["green"])
        self._txt.tag_config("clear",    foreground=C["green"])
        self._txt.tag_config("info",     foreground=C["accent"])
        self._txt.tag_config("error",    foreground="#ff6060")

    def log(self, message: str, tag: str = "info"):
        ts = datetime.now().strftime("%H:%M:%S")
        self._txt.config(state="normal")
        self._txt.insert("end", f"[{ts}] {message}\n", tag)
        self._txt.see("end")
        self._txt.config(state="disabled")


# ═════════════════════════════════════════════════════════════════════════════
#  Tab 1: Image Upload
# ═════════════════════════════════════════════════════════════════════════════

class ImageTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=C["bg"])
        self.app     = app
        self._image_path = None
        self._scanner    = Scanner(
            on_complete = self._on_complete,
        )
        self._build()

    def _build(self):
        # Split layout
        left  = tk.Frame(self, bg=C["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(15, 8), pady=15)
        right = tk.Frame(self, bg=C["bg2"], width=360)
        right.pack(side="right", fill="y", padx=(0, 15), pady=15)
        right.pack_propagate(False)

        # ── Left: Preview canvas ──────────────────────────────────────────────
        tk.Label(left, text="IMAGE ANALYSIS", bg=C["bg"], fg=C["accent"],
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))

        self._canvas = tk.Canvas(left, bg=C["bg3"], highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Configure>", self._refresh_canvas)

        self._preview_img = None  # Hold PhotoImage ref

        # Drop zone overlay
        self._canvas.create_text(
            400, 260, text="📁  DROP IMAGE OR CLICK BROWSE",
            fill=C["gray"], font=("Segoe UI", 14), tags="placeholder"
        )

        # ── Bottom buttons ────────────────────────────────────────────────────
        btn_row = tk.Frame(left, bg=C["bg"])
        btn_row.pack(fill="x", pady=(8, 0))

        self._btn_browse = self._btn(btn_row, "\U0001f4c2 Browse Image", self._browse, C["bg3"])
        self._btn_browse.pack(side="left", padx=(0, 8))
        self._btn_analyze = self._btn(btn_row, "\U0001f50d Analyze", self._analyze, C["purple"], state="disabled")
        self._btn_analyze.pack(side="left")

        # Make Register Face button stand out (Gold) and always visible as an 'option'
        self._btn_register = self._btn(btn_row, "\U0001f517 Register Face",
                                       self._open_register_dialog, C["gold"], state="disabled")
        self._btn_register.config(fg="black") # Better contrast on gold
        self._btn_register.pack(side="left", padx=(8, 0))

        self._status_lbl = tk.Label(btn_row, text="", bg=C["bg"], fg=C["gray"],
                                     font=("Segoe UI", 9))
        self._status_lbl.pack(side="left", padx=12)

        # ── Right: Verdict panel ──────────────────────────────────────────────
        self._verdict = VerdictPanel(right)
        self._verdict.pack(fill="both", expand=True, padx=5, pady=5)

    def _btn(self, parent, text, cmd, bg, state="normal"):
        return tk.Button(parent, text=text, command=cmd,
                         bg=bg, fg="white", font=("Segoe UI", 10, "bold"),
                         relief="flat", padx=16, pady=7, cursor="hand2",
                         activebackground=C["accent"], activeforeground="black",
                         state=state)

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")]
        )
        if path:
            self._image_path = path
            self._show_preview(path)
            self._btn_analyze.config(state="normal")
            self._status_lbl.config(text=os.path.basename(path), fg=C["txt"])

    def _show_preview(self, path):
        try:
            frame = cv2.imread(path)
            if frame is not None:
                self._raw_frame = frame
                self._refresh_canvas()
        except Exception:
            pass

    def _refresh_canvas(self, event=None):
        if not hasattr(self, "_raw_frame") or self._raw_frame is None:
            return
        try:
            cw = self._canvas.winfo_width() or 700
            ch = self._canvas.winfo_height() or 500
            if cw < 10 or ch < 10:
                return
            ph = frame_to_photoimage(self._raw_frame, cw, ch)
            self._preview_img = ph
            self._canvas.delete("all")
            self._canvas.create_image(0, 0, anchor="nw", image=ph)
        except Exception:
            pass

    def _analyze(self):
        if not self._image_path:
            return
        self._btn_analyze.config(state="disabled")
        self._status_lbl.config(text="Analyzing...", fg=C["accent"])
        self._verdict.set_scanning("ANALYZING IMAGE...")
        threading.Thread(target=self._run_scan, daemon=True).start()

    def _run_scan(self):
        self._scanner.reset()
        self._scanner.scan_image(self._image_path)

    def _open_register_dialog(self):
        """Open RegisterFaceDialog to tag the current face to a known criminal."""
        if hasattr(self, "_last_face_fv") and self._last_face_fv:
            RegisterFaceDialog(
                self.winfo_toplevel(),
                self._last_face_fv,
                on_registered=lambda name: (
                    self._verdict._suspect.config(
                        text=f"\u2705 Registered as: {name}",
                        fg=C["green"]
                    ),
                    self._verdict._crime.config(
                        text="Face biometric saved. Future uploads will match correctly.",
                        fg=C["gray"]
                    ),
                    self.app._tab_db.refresh(),
                )
            )

    def _on_complete(self, sr: ScanResult):
        def update():
            if sr.success and sr.fusion:
                simple_verdict = get_simple_image_verdict(sr.fusion.fusion_conf)
                color = VERDICT_COLORS.get(simple_verdict, C["gray"])
                icon  = verdict_icon(simple_verdict)

                # ── DB face match (only fires for REAL registered faces) ────────
                db_match = None
                db_sim   = 0.0
                fv = sr.face.feature_vector if sr.face else []
                self._last_face_fv = fv  # store for Register Face button
                if fv:
                    matches = search_by_face_features(fv, top_k=1, min_similarity=0.82)
                    if matches:
                        db_match = matches[0]
                        db_sim   = db_match["similarity"]

                # ── VerdictPanel ──────────────────────────────────────────────
                self._verdict._title.config(text=simple_verdict, fg=color)
                self._verdict._icon.config(text=icon, fg=color)
                self._verdict._score.config(text=f"{sr.fusion.fusion_conf:.0%}", fg=color)

                if db_match:
                    # Real registered match found
                    alias_txt = f" ({db_match['alias']})" if db_match.get("alias") else ""
                    self._verdict._suspect.config(
                        text=f"\U0001f6c8 {db_match['name']}{alias_txt}",
                        fg=C["accent"]
                    )
                    self._verdict._crime.config(
                        text=f"Crime: {db_match['crime_type']}  |  "
                             f"Risk: {db_match['risk_level']}  |  "
                             f"Similarity: {db_sim:.0%}",
                        fg=C["gray"]
                    )
                    self._btn_register.config(state="disabled")
                elif fv:
                    # Face detected but not in DB — prompt registration
                    self._verdict._suspect.config(
                        text="\u2754 Not in biometric database",
                        fg=C["orange"]
                    )
                    self._verdict._crime.config(
                        text="Click \U0001f517 Register Face to link this face to a known criminal.",
                        fg=C["gray"]
                    )
                    self._btn_register.config(state="normal")
                else:
                    # No face detected at all
                    self._verdict._suspect.config(
                        text="No face detected in image",
                        fg=C["gray"]
                    )
                    self._verdict._crime.config(text="", fg=C["gray"])
                    self._btn_register.config(state="disabled")

                self._verdict._bar_face.set_value(sr.fusion.face_conf)
                self._verdict._bar_gait.set_value(sr.fusion.gait_conf)
                self._verdict._bar_behav.set_value(sr.fusion.behavior_conf)
                self._verdict._bar_fused.set_value(sr.fusion.fusion_conf)

                self._verdict._log.config(state="normal")
                self._verdict._log.delete("1.0", "end")
                self._verdict._log.insert("end", f"Verdict  : {simple_verdict}\n")
                self._verdict._log.insert("end", f"Score    : {sr.fusion.fusion_conf:.0%}\n")
                if db_match:
                    self._verdict._log.insert(
                        "end", f"Identity : {db_match['name']} ({db_sim:.0%} match)\n"
                    )
                else:
                    self._verdict._log.insert(
                        "end", "Identity : Not registered\n"
                              "           Click Register Face to tag this person.\n"
                    )
                self._verdict._log.config(state="disabled")

                if sr.face and sr.face.detected and self._image_path:
                    frame = cv2.imread(self._image_path)
                    if frame is not None:
                        annotated = self._scanner.face_analyzer.draw_overlay(
                            frame, sr.face, verdict=simple_verdict
                        )
                        self._raw_frame = annotated
                        self._refresh_canvas()

                match_txt = f" \u2192 {db_match['name']}" if db_match else ""
                self._status_lbl.config(
                    text=f"{icon} {simple_verdict} ({sr.fusion.fusion_conf:.0%}){match_txt}",
                    fg=color
                )
                self.app.activity_log.log(
                    f"[IMAGE] {simple_verdict} {sr.fusion.fusion_conf:.0%}"
                    f"{match_txt} \u2014 {os.path.basename(self._image_path)}",
                    tag=simple_verdict.lower()
                )
            else:
                self._status_lbl.config(text=f"Error: {sr.error}", fg=C["red"])
            self._btn_analyze.config(state="normal")
        self.after(0, update)


# ═════════════════════════════════════════════════════════════════════════════
#  Tab 2: Video Upload
# ═════════════════════════════════════════════════════════════════════════════

class VideoTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=C["bg"])
        self.app = app
        self._video_path = None
        self._frame_q    = queue.Queue(maxsize=4)
        self._result_ref = [None]
        self._running    = False
        self._scanner    = Scanner(
            on_frame    = self._on_frame,
            on_complete = self._on_complete,
        )
        self._build()
        self.after(33, self._poll_frames)

    def _build(self):
        left  = tk.Frame(self, bg=C["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(15, 8), pady=15)
        right = tk.Frame(self, bg=C["bg2"], width=360)
        right.pack(side="right", fill="y", padx=(0, 15), pady=15)
        right.pack_propagate(False)

        tk.Label(left, text="VIDEO ANALYSIS", bg=C["bg"], fg=C["accent"],
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))

        self._canvas = tk.Canvas(left, bg=C["bg3"], highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)
        self._preview_img = None

        self._canvas.create_text(
            400, 260, text="📹  SELECT A VIDEO FILE TO BEGIN",
            fill=C["gray"], font=("Segoe UI", 14), tags="placeholder"
        )

        # Progress bar
        prog_frame = tk.Frame(left, bg=C["bg"])
        prog_frame.pack(fill="x", pady=(6, 0))
        tk.Label(prog_frame, text="Progress:", bg=C["bg"], fg=C["gray"],
                 font=("Segoe UI", 8)).pack(side="left")
        self._progress = ttk.Progressbar(prog_frame, mode="determinate", length=400)
        self._progress.pack(side="left", fill="x", expand=True, padx=8)
        self._prog_lbl = tk.Label(prog_frame, text="0%", bg=C["bg"], fg=C["gray"],
                                   font=("Segoe UI", 8), width=6)
        self._prog_lbl.pack(side="right")

        btn_row = tk.Frame(left, bg=C["bg"])
        btn_row.pack(fill="x", pady=(8, 0))

        self._btn_load = self._btn(btn_row, "📂 Load Video", self._load, C["bg3"])
        self._btn_load.pack(side="left", padx=(0, 8))
        self._btn_start = self._btn(btn_row, "▶ Start", self._start, C["green"], state="disabled")
        self._btn_start.pack(side="left", padx=(0, 8))
        self._btn_stop  = self._btn(btn_row, "⏹ Stop", self._stop,  C["red"], state="disabled")
        self._btn_stop.pack(side="left")
        self._status_lbl = tk.Label(btn_row, text="", bg=C["bg"], fg=C["gray"],
                                     font=("Segoe UI", 9))
        self._status_lbl.pack(side="left", padx=12)

        self._verdict = VerdictPanel(right)
        self._verdict.pack(fill="both", expand=True, padx=5, pady=5)

    def _btn(self, parent, text, cmd, bg, state="normal"):
        return tk.Button(parent, text=text, command=cmd,
                         bg=bg, fg="white", font=("Segoe UI", 10, "bold"),
                         relief="flat", padx=14, pady=7, cursor="hand2",
                         activebackground=C["accent"], activeforeground="black",
                         state=state)

    def _load(self):
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All", "*.*")]
        )
        if path:
            self._video_path = path
            self._btn_start.config(state="normal")
            self._status_lbl.config(text=os.path.basename(path), fg=C["txt"])

    def _start(self):
        if not self._video_path or self._running:
            return
        self._running = True
        self._scanner.reset()
        self._progress["value"] = 0
        self._verdict.set_scanning("PROCESSING VIDEO...")
        self._btn_start.config(state="disabled")
        self._btn_stop.config(state="normal")
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        def on_progress(cur, total, label, frame):
            pct = int(cur / max(total, 1) * 100)
            self.after(0, lambda: (
                self._progress.__setitem__("value", pct),
                self._prog_lbl.config(text=f"{pct}%"),
                self._status_lbl.config(text=label, fg=C["accent"]),
            ))
        self._scanner.on_progress = on_progress
        self._scanner.scan_video(self._video_path)

    def _stop(self):
        self._scanner.stop()
        self._running = False
        self._btn_start.config(state="normal")
        self._btn_stop.config(state="disabled")
        self._status_lbl.config(text="Stopped", fg=C["gray"])

    def _on_frame(self, frame: np.ndarray, fr):
        try:
            self._frame_q.put_nowait((frame, fr))
        except queue.Full:
            pass

    def _poll_frames(self):
        try:
            frame, fr = self._frame_q.get_nowait()
            cw = self._canvas.winfo_width() or 800
            ch = self._canvas.winfo_height() or 520
            if cw > 10 and ch > 10:
                ph = frame_to_photoimage(frame, cw, ch)
                self._preview_img = ph
                self._canvas.delete("all")
                self._canvas.create_image(0, 0, anchor="nw", image=ph)
        except queue.Empty:
            pass
        self.after(33, self._poll_frames)

    def _on_complete(self, sr: ScanResult):
        self._running = False
        def update():
            self._btn_start.config(state="normal")
            self._btn_stop.config(state="disabled")
            if sr.success and sr.fusion:
                self._verdict.update_result(sr.fusion)
                self._status_lbl.config(
                    text=f"Complete — {sr.fusion.verdict} ({sr.fusion.fusion_conf:.0%})",
                    fg=VERDICT_COLORS.get(sr.fusion.verdict, C["gray"])
                )
                self.app.activity_log.log(
                    f"[VIDEO] {sr.fusion.verdict} {sr.fusion.fusion_conf:.0%} — {os.path.basename(self._video_path)}",
                    tag=sr.fusion.verdict.lower().replace(" ", "")
                )
            else:
                self._status_lbl.config(text=f"Error: {sr.error}", fg=C["red"])
            self._progress["value"] = 100
        self.after(0, update)


# ═════════════════════════════════════════════════════════════════════════════
#  Tab 3: Live Webcam
# ═════════════════════════════════════════════════════════════════════════════

class WebcamTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=C["bg"])
        self.app          = app
        self._running     = False
        self._frame_q     = queue.Queue(maxsize=2)
        self._cam_index   = tk.IntVar(value=0)   # Selected camera index
        self._scanner = Scanner(
            on_frame    = self._on_frame,
            on_complete = self._on_complete,
            on_progress = self._on_progress,
        )
        self._build()
        self.after(33, self._poll_frames)
        self.after(800, self._auto_detect_camera)  # Auto-detect laptop cam on load

    def _build(self):
        left  = tk.Frame(self, bg=C["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(15, 8), pady=15)
        right = tk.Frame(self, bg=C["bg2"], width=380)
        right.pack(side="right", fill="y", padx=(0, 15), pady=15)
        right.pack_propagate(False)

        # Title row
        title_row = tk.Frame(left, bg=C["bg"])
        title_row.pack(fill="x", pady=(0, 8))
        tk.Label(title_row, text="LIVE WEBCAM — 3-SCAN PIPELINE", bg=C["bg"], fg=C["accent"],
                 font=("Segoe UI", 13, "bold")).pack(side="left")
        self._scan_status = tk.Label(title_row, text="", bg=C["bg"], fg=C["accent"],
                                      font=("Segoe UI", 11, "bold"))
        self._scan_status.pack(side="right")

        # Canvas
        self._canvas = tk.Canvas(left, bg=C["bg3"], highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)
        self._preview_img = None

        # Scan progress indicators
        scan_row = tk.Frame(left, bg=C["bg"])
        scan_row.pack(fill="x", pady=(6, 0))
        self._scan_dots = []
        for i in range(1, 4):
            dot = tk.Label(scan_row, text=f"●  Scan {i}", bg=C["bg"], fg=C["bg3"],
                           font=("Segoe UI", 10, "bold"))
            dot.pack(side="left", padx=12)
            self._scan_dots.append(dot)
        self._scan_timer = tk.Label(scan_row, text="", bg=C["bg"], fg=C["gray"],
                                     font=("Segoe UI", 9))
        self._scan_timer.pack(side="right")

        # Camera selector row
        cam_row = tk.Frame(left, bg=C["bg"])
        cam_row.pack(fill="x", pady=(6, 0))
        tk.Label(cam_row, text="📷 Camera:", bg=C["bg"], fg=C["gray"],
                 font=("Segoe UI", 9)).pack(side="left")
        self._cam_combo = ttk.Combobox(
            cam_row, textvariable=self._cam_index,
            values=["Auto", "0 (Built-in/Laptop)", "1", "2", "3", "4"],
            state="readonly", width=18, font=("Segoe UI", 9)
        )
        self._cam_combo.current(1)   # Default: index 0 (laptop built-in)
        self._cam_combo.pack(side="left", padx=(4, 10))
        self._cam_combo.bind("<<ComboboxSelected>>", self._on_cam_change)
        self._btn_test_cam = self._btn(cam_row, "🔌 Test Cam", self._test_camera, C["bg3"])
        self._btn_test_cam.pack(side="left", padx=(0, 12))
        self._cam_status_lbl = tk.Label(cam_row, text="", bg=C["bg"], fg=C["gray"],
                                         font=("Segoe UI", 9))
        self._cam_status_lbl.pack(side="left")

        # Buttons
        btn_row = tk.Frame(left, bg=C["bg"])
        btn_row.pack(fill="x", pady=(8, 0))
        self._btn_start = self._btn(btn_row, "▶ Start 3-Scan", self._start, C["purple"])
        self._btn_start.pack(side="left", padx=(0, 8))
        self._btn_stop  = self._btn(btn_row, "⏹ Stop",         self._stop, C["red"], state="disabled")
        self._btn_stop.pack(side="left", padx=(0, 8))
        self._btn_reset = self._btn(btn_row, "↺ Reset",         self._reset, C["bg3"])
        self._btn_reset.pack(side="left")
        self._status_lbl = tk.Label(btn_row, text="Ready. Click Start.", bg=C["bg"], fg=C["gray"],
                                     font=("Segoe UI", 9))
        self._status_lbl.pack(side="left", padx=12)

        # Right verdict panel
        self._verdict = VerdictPanel(right)
        self._verdict.pack(fill="both", expand=True, padx=5, pady=5)

        # Placeholder
        ph = make_placeholder_frame(800, 500, "CAMERA STANDBY")
        self._preview_img = frame_to_photoimage(ph, 800, 500)
        self._canvas.create_image(0, 0, anchor="nw", image=self._preview_img)

    def _btn(self, parent, text, cmd, bg, state="normal"):
        return tk.Button(parent, text=text, command=cmd,
                         bg=bg, fg="white", font=("Segoe UI", 10, "bold"),
                         relief="flat", padx=14, pady=7, cursor="hand2",
                         activebackground=C["accent"], activeforeground="black",
                         state=state)

    def _start(self):
        if self._running:
            return
        self._running = True
        self._scanner.reset()
        self._reset_scan_dots()
        self._verdict.set_scanning("SCAN 1/3 INITIATING...")
        self._btn_start.config(state="disabled")
        self._btn_stop.config(state="normal")
        self._status_lbl.config(text="Scanning...", fg=C["accent"])
        threading.Thread(target=self._run, daemon=True).start()

    def _get_cam_index(self) -> int:
        """Parse the selected combobox value to a camera index integer."""
        val = self._cam_combo.get()
        if val.startswith("Auto"):
            return -1   # -1 = let scanner auto-detect
        try:
            return int(val.split(" ")[0])
        except Exception:
            return 0

    def _on_cam_change(self, event=None):
        self._cam_status_lbl.config(text="", fg=C["gray"])

    def _auto_detect_camera(self):
        """Silently try cam 0 at startup and show green tick if found."""
        def probe():
            import cv2 as _cv2
            for backend in [_cv2.CAP_DSHOW, _cv2.CAP_ANY]:
                cap = _cv2.VideoCapture(0, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        self.after(0, lambda: self._cam_status_lbl.config(
                            text="✓ Laptop cam detected (index 0)", fg=C["green"]))
                        return
            self.after(0, lambda: self._cam_status_lbl.config(
                text="⚠ Cam 0 not found — try another index", fg=C["orange"]))
        threading.Thread(target=probe, daemon=True).start()

    def _test_camera(self):
        """Open a preview test without running a full scan."""
        if self._running:
            return
        idx = self._get_cam_index()
        self._cam_status_lbl.config(text="Testing...", fg=C["accent"])
        self._btn_test_cam.config(state="disabled")
        def do_test():
            import cv2 as _cv2
            indices = list(range(5)) if idx == -1 else [idx]
            backends = [_cv2.CAP_DSHOW, _cv2.CAP_ANY]
            for i in indices:
                for b in backends:
                    cap = _cv2.VideoCapture(i, b)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        cap.release()
                        if ret and frame is not None:
                            bname = 'DSHOW' if b == _cv2.CAP_DSHOW else 'ANY'
                            self.after(0, lambda i=i, bname=bname: (
                                self._cam_status_lbl.config(
                                    text=f"✓ Camera {i} OK ({bname})", fg=C["green"]),
                                self._btn_test_cam.config(state="normal"),
                            ))
                            return
            self.after(0, lambda: (
                self._cam_status_lbl.config(
                    text="✗ No working camera found", fg=C["red"]),
                self._btn_test_cam.config(state="normal"),
                messagebox.showerror(
                    "Camera Error",
                    "No working camera found.\n\n"
                    "Tips:\n"
                    "• Make sure your laptop webcam is enabled (Fn + camera key)\n"
                    "• Close Teams / Zoom / other apps using the camera\n"
                    "• Try selecting a different Camera index above\n"
                    "• Check Device Manager → Cameras"
                ),
            ))
        threading.Thread(target=do_test, daemon=True).start()

    def _run(self):
        idx = self._get_cam_index()
        if idx == -1:
            idx = 0   # Scanner._open_camera already tries all indices
        self._scanner.scan_webcam(camera_index=idx)

    def _stop(self):
        self._scanner.stop()
        self._running = False
        self._btn_start.config(state="normal")
        self._btn_stop.config(state="disabled")
        self._status_lbl.config(text="Stopped", fg=C["gray"])
        self._scan_status.config(text="")

    def _reset(self):
        self._stop()
        self._reset_scan_dots()
        self._verdict.set_scanning("AWAITING SCAN")
        self._status_lbl.config(text="Ready.")

    def _reset_scan_dots(self):
        for dot in self._scan_dots:
            dot.config(fg=C["bg3"])

    def _on_frame(self, frame: np.ndarray, fr):
        try:
            self._frame_q.put_nowait(frame)
        except queue.Full:
            pass

    def _on_progress(self, scan_num, total_scans, label, frame):
        def update():
            self._scan_status.config(text=label, fg=C["accent"])
            self._verdict.set_scanning(label)
            for i, dot in enumerate(self._scan_dots):
                if i < scan_num:
                    dot.config(fg=C["accent"])
                else:
                    dot.config(fg=C["bg3"])
        self.after(0, update)

    def _poll_frames(self):
        try:
            frame = self._frame_q.get_nowait()
            cw = self._canvas.winfo_width() or 800
            ch = self._canvas.winfo_height() or 520
            if cw > 10 and ch > 10:
                ph = frame_to_photoimage(frame, cw, ch)
                self._preview_img = ph
                self._canvas.delete("all")
                self._canvas.create_image(0, 0, anchor="nw", image=ph)
        except queue.Empty:
            pass
        self.after(33, self._poll_frames)

    def _on_complete(self, sr: ScanResult):
        self._running = False
        def update():
            self._btn_start.config(state="normal")
            self._btn_stop.config(state="disabled")
            self._scan_status.config(text="COMPLETE", fg=C["green"])
            for dot in self._scan_dots:
                dot.config(fg=C["accent"])

            if sr.success and sr.fusion:
                # ── Webcam: simple CRIMINAL / SUSPECT / INNOCENT only ────────
                simple_verdict = get_simple_image_verdict(sr.fusion.fusion_conf)
                verdict_color  = VERDICT_COLORS.get(simple_verdict, C["gray"])
                icon           = verdict_icon(simple_verdict)

                # Update verdict panel with simple labels
                self._verdict._title.config(text=simple_verdict, fg=verdict_color)
                self._verdict._icon.config(text=icon, fg=verdict_color)
                self._verdict._score.config(text=f"{sr.fusion.fusion_conf:.0%}", fg=verdict_color)
                self._verdict._suspect.config(text="", fg=C["gray"])
                self._verdict._crime.config(text="", fg=C["gray"])
                self._verdict._bar_face.set_value(sr.fusion.face_conf)
                self._verdict._bar_gait.set_value(sr.fusion.gait_conf)
                self._verdict._bar_behav.set_value(sr.fusion.behavior_conf)
                self._verdict._bar_fused.set_value(sr.fusion.fusion_conf)
                self._verdict._log.config(state="normal")
                self._verdict._log.delete("1.0", "end")
                for line in (sr.fusion.reasoning or []):
                    self._verdict._log.insert("end", f"• {line}\n")
                self._verdict._log.config(state="disabled")

                for dot in self._scan_dots:
                    dot.config(fg=verdict_color)

                self._status_lbl.config(
                    text=f"{icon} {simple_verdict} — {sr.fusion.fusion_conf:.0%}",
                    fg=verdict_color
                )
                self.app.activity_log.log(
                    f"[WEBCAM] {simple_verdict} {sr.fusion.fusion_conf:.0%} "
                    f"Face:{sr.fusion.face_conf:.0%} Gait:{sr.fusion.gait_conf:.0%} "
                    f"Behavior:{sr.fusion.behavior_conf:.0%}",
                    tag=simple_verdict.lower()
                )

                # ── Ask user whether to add to DB ────────────────────────────────
                face_fv = sr.face.feature_vector if sr.face else []
                self.after(600, lambda: AddToDatabaseDialog(
                    self.winfo_toplevel(), simple_verdict,
                    sr.fusion.fusion_conf, face_fv,
                    on_added=self.app._tab_db.refresh
                ))

            else:
                err_msg = sr.error or "Unknown error"
                self._status_lbl.config(text=f"Error: {err_msg}", fg=C["red"])
                if "camera" in err_msg.lower() or "no cam" in err_msg.lower():
                    messagebox.showerror(
                        "Camera Not Found",
                        f"{err_msg}\n\nTips:\n"
                        "• Enable your laptop webcam (check Fn keys or Device Manager)\n"
                        "• Close Teams / Zoom / Skype\n"
                        "• Select a different camera index in the dropdown above\n"
                        "• Click 'Test Cam' to verify which index works"
                    )
        self.after(0, update)



# ═════════════════════════════════════════════════════════════════════════════
#  Register Face Dialog  (links a detected face to an existing criminal)
# ═════════════════════════════════════════════════════════════════════════════

class RegisterFaceDialog(tk.Toplevel):
    """
    Modal dialog shown when the user clicks "Register Face" in the Image tab.
    Lets the officer select which criminal from the registry this face belongs to,
    then stores the real face embedding so future uploads match correctly.
    """

    def __init__(self, parent, face_features: list, on_registered=None):
        super().__init__(parent)
        self.withdraw()  # Hide until centered
        self.transient(parent)
        self._fv           = face_features
        self._on_registered = on_registered

        self.title("Register Face to Criminal")
        self.configure(bg=C["bg"])
        self.resizable(False, False)

        # ── Robust Centering ──
        self.update_idletasks()
        w, h = 540, 420
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()

        if pw <= 1: # Parent not ready or minimized
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
            x, y = (sw - w) // 2, (sh - h) // 2
        else:
            x, y = px + (pw - w) // 2, py + (ph - h) // 2

        self.geometry(f"{w}x{h}+{x}+{y}")
        self.deiconify()
        self.lift()
        self.focus_force()
        self.grab_set()

        self._build()

    def _build(self):
        # ── Main container to prevent 'tiny box' collapse ──
        main = tk.Frame(self, bg=C["bg"])
        main.pack(fill="both", expand=True)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(main, bg=C["bg3"])
        hdr.pack(fill="x")
        tk.Label(hdr, text="\U0001f517  Link Face to Criminal Record",
                 bg=C["bg3"], fg=C["accent"],
                 font=("Segoe UI", 15, "bold")).pack(padx=20, pady=14, anchor="w")

        # ── Instruction ───────────────────────────────────────────────────────
        tk.Label(main,
                 text="Select the criminal this face belongs to.\n"
                      "The extracted face embedding will be saved so future\n"
                      "uploads of the same person match correctly.",
                 bg=C["bg"], fg=C["gray"],
                 font=("Segoe UI", 9), justify="left").pack(padx=20, pady=(14, 8), anchor="w")

        # ── Criminal dropdown ─────────────────────────────────────────────────
        criminals = get_all_criminals()      # list of dicts
        if not criminals:
            tk.Label(main, text="No criminals in database.", bg=C["bg"],
                     fg=C["red"], font=("Segoe UI", 10)).pack(pady=20)
            tk.Button(main, text="Close", command=self.destroy,
                      bg=C["bg3"], fg=C["gray"], relief="flat",
                      padx=20, pady=6).pack()
            return

        # Build display strings and ID map
        self._id_map = {}
        display_list = []
        for cr in criminals:
            risk   = cr.get("risk_level", "")
            alias  = f" ({cr['alias']})" if cr.get("alias") else ""
            label  = f"{cr['name']}{alias}  [{risk}]  (ID {cr['id']})"
            display_list.append(label)
            self._id_map[label] = cr["id"]

        form = tk.Frame(main, bg=C["bg"])
        form.pack(fill="x", padx=20, pady=4)
        tk.Label(form, text="Criminal:", bg=C["bg"], fg=C["gray"],
                 font=("Segoe UI", 9), width=12, anchor="w").pack(side="left")

        self._sel_var = tk.StringVar(value=display_list[0])
        cb = ttk.Combobox(form, textvariable=self._sel_var,
                          values=display_list, state="readonly",
                          font=("Segoe UI", 10), width=42)
        cb.pack(side="left", fill="x", expand=True)

        # ── Feedback ──────────────────────────────────────────────────────────
        self._msg = tk.Label(main, text="", bg=C["bg"],
                             font=("Segoe UI", 9))
        self._msg.pack(pady=(12, 0))

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = tk.Frame(main, bg=C["bg"])
        btn_row.pack(pady=16)

        tk.Button(btn_row, text="\u2714  Register & Save",
                  command=self._register,
                  bg=C["red"], fg="white", font=("Segoe UI", 11, "bold"),
                  relief="flat", padx=20, pady=8, cursor="hand2",
                  activebackground=C["accent"]).pack(side="left", padx=10)

        tk.Button(btn_row, text="\u2716  Cancel",
                  command=self.destroy,
                  bg=C["bg3"], fg=C["gray"], font=("Segoe UI", 11),
                  relief="flat", padx=20, pady=8, cursor="hand2").pack(side="left", padx=10)

    def _register(self):
        selected = self._sel_var.get()
        cid = self._id_map.get(selected)
        if not cid:
            self._msg.config(text="Please select a criminal.", fg=C["orange"])
            return
        ok = register_criminal_face(cid, self._fv)
        if ok:
            # Parse name from displayed label for callback
            name = selected.split("  [")[0].split(" (")[0].strip()
            self._msg.config(
                text=f"\u2705 Face registered for: {name}",
                fg=C["green"]
            )
            if self._on_registered:
                self._on_registered(name)
            self.after(1400, self.destroy)
        else:
            self._msg.config(text="\u2717 Failed to save — see console.", fg=C["red"])


# ═════════════════════════════════════════════════════════════════════════════
#  Add-to-Database Dialog  (shown after Webcam scan)
# ═════════════════════════════════════════════════════════════════════════════

class AddToDatabaseDialog(tk.Toplevel):
    """
    Modal popup that asks the officer whether to register the scanned
    person in the criminal database.  Appears automatically after each
    webcam 3-scan cycle.
    """

    CRIME_TYPES = [
        "Armed Robbery", "Assault", "Burglary", "Cybercrime",
        "Drug Trafficking", "Fraud", "Homicide", "Identity Theft",
        "Kidnapping", "Money Laundering", "Murder", "Organized Crime",
        "Smuggling", "Terrorism", "Terror Financing", "Vehicle Theft", "Other",
    ]
    RISK_LEVELS = ["HIGH", "MEDIUM", "LOW"]
    STATUSES    = ["At Large", "Imprisoned", "Deceased", "Unknown"]

    def __init__(self, parent, verdict: str, fusion_score: float,
                 face_features: list, on_added=None):
        super().__init__(parent)
        self.withdraw()  # Hide until centered
        self.transient(parent)
        self._face_features = face_features
        self._on_added      = on_added

        self.title("Register Detected Person")
        self.configure(bg=C["bg"])
        self.resizable(False, False)

        # ── Robust Centering ──
        self.update_idletasks()
        w, h = 540, 560
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()

        if pw <= 1:
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
            x, y = (sw - w) // 2, (sh - h) // 2
        else:
            x, y = px + (pw - w) // 2, py + (ph - h) // 2

        self.geometry(f"{w}x{h}+{x}+{y}")
        self.deiconify()
        self.lift()
        self.focus_force()
        self.grab_set()

        self._build(verdict, fusion_score)

    def _build(self, verdict: str, score: float):
        # ── Main container to prevent 'tiny box' collapse ──
        main = tk.Frame(self, bg=C["bg"])
        main.pack(fill="both", expand=True)

        # ── Header ────────────────────────────────────────────────────────────
        color = VERDICT_COLORS.get(verdict, C["gray"])
        hdr = tk.Frame(main, bg=C["bg3"])
        hdr.pack(fill="x")

        tk.Label(hdr, text=f"{verdict_icon(verdict)}  {verdict}",
                 bg=C["bg3"], fg=color,
                 font=("Segoe UI", 20, "bold")).pack(side="left", padx=20, pady=14)
        tk.Label(hdr, text=f"{score:.0%} Confidence",
                 bg=C["bg3"], fg=C["gray"],
                 font=("Segoe UI", 11)).pack(side="left")

        # ── Prompt ────────────────────────────────────────────────────────────
        tk.Label(main,
                 text="Add this person to the criminal database?",
                 bg=C["bg"], fg=C["txt"],
                 font=("Segoe UI", 11, "bold")).pack(pady=(18, 4))
        tk.Label(main,
                 text="Fill in the details below, then click Register.",
                 bg=C["bg"], fg=C["gray"],
                 font=("Segoe UI", 9)).pack(pady=(0, 12))

        # ── Form ──────────────────────────────────────────────────────────────
        form = tk.Frame(main, bg=C["bg"])
        form.pack(fill="x", padx=30)

        def row(label, widget_factory):
            r = tk.Frame(form, bg=C["bg"])
            r.pack(fill="x", pady=4)
            tk.Label(r, text=label, bg=C["bg"], fg=C["gray"],
                     font=("Segoe UI", 9), width=14, anchor="w").pack(side="left")
            w = widget_factory(r)
            w.pack(side="left", fill="x", expand=True)
            return w

        entry_kw = dict(bg=C["bg3"], fg=C["txt"], insertbackground=C["txt"],
                        font=("Segoe UI", 10), relief="flat", bd=4)

        self._name_var  = tk.StringVar()
        self._alias_var = tk.StringVar()
        self._crime_var = tk.StringVar(value=self.CRIME_TYPES[0])
        self._risk_var  = tk.StringVar(value="HIGH")
        self._status_var= tk.StringVar(value="At Large")
        self._notes_var = tk.StringVar()

        row("Full Name *",  lambda p: tk.Entry(p, textvariable=self._name_var,  **entry_kw))
        row("Alias / Nick", lambda p: tk.Entry(p, textvariable=self._alias_var, **entry_kw))
        row("Crime Type",   lambda p: ttk.Combobox(p, textvariable=self._crime_var,
                                                    values=self.CRIME_TYPES,
                                                    state="readonly", font=("Segoe UI", 10)))
        row("Risk Level",   lambda p: ttk.Combobox(p, textvariable=self._risk_var,
                                                    values=self.RISK_LEVELS,
                                                    state="readonly", font=("Segoe UI", 10), width=12))
        row("Status",       lambda p: ttk.Combobox(p, textvariable=self._status_var,
                                                    values=self.STATUSES,
                                                    state="readonly", font=("Segoe UI", 10)))
        row("Notes",        lambda p: tk.Entry(p, textvariable=self._notes_var, **entry_kw))

        # ── Feedback label ────────────────────────────────────────────────────
        self._msg = tk.Label(main, text="", bg=C["bg"], fg=C["red"],
                             font=("Segoe UI", 9))
        self._msg.pack(pady=(8, 0))

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = tk.Frame(main, bg=C["bg"])
        btn_row.pack(pady=16)

        tk.Button(btn_row, text="✔  Register",
                  command=self._register,
                  bg=C["red"], fg="white", font=("Segoe UI", 11, "bold"),
                  relief="flat", padx=24, pady=8, cursor="hand2",
                  activebackground=C["accent"]).pack(side="left", padx=10)

        tk.Button(btn_row, text="✖  Skip",
                  command=self.destroy,
                  bg=C["bg3"], fg=C["gray"], font=("Segoe UI", 11),
                  relief="flat", padx=24, pady=8, cursor="hand2").pack(side="left", padx=10)

    def _register(self):
        name = self._name_var.get().strip()
        if not name:
            self._msg.config(text="⚠ Full Name is required.", fg=C["orange"])
            return
        cid = add_criminal_to_db(
            name        = name,
            alias       = self._alias_var.get(),
            crime_type  = self._crime_var.get(),
            risk_level  = self._risk_var.get(),
            status      = self._status_var.get(),
            notes       = self._notes_var.get(),
            face_features = self._face_features,
        )
        if cid > 0:
            self._msg.config(text=f"✓ Registered '{name}' (ID {cid})", fg=C["green"])
            if self._on_added:
                self._on_added()
            self.after(1200, self.destroy)
        else:
            self._msg.config(text="✗ DB error — see console.", fg=C["red"])


# ═════════════════════════════════════════════════════════════════════════════
#  Tab 4: Database Viewer
# ═════════════════════════════════════════════════════════════════════════════

class DatabaseTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=C["bg"])
        self.app = app
        self._build()
        self.refresh()

    def _build(self):
        # ── Top stats bar ─────────────────────────────────────────────────────
        stats_bar = tk.Frame(self, bg=C["bg3"])
        stats_bar.pack(fill="x", padx=15, pady=(12, 0))
        self._stat_labels = {}
        for key, label, color in [
            ("total_criminals", "SUSPECT REGISTRY", C["accent"]),
            ("total_scans",     "TOTAL SCANS",      C["white"]),
            ("criminal_hits",   "CRIMINAL HITS",    C["red"]),
            ("watchlist_hits",  "WATCH LIST",        C["orange"]),
            ("clear_hits",      "CLEARED",           C["green"]),
        ]:
            col = tk.Frame(stats_bar, bg=C["bg3"])
            col.pack(side="left", expand=True, fill="x", padx=12, pady=10)
            val_lbl = tk.Label(col, text="—", bg=C["bg3"], fg=color,
                               font=("Segoe UI", 24, "bold"))
            val_lbl.pack()
            tk.Label(col, text=label, bg=C["bg3"], fg=C["gray"],
                     font=("Segoe UI", 8)).pack()
            self._stat_labels[key] = val_lbl

        # ── Main split ────────────────────────────────────────────────────────
        split = tk.Frame(self, bg=C["bg"])
        split.pack(fill="both", expand=True, padx=15, pady=12)

        # Left: Criminal registry table
        left = tk.Frame(split, bg=C["bg2"])
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        hdr_row = tk.Frame(left, bg=C["bg2"])
        hdr_row.pack(fill="x", padx=10, pady=(10, 6))
        tk.Label(hdr_row, text="CRIMINAL REGISTRY", bg=C["bg2"], fg=C["accent"],
                 font=("Segoe UI", 11, "bold")).pack(side="left")
        tk.Button(hdr_row, text="↻ Refresh", command=self.refresh,
                  bg=C["bg3"], fg=C["txt"], font=("Segoe UI", 8), relief="flat",
                  padx=8, cursor="hand2").pack(side="right")

        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._filter_criminals())
        search_frame = tk.Frame(left, bg=C["bg2"])
        search_frame.pack(fill="x", padx=10, pady=(0, 6))
        tk.Label(search_frame, text="🔍", bg=C["bg2"], fg=C["gray"],
                 font=("Segoe UI", 11)).pack(side="left")
        tk.Entry(search_frame, textvariable=self._search_var, bg=C["bg3"],
                 fg=C["txt"], insertbackground=C["txt"], font=("Segoe UI", 9),
                 relief="flat", bd=0).pack(side="left", fill="x", expand=True, padx=(4, 0))

        cols_c = ("Name", "Alias", "Crime", "Risk", "Status")
        self._tree_c = ttk.Treeview(left, columns=cols_c, show="headings", height=14)
        for col, w in zip(cols_c, [160, 100, 150, 80, 80]):
            self._tree_c.heading(col, text=col)
            self._tree_c.column(col, width=w, anchor="w")
        sb = ttk.Scrollbar(left, command=self._tree_c.yview)
        self._tree_c.config(yscrollcommand=sb.set)
        self._tree_c.pack(fill="both", expand=True, padx=10)
        sb.pack(side="right", fill="y")

        # Right: Detection log
        right = tk.Frame(split, bg=C["bg2"], width=400)
        right.pack(side="right", fill="both")
        right.pack_propagate(False)

        hdr2 = tk.Frame(right, bg=C["bg2"])
        hdr2.pack(fill="x", padx=10, pady=(10, 6))
        tk.Label(hdr2, text="DETECTION LOG", bg=C["bg2"], fg=C["accent"],
                 font=("Segoe UI", 11, "bold")).pack(side="left")

        cols_d = ("Time", "Verdict", "Mode", "Conf%")
        self._tree_d = ttk.Treeview(right, columns=cols_d, show="headings", height=14)
        for col, w in zip(cols_d, [90, 90, 70, 60]):
            self._tree_d.heading(col, text=col)
            self._tree_d.column(col, width=w, anchor="center")
        sb2 = ttk.Scrollbar(right, command=self._tree_d.yview)
        self._tree_d.config(yscrollcommand=sb2.set)
        self._tree_d.pack(fill="both", expand=True, padx=10)
        sb2.pack(side="right", fill="y")

        # Row tag colours
        self._tree_d.tag_configure("CRIMINAL",   foreground=C["red"])
        self._tree_d.tag_configure("WATCH LIST", foreground=C["orange"])
        self._tree_d.tag_configure("CLEAR",      foreground=C["green"])
        self._tree_c.tag_configure("HIGH",   foreground=C["red"])
        self._tree_c.tag_configure("MEDIUM", foreground=C["orange"])
        self._tree_c.tag_configure("LOW",    foreground=C["green"])

    def refresh(self):
        """Reload data from database."""
        try:
            # Stats
            stats = get_stats()
            for key, lbl in self._stat_labels.items():
                lbl.config(text=str(stats.get(key, 0)))

            # Criminals
            self._all_criminals = get_all_criminals()
            self._populate_criminals(self._all_criminals)

            # Detection log
            for item in self._tree_d.get_children():
                self._tree_d.delete(item)
            detections = get_recent_detections(limit=50)
            for d in detections:
                ts  = d.get("timestamp", "")[:16]
                tag = d.get("verdict", "CLEAR")
                self._tree_d.insert("", "end", values=(
                    ts,
                    f"{verdict_icon(tag)} {tag}",
                    d.get("mode", "").upper(),
                    f"{d.get('fusion_conf', 0)*100:.0f}%",
                ), tags=(tag,))

        except Exception as e:
            print(f"DB refresh error: {e}")

    def _populate_criminals(self, criminals: list):
        for item in self._tree_c.get_children():
            self._tree_c.delete(item)
        for cr in criminals:
            risk = cr.get("risk_level", "MEDIUM")
            self._tree_c.insert("", "end", values=(
                cr.get("name", ""),
                cr.get("alias", ""),
                cr.get("crime_type", ""),
                cr.get("risk_level", ""),
                cr.get("status", ""),
            ), tags=(risk,))

    def _filter_criminals(self):
        q = self._search_var.get().lower()
        filtered = [
            c for c in self._all_criminals
            if q in c.get("name", "").lower()
            or q in c.get("alias", "").lower()
            or q in c.get("crime_type", "").lower()
        ]
        self._populate_criminals(filtered)


# ═════════════════════════════════════════════════════════════════════════════
#  Main Application Window
# ═════════════════════════════════════════════════════════════════════════════

class CISv2App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.configure(bg=C["bg"])
        self.root.minsize(1200, 700)

        # Apply dark ttk theme
        self._apply_ttk_style()

        # ── Init database ────────────────────────────────────────────────────
        try:
            init_db()
        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to initialize DB:\n{e}")

        # ── Build layout ─────────────────────────────────────────────────────
        self._build_header()
        self._build_main()
        self._build_footer()

    def _apply_ttk_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TNotebook",          background=C["bg"], borderwidth=0)
        style.configure("TNotebook.Tab",      background=C["bg3"], foreground=C["gray"],
                        padding=[18, 8], font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", C["bg2"])],
                  foreground=[("selected", C["accent"])])
        style.configure("Treeview",           background=C["bg2"], foreground=C["txt"],
                        fieldbackground=C["bg2"], rowheight=24, font=("Segoe UI", 9))
        style.configure("Treeview.Heading",   background=C["bg3"], foreground=C["accent"],
                        font=("Segoe UI", 9, "bold"))
        style.configure("TProgressbar",       troughcolor=C["bg3"], background=C["accent"])
        style.configure("TScrollbar",         background=C["bg3"], troughcolor=C["bg"])
        style.map("Treeview", background=[("selected", C["bg4"])])

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=C["bg3"], height=60)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        # Left: Logo
        logo = tk.Frame(hdr, bg=C["bg3"])
        logo.pack(side="left", padx=20)
        tk.Label(logo, text="⬡ CIS", bg=C["bg3"], fg=C["accent"],
                 font=("Segoe UI", 22, "bold")).pack(side="left")
        tk.Label(logo, text="v2", bg=C["bg3"], fg=C["purple"],
                 font=("Segoe UI", 12, "bold")).pack(side="left", padx=(2, 8), pady=(8, 0))
        tk.Label(logo, text="Criminal Identification System  •  BioFuse", bg=C["bg3"], fg=C["gray"],
                 font=("Segoe UI", 9)).pack(side="left")

        # Right: Live clock + status
        right = tk.Frame(hdr, bg=C["bg3"])
        right.pack(side="right", padx=20)
        self._clock_lbl = tk.Label(right, text="", bg=C["bg3"], fg=C["accent"],
                                    font=("Consolas", 11))
        self._clock_lbl.pack(side="right")
        self._db_status = tk.Label(right, text="● DB ONLINE", bg=C["bg3"], fg=C["green"],
                                    font=("Segoe UI", 9))
        self._db_status.pack(side="right", padx=20)
        self._update_clock()

    def _update_clock(self):
        now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        self._clock_lbl.config(text=now)
        self.root.after(1000, self._update_clock)

    def _build_main(self):
        # Main body: tabs on left content, narrow activity log on right
        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=0, pady=0)

        # Notebook
        self._nb = ttk.Notebook(body)
        self._nb.pack(side="left", fill="both", expand=True)

        # Activity Log sidebar
        self.activity_log = LogPanel(body, title="ACTIVITY LOG")
        self.activity_log.pack(side="right", fill="y", padx=(0, 0), pady=0)
        self.activity_log.config(width=260)
        self.activity_log.pack_propagate(False)

        # Create tabs
        self._tab_image   = ImageTab(self._nb, self)
        self._tab_video   = VideoTab(self._nb, self)
        self._tab_webcam  = WebcamTab(self._nb, self)
        self._tab_db      = DatabaseTab(self._nb, self)

        self._nb.add(self._tab_image,  text="  📷 Image Upload  ")
        self._nb.add(self._tab_video,  text="  📹 Video Upload  ")
        self._nb.add(self._tab_webcam, text="  🎥 Live Webcam  ")
        self._nb.add(self._tab_db,     text="  🗄️ Database  ")

        # Refresh DB tab when selected
        self._nb.bind("<<NotebookTabChanged>>", self._on_tab_change)

        self.activity_log.log("CIS v2 initialized — Database online", "info")
        self.activity_log.log("Thresholds: Criminal>62% | Watch>45%", "info")

    def _on_tab_change(self, event):
        selected = self._nb.select()
        tab_text = self._nb.tab(selected, "text")
        if "Database" in tab_text:
            self._tab_db.refresh()

    def _build_footer(self):
        footer = tk.Frame(self.root, bg=C["bg3"], height=28)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        tk.Label(
            footer,
            text=("IGNISIA  •  Team BioFuse  •  Criminal Identification System v2  "
                  "•  Face 45% | Gait 30% | Behavior 25%  •  "
                  "CRIMINAL >62%  |  WATCH LIST >45%  |  CLEAR <45%"),
            bg=C["bg3"], fg=C["gray"], font=("Segoe UI", 8)
        ).pack(side="left", padx=16, pady=4)
        tk.Label(footer, text="⚠ FOR LAW ENFORCEMENT USE ONLY", bg=C["bg3"], fg=C["red"],
                 font=("Segoe UI", 8, "bold")).pack(side="right", padx=16)


# ═════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()

    # Try to set icon (ignore if fails)
    try:
        root.iconbitmap(default="")
    except Exception:
        pass

    app = CISv2App(root)

    # Center on screen
    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    ww, wh = 1500, 900
    x = (sw - ww) // 2
    y = (sh - wh) // 2
    root.geometry(f"{ww}x{wh}+{x}+{y}")

    root.mainloop()


if __name__ == "__main__":
    main()
