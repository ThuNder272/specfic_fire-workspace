#!/usr/bin/env python
# coding=utf-8

"""Middle-process latency visualization for the vision -> EC -> motor chain.

Renders two small rolling OpenCV panels (yaw + pitch) onto the display frame:
  * blue  line = TARGET  (vision-sent command yaw/pitch)
  * green line = REAL    (motor feedback yaw/pitch from the EC board)

The gap between the two curves is the visible tracking lag/stutter the sentry
auto-aim is being debugged for.

Only two of the three conceptual values from the design are measurable today:
  1. TARGET (vision-sent)  -> available (the command we transmit)
  2. EC-received           -> NOT measurable until the EC board echoes the
                              vision send-timestamp back; a hook is left below
  3. REAL (motor feedback) -> available (EC feedback packet)

An optional CSV log mirrors the columns of the C++ AutoAim_work_v2 version so
both pipelines can be compared with the same offline tooling / VOFA+.
"""

import os
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np


# BGR colors.
_TARGET_COLOR = (255, 0, 0)     # blue  -> vision-sent target
_REAL_COLOR = (0, 255, 0)       # green -> motor real feedback
_TEXT_COLOR = (255, 255, 255)
_AXIS_COLOR = (90, 90, 90)
_PANEL_COLOR = (20, 20, 20)
_PANEL_ALPHA = 0.55


class LatencyPlot:
    """Rolling TARGET-vs-REAL yaw/pitch visualization (OpenCV overlay + CSV)."""

    def __init__(
        self,
        enabled: bool = True,
        window_seconds: float = 5.0,
        csv_path: Optional[str] = None,
        max_samples: int = 4000,
    ):
        self.enabled = bool(enabled)
        self.window_seconds = max(0.1, float(window_seconds))
        self.max_samples = max(10, int(max_samples))
        self._samples = deque()  # each: dict with t, target/real yaw/pitch, flags
        self._latest_latency_ms: float = -1.0

        self._csv_file = None
        self._csv_pending = 0
        self._csv_flush_interval = 30
        if csv_path:
            self._open_csv(csv_path)

    # ------------------------------------------------------------------ CSV ---
    def _open_csv(self, csv_path: str):
        try:
            parent = os.path.dirname(os.path.abspath(csv_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._csv_file = open(csv_path, "w", buffering=1)
            self._csv_file.write(
                "t_s,has_target,feedback_valid,target_yaw_deg,target_pitch_deg,"
                "real_yaw_deg,real_pitch_deg,yaw_err_deg,pitch_err_deg\n"
            )
            print(f"[latency_viz] writing latency log to {csv_path}")
        except Exception as exc:  # noqa: BLE001 - best effort logging only
            print(f"[latency_viz] failed to open csv {csv_path}: {exc}")
            self._csv_file = None

    def _write_csv_row(self, sample: dict, yaw_err: float, pitch_err: float):
        if self._csv_file is None:
            return
        try:
            self._csv_file.write(
                f"{sample['t']:.6f},{1 if sample['has_target'] else 0},"
                f"{1 if sample['feedback_valid'] else 0},"
                f"{sample['target_yaw']:.4f},{sample['target_pitch']:.4f},"
                f"{sample['real_yaw']:.4f},{sample['real_pitch']:.4f},"
                f"{yaw_err:.4f},{pitch_err:.4f}\n"
            )
            self._csv_pending += 1
            if self._csv_pending >= self._csv_flush_interval:
                self._csv_file.flush()
                self._csv_pending = 0
        except Exception:  # noqa: BLE001
            pass

    def close(self):
        if self._csv_file is not None:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception:  # noqa: BLE001
                pass
            self._csv_file = None

    # ----------------------------------------------------------------- data ---
    def active(self) -> bool:
        return self.enabled or self._csv_file is not None

    def push(
        self,
        t_seconds: float,
        has_target: bool,
        feedback_valid: bool,
        target_yaw: float,
        target_pitch: float,
        real_yaw: float,
        real_pitch: float,
        transport_latency_ms: float = -1.0,
    ):
        if not self.active():
            return

        if transport_latency_ms >= 0.0:
            self._latest_latency_ms = transport_latency_ms

        sample = {
            "t": float(t_seconds),
            "has_target": bool(has_target),
            "feedback_valid": bool(feedback_valid),
            "target_yaw": float(target_yaw),
            "target_pitch": float(target_pitch),
            "real_yaw": float(real_yaw),
            "real_pitch": float(real_pitch),
        }

        if self.enabled:
            self._samples.append(sample)
            self._trim()

        if self._csv_file is not None:
            self._write_csv_row(
                sample, target_yaw - real_yaw, target_pitch - real_pitch
            )

    def _trim(self):
        if not self._samples:
            return
        newest = self._samples[-1]["t"]
        while self._samples and (newest - self._samples[0]["t"]) > self.window_seconds:
            self._samples.popleft()
        while len(self._samples) > self.max_samples:
            self._samples.popleft()

    # ----------------------------------------------------------------- draw ---
    def draw(self, frame: np.ndarray):
        if not self.enabled or frame is None or len(self._samples) < 2:
            return

        h, w = frame.shape[:2]
        margin = 10
        panel_w = int(min(300, max(160, w / 3)))
        panel_h = 96
        x = w - panel_w - margin
        self._draw_panel(
            frame, x, margin, panel_w, panel_h,
            "YAW deg (blue=tx  green=real)", "target_yaw", "real_yaw",
        )
        self._draw_panel(
            frame, x, margin + panel_h + margin, panel_w, panel_h,
            "PITCH deg (blue=tx  green=real)", "target_pitch", "real_pitch",
        )

        # Latency readout below the two panels.
        lat_y = margin + panel_h + margin + panel_h + 4
        if self._latest_latency_ms >= 0.0:
            lat_text = f"latency={self._latest_latency_ms:.1f}ms"
        else:
            lat_text = "latency=n/a (needs EC fw update)"
        cv2.putText(
            frame, lat_text, (x, lat_y + 12), cv2.FONT_HERSHEY_SIMPLEX,
            0.40, _TEXT_COLOR, 1, cv2.LINE_AA,
        )

    def _draw_panel(self, frame, px, py, pw, ph, title, target_key, real_key):
        h, w = frame.shape[:2]
        px = max(0, min(px, w - 1))
        py = max(0, min(py, h - 1))
        pw = min(pw, w - px)
        ph = min(ph, h - py)
        if pw < 40 or ph < 30:
            return

        # Semi-transparent dark background.
        roi = frame[py:py + ph, px:px + pw]
        bg = np.empty_like(roi)
        bg[:] = _PANEL_COLOR
        cv2.addWeighted(bg, _PANEL_ALPHA, roi, 1.0 - _PANEL_ALPHA, 0.0, roi)
        cv2.rectangle(frame, (px, py), (px + pw - 1, py + ph - 1), _AXIS_COLOR, 1)

        pad_left, pad_right, pad_top, pad_bottom = 6, 6, 16, 16
        x0 = px + pad_left
        x1 = px + pw - pad_right
        y0 = py + pad_top
        y1 = py + ph - pad_bottom
        if x1 - x0 < 10 or y1 - y0 < 10:
            return

        cv2.putText(
            frame, title, (px + 4, py + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
            _TEXT_COLOR, 1, cv2.LINE_AA,
        )

        t_newest = self._samples[-1]["t"]
        t_oldest = self._samples[0]["t"]
        t_span = max(1e-6, t_newest - t_oldest)

        v_min = float("inf")
        v_max = float("-inf")
        for s in self._samples:
            tv = s[target_key]
            v_min = min(v_min, tv)
            v_max = max(v_max, tv)
            if s["feedback_valid"]:
                rv = s[real_key]
                v_min = min(v_min, rv)
                v_max = max(v_max, rv)
        if v_min > v_max:
            return
        if v_max - v_min < 1e-3:
            v_min -= 0.5
            v_max += 0.5
        v_span = v_max - v_min

        def map_x(t):
            return int(round(x0 + (t - t_oldest) / t_span * (x1 - x0)))

        def map_y(v):
            return int(round(y1 - (v - v_min) / v_span * (y1 - y0)))

        # Target curve (always valid).
        tgt_pts = np.array(
            [[map_x(s["t"]), map_y(s[target_key])] for s in self._samples],
            dtype=np.int32,
        )
        if len(tgt_pts) >= 2:
            cv2.polylines(frame, [tgt_pts], False, _TARGET_COLOR, 1, cv2.LINE_AA)

        # Real curve (break across invalid segments).
        segment = []
        for s in self._samples:
            if s["feedback_valid"]:
                segment.append([map_x(s["t"]), map_y(s[real_key])])
            elif segment:
                if len(segment) >= 2:
                    cv2.polylines(
                        frame, [np.array(segment, dtype=np.int32)], False,
                        _REAL_COLOR, 1, cv2.LINE_AA,
                    )
                segment = []
        if len(segment) >= 2:
            cv2.polylines(
                frame, [np.array(segment, dtype=np.int32)], False,
                _REAL_COLOR, 1, cv2.LINE_AA,
            )

        # Numeric readout.
        last = self._samples[-1]
        tgt = last[target_key]
        real = last[real_key]
        if last["feedback_valid"]:
            readout = f"tx={tgt:.2f} re={real:.2f} err={tgt - real:.2f}"
        else:
            readout = f"tx={tgt:.2f} re=n/a"
        cv2.putText(
            frame, readout, (px + 4, py + ph - 4), cv2.FONT_HERSHEY_SIMPLEX,
            0.34, _TEXT_COLOR, 1, cv2.LINE_AA,
        )
