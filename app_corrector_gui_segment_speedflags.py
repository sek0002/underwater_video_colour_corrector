import sys
import shlex
import runpy
from pathlib import Path
from typing import Optional , Tuple, List
from PySide6 import QtCore
from PySide6.QtCore import Qt, QPointF, QRect, QEvent, Signal, QSize
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QIcon
import numpy as np
from PySide6.QtWidgets import QSlider, QStyle, QStyleOptionSlider
from dataclasses import dataclass

try:
    import cv2  # Optional: for probing video duration
except Exception:
    cv2 = None

from PySide6.QtCore import Qt, QProcess, QTimer, QUrl
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QTextEdit, QLineEdit, QLabel, QFileDialog, QMessageBox,
    QDoubleSpinBox, QSpinBox, QGroupBox, QCheckBox, QProgressBar, QSlider,
    QSizePolicy, QSplitter, QComboBox, QToolBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QScrollArea, QToolButton, QButtonGroup
)

# from __future__ import annotations
    
# Import multimedia lazily inside init to avoid hard failures on some systems.
class ClickSeekSlider(QSlider):
    clickedValue = Signal(int)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)

            # Where is the handle?
            handle_rect = self.style().subControlRect(
                QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self
            )

            # If clicking on the handle, allow normal dragging
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            if handle_rect.contains(pos):
                return super().mousePressEvent(event)

            # Otherwise, click on groove: jump-to-click and emit
            if self.orientation() == Qt.Horizontal:
                x = pos.x()
                val = QStyle.sliderValueFromPosition(
                    self.minimum(), self.maximum(), x, self.width(), opt.upsideDown
                )
            else:
                y = pos.y()
                val = QStyle.sliderValueFromPosition(
                    self.minimum(), self.maximum(), self.height() - y, self.height(), opt.upsideDown
                )

            self.setValue(val)
            self.clickedValue.emit(val)
            event.accept()
            return

        return super().mousePressEvent(event)


class _ResizeHook(QtCore.QObject):
    """Event filter to call a function whenever a watched widget is resized."""

    def __init__(self, on_resize):
        super().__init__()
        self._on_resize = on_resize

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Resize:
            try:
                self._on_resize()
            except Exception:
                pass
        return False


def resource_path(relative_name: str) -> str:
    base = getattr(sys, "_MEIPASS", str(Path(__file__).resolve().parent))
    return str(Path(base) / relative_name)


# -----------------------------
# Worker + result definitions
# -----------------------------

@dataclass(frozen=True)
class ScopeGeometry:
    # Each is an (N, 2) int32 array of xy points in widget coordinates
    parade_r: np.ndarray
    parade_g: np.ndarray
    parade_b: np.ndarray
    waveform_y: np.ndarray


def _apply_compact_margins(layout):
    """Recursively reduce margins/spacings for a layout tree."""
    if layout is None:
        return
    try:
        layout.setContentsMargins(2, 2, 2, 2)
    except Exception:
        pass
    try:
        layout.setSpacing(4)
    except Exception:
        pass
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item is None:
            continue
        child_layout = item.layout()
        if child_layout is not None:
            _apply_compact_margins(child_layout)

class _ScopeWorkerSignals(QtCore.QObject):
    done = Signal(object)  # emits ScopeGeometry


class _ScopeWorker(QtCore.QRunnable):
    """
    Compute geometry from a BGR uint8 frame and the target widget rects.
    Returns only primitive/numpy types (no QObjects) to avoid cross-thread Qt issues.
    """
    def __init__(
        self,
        frame_bgr: np.ndarray,
        w: int,
        h: int,
        title: str,
        mode: str,
    ):
        super().__init__()
        self.signals = _ScopeWorkerSignals()
        self._frame = frame_bgr
        self._w = int(w)
        self._h = int(h)
        self._title = title
        self._mode = mode

    @staticmethod
    def _compute_geometry(frame_bgr: np.ndarray, w: int, h: int, mode: str) -> ScopeGeometry:
        # Safety
        if frame_bgr is None or w <= 10 or h <= 10:
            z = np.zeros((0, 2), dtype=np.int32)
            return ScopeGeometry(z, z, z, z)

        # Clamp compute cost: downscale aggressively for scope computation
        hh, ww = frame_bgr.shape[:2]
        max_dim = 220  # keep small; scopes are qualitative
        scale = max(hh / max_dim, ww / max_dim, 1.0)
        if scale > 1.0:
            nh = max(2, int(hh / scale))
            nw = max(2, int(ww / scale))
            # fast nearest-neighbor downscale
            frame = frame_bgr[:: max(1, hh // nh), :: max(1, ww // nw)].copy()
        else:
            frame = frame_bgr

        hh, ww = frame.shape[:2]

        # Layout similar to your existing widget:
        title_h = 18
        pad = 6
        content_top = title_h + pad
        content_h = max(10, h - content_top - pad)

        # Split vertically: top = parade, bottom = waveform
        parade_h = int(content_h * 0.58)
        wave_h = content_h - parade_h - pad

        parade_rect = (pad, content_top, w - 2 * pad, parade_h)
        wave_rect = (pad, content_top + parade_h + pad, w - 2 * pad, wave_h)

        pr_x, pr_y, pr_w, pr_h = parade_rect
        wr_x, wr_y, wr_w, wr_h = wave_rect

        if pr_w <= 10 or pr_h <= 10 or wr_w <= 10 or wr_h <= 10:
            z = np.zeros((0, 2), dtype=np.int32)
            return ScopeGeometry(z, z, z, z)

        third = pr_w // 3

        # Sample fewer pixels: choose a sparse grid
        # - columns: up to 160
        # - rows: up to 120
        cols = np.linspace(0, ww - 1, num=min(160, ww), dtype=np.int32)
        rows = np.linspace(0, hh - 1, num=min(120, hh), dtype=np.int32)

        # Pull samples
        # frame is BGR
        b = frame[rows[:, None], cols[None, :], 0].astype(np.uint8)
        g = frame[rows[:, None], cols[None, :], 1].astype(np.uint8)
        r = frame[rows[:, None], cols[None, :], 2].astype(np.uint8)

        # For each channel, x is mapped to its third, y mapped by value
        # Reduce points: sub-sample rows to ~70 per column max
        row_step = max(1, len(rows) // 70)

        # Build coordinate arrays efficiently
        # x positions for each column
        x_norm = np.linspace(0, third - 1, num=len(cols), dtype=np.float32)
        x_third = x_norm.astype(np.int32)

        # y mapping helper
        def y_map(vals: np.ndarray, top: int, height: int) -> np.ndarray:
            # vals uint8 -> y pixels (bottom-up)
            return (top + height - 1 - ((vals.astype(np.float32) / 255.0) * (height - 1))).astype(np.int32)

        # Parade points
        # Shape: (nrows, ncols). We want list of points (x,y)
        rr = r[::row_step, :]
        gg = g[::row_step, :]
        bb = b[::row_step, :]

        # Broadcast x to match rr
        x0 = (pr_x + x_third)[None, :]  # (1, ncols)
        x1 = (pr_x + third + x_third)[None, :]
        x2 = (pr_x + 2 * third + x_third)[None, :]

        y_r = y_map(rr, pr_y, pr_h)
        y_g = y_map(gg, pr_y, pr_h)
        y_b = y_map(bb, pr_y, pr_h)

        # Flatten
        parade_r = np.stack([np.broadcast_to(x0, y_r.shape), y_r], axis=-1).reshape(-1, 2)
        parade_g = np.stack([np.broadcast_to(x1, y_g.shape), y_g], axis=-1).reshape(-1, 2)
        parade_b = np.stack([np.broadcast_to(x2, y_b.shape), y_b], axis=-1).reshape(-1, 2)

        # Waveform (luma)
        # ITU-R BT.709 coefficients close to what you used
        y = (0.2126 * r.astype(np.float32) + 0.7152 * g.astype(np.float32) + 0.0722 * b.astype(np.float32)).astype(np.float32)
        # Reduce points
        y_small = y[::row_step, :]  # (nrows2, ncols)
        xw = np.linspace(0, wr_w - 1, num=len(cols), dtype=np.int32)[None, :]
        xw = (wr_x + xw).astype(np.int32)
        yw = (wr_y + wr_h - 1 - ((y_small / 255.0) * (wr_h - 1))).astype(np.int32)
        waveform = np.stack([np.broadcast_to(xw, yw.shape), yw], axis=-1).reshape(-1, 2)

        # Final hygiene: ensure int32
        return ScopeGeometry(
            parade_r.astype(np.int32, copy=False),
            parade_g.astype(np.int32, copy=False),
            parade_b.astype(np.int32, copy=False),
            waveform.astype(np.int32, copy=False),
        )

    def run(self) -> None:
        try:
            geom = self._compute_geometry(self._frame, self._w, self._h, self._mode)
        except Exception:
            z = np.zeros((0, 2), dtype=np.int32)
            geom = ScopeGeometry(z, z, z, z)
        self.signals.done.emit(geom)


# -----------------------------
# Widget
# -----------------------------

class AsyncScopesWidget(QWidget):
    """
    Cached, async scopes widget.

    Public API:
      - set_frame_bgr(frame_bgr): schedule async recompute; latest-frame-wins; drop if busy
      - set_mode("parade"|"waveform"|"both"): (optional) default "both"
    """

    def __init__(self, title: str = "Scopes"):
        super().__init__()

        # --- Compact UI spacing ---
        compact_stylesheet = """
        QWidget { margin: 0px; padding: 0px; }
        QTabWidget::pane { margin: 0px; padding: 0px; border: 0px; }
        QTabBar::tab { margin: 0px; padding: 3px 8px; }
        QGroupBox { margin: 0px; padding: 4px; }
        QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0px 2px; }
        QLabel { margin: 0px; padding: 0px; }
        QPushButton { margin: 0px; padding: 3px 8px; }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { margin: 0px; padding: 2px 4px; }
        QSlider { margin: 0px; }
        QProgressBar { margin: 0px; padding: 0px; }
        QScrollArea { margin: 0px; padding: 0px; border: 0px; }
        """
        self.setStyleSheet(compact_stylesheet)
        self._title = title
        self._mode = "both"

        self.setMinimumSize(260, 180)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        self._thread_pool = QtCore.QThreadPool.globalInstance()
        self._busy = False
        self._pending_frame: Optional[np.ndarray] = None

        # Cached geometry (widget coordinates)
        z = np.zeros((0, 2), dtype=np.int32)
        self._geom = ScopeGeometry(z, z, z, z)

        # Static render cache
        self._static_grid_size = QSize(0, 0)
        self._static_grid = None  # QPixmap, built lazily in paint

        # Soft throttle: do not recompute more often than this (ms) when frames spam
        self._min_interval_ms = 80
        self._last_submit = QtCore.QElapsedTimer()
        self._last_submit.start()
        self._points_layer_size = QSize(0, 0)
        self._points_layer = None  # QPixmap with rendered points

    def set_mode(self, mode: str) -> None:
        mode = (mode or "both").lower().strip()
        if mode not in ("both", "parade", "waveform"):
            mode = "both"
        self._mode = mode
        self.update()

    def set_frame_bgr(self, frame_bgr: np.ndarray) -> None:
        """
        Request a scopes refresh from frame_bgr. Does not block.
        Drops intermediate frames if compute is still running.
        """
        if frame_bgr is None:
            return

        # Throttle submissions
        if self._last_submit.elapsed() < self._min_interval_ms:
            # keep the latest, do not submit now
            self._pending_frame = frame_bgr
            return

        if self._busy:
            self._pending_frame = frame_bgr
            return

        self._submit(frame_bgr)

    def _submit(self, frame_bgr: np.ndarray) -> None:
        self._busy = True
        self._pending_frame = None
        self._last_submit.restart()

        w, h = int(self.width()), int(self.height())
        worker = _ScopeWorker(frame_bgr, w, h, self._title, self._mode)
        worker.signals.done.connect(self._on_worker_done, QtCore.Qt.QueuedConnection)
        self._thread_pool.start(worker)

    @QtCore.Slot(object)
    def _on_worker_done(self, geom: ScopeGeometry) -> None:
        self._geom = geom
        self._busy = False

        # NEW: build points pixmap once per update (cheaper than per-paint QPoint allocation)
        self._rebuild_points_layer()

        self.update()

        if self._pending_frame is not None:
            pf = self._pending_frame
            self._pending_frame = None
            self._submit(pf)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._static_grid_size = QSize(0, 0)
        self._static_grid = None
        # NEW: invalidate points layer on resize
        self._points_layer_size = QSize(0, 0)
        self._points_layer = None

    def _rebuild_points_layer(self) -> None:
        """Render points into an offscreen pixmap so paintEvent is just a blit."""
        from PySide6.QtGui import QPixmap

        size = self.size()
        if size.width() <= 0 or size.height() <= 0:
            self._points_layer = None
            self._points_layer_size = QSize(0, 0)
            return

        pm = QPixmap(size)
        pm.fill(Qt.transparent)

        p = QPainter(pm)
        try:
            p.setRenderHint(QPainter.Antialiasing, False)

            if self._mode in ("both", "parade"):
                p.setPen(QPen(QColor(230, 80, 80), 1))
                self._draw_points_fast(p, self._geom.parade_r)

                p.setPen(QPen(QColor(80, 230, 120), 1))
                self._draw_points_fast(p, self._geom.parade_g)

                p.setPen(QPen(QColor(80, 120, 230), 1))
                self._draw_points_fast(p, self._geom.parade_b)

            if self._mode in ("both", "waveform"):
                p.setPen(QPen(QColor(220, 220, 220), 1))
                self._draw_points_fast(p, self._geom.waveform_y)

        finally:
            if p.isActive():
                p.end()

        self._points_layer = pm
        self._points_layer_size = size


    def _ensure_static_grid(self) -> None:
        # Lazy-create a static background grid cache
        size = self.size()
        if self._static_grid is not None and self._static_grid_size == size:
            return

        from PySide6.QtGui import QPixmap
        pm = QPixmap(size)
        pm.fill(QColor(18, 18, 18))

        title_h = 18
        pad = 6
        content_top = title_h + pad
        content_h = max(10, size.height() - content_top - pad)
        parade_h = int(content_h * 0.58)
        wave_h = content_h - parade_h - pad

        parade_rect = QRect(pad, content_top, size.width() - 2 * pad, parade_h)
        wave_rect = QRect(pad, content_top + parade_h + pad, size.width() - 2 * pad, wave_h)

        p = QPainter(pm)
        try:
            p.setRenderHint(QPainter.Antialiasing, False)

            # Title
            p.setPen(QColor(220, 220, 220))
            p.drawText(6, 14, self._title)

            # Frames
            p.setPen(QColor(70, 70, 70))
            p.drawRect(parade_rect)
            p.drawRect(wave_rect)

            # Grid lines (parade thirds)
            third = parade_rect.width() // 3
            p.setPen(QColor(40, 40, 40))
            p.drawLine(parade_rect.left() + third, parade_rect.top(),
                    parade_rect.left() + third, parade_rect.bottom())
            p.drawLine(parade_rect.left() + 2 * third, parade_rect.top(),
                    parade_rect.left() + 2 * third, parade_rect.bottom())

            # Horizontal grid lines
            for frac in (0.25, 0.5, 0.75):
                y = parade_rect.top() + int(parade_rect.height() * frac)
                p.drawLine(parade_rect.left(), y, parade_rect.right(), y)

                y2 = wave_rect.top() + int(wave_rect.height() * frac)
                p.drawLine(wave_rect.left(), y2, wave_rect.right(), y2)

        finally:
            if p.isActive():
                p.end()

        self._static_grid = pm
        self._static_grid_size = size

    def paintEvent(self, event) -> None:
        self._ensure_static_grid()

        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.Antialiasing, False)

            if self._static_grid is not None:
                p.drawPixmap(0, 0, self._static_grid)

            # Blit cached points pixmap (fast path)
            if self._points_layer is not None and self._points_layer_size == self.size():
                p.drawPixmap(0, 0, self._points_layer)

        finally:
            if p.isActive():
                p.end()

    def set_mode(self, mode: str) -> None:
        mode = (mode or "both").lower().strip()
        if mode not in ("both", "parade", "waveform"):
            mode = "both"
        if mode == self._mode:
            return
        self._mode = mode
        self._rebuild_points_layer()  # refresh cached layer for current geom
        self.update()

    @staticmethod
    def _draw_points_fast(p: QPainter, pts: np.ndarray) -> None:
        if pts is None or pts.size == 0:
            return
        from PySide6.QtCore import QPoint
        # Keep your existing bounded point counts; one list build per update instead of per repaint
        qpts = [QPoint(int(x), int(y)) for x, y in pts]
        p.drawPoints(qpts)



class BlueCorrectorSingleGUI(QWidget):
    def eventFilter(self, obj, event):
        try:
            if obj is getattr(self, "preview_time", None):
                if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                    self._prompt_seek_time_playback()
                    return True
        except Exception:
            pass
        return super().eventFilter(obj, event)
    def _prompt_seek_time_playback(self):
        if not getattr(self, "_preview_available", False):
            return

        from PySide6.QtWidgets import QInputDialog
        # Clear any previous processed playback error once we have a source.
        if not self.player_proc.source().isEmpty():
            self._clear_processed_playback_message()

        from PySide6.QtMultimedia import QMediaPlayer

        # Preserve play/pause state
        try:
            was_playing = (self.player_orig.playbackState() == QMediaPlayer.PlayingState)
        except Exception:
            was_playing = False

        # Default entry = current relative time
        try:
            cur_rel_ms = int(self.preview_slider.value())
        except Exception:
            cur_rel_ms = 0

        default_txt = self._fmt_ms(cur_rel_ms) if hasattr(self, "_fmt_ms") else "00:00"

        txt, ok = QInputDialog.getText(
            self,
            "Go to time",
            "Enter time (SS, MM:SS, HH:MM:SS)\n"
            "Use 'orig ' prefix for original timeline (e.g., orig 12:34):",
            text=default_txt
        )
        if not ok:
            return

        target_rel_ms = self._parse_seek_text_to_rel_ms(txt)
        if target_rel_ms is None:
            return

        # Clamp to window
        win = int(getattr(self, "_preview_window_ms", 0) or 0)
        if win > 0:
            target_rel_ms = max(0, min(int(target_rel_ms), win))
        else:
            target_rel_ms = max(0, int(target_rel_ms))

        # Seek both slider + players using your existing seek path
        try:
            self.preview_slider.setValue(int(target_rel_ms))
        except Exception:
            pass
        self.preview_seek(int(target_rel_ms))

        # If it was playing, continue
        if was_playing:
            try:
                self.player_orig.play()
                if not self.player_proc.source().isEmpty():
                    self.player_proc.play()
            except Exception:
                pass


    def _parse_seek_text_to_rel_ms(self, txt: str):
        if txt is None:
            return None
        s = str(txt).strip()
        if not s:
            return None

        # Absolute original timeline mode: "orig 12:34" / "original 1:02:03"
        lower = s.lower()
        is_orig = False
        if lower.startswith("orig "):
            is_orig = True
            s = s[5:].strip()
        elif lower.startswith("original "):
            is_orig = True
            s = s[9:].strip()

        sec = self._parse_hms_to_seconds(s)
        if sec is None:
            return None

        abs_ms = int(sec * 1000)

        if is_orig:
            # Convert absolute original -> relative window
            try:
                base = int(self._orig_base_ms())
            except Exception:
                base = 0
            rel = abs_ms - base
            return max(0, int(rel))

        # Otherwise treat as relative time inside the preview window
        return max(0, abs_ms)


    def _parse_hms_to_seconds(self, s: str):
        """
        Accepts:
        "75" -> 75s
        "01:15" -> 75s
        "1:02:03" -> 3723s
        """
        try:
            parts = [p.strip() for p in s.split(":")]
            if any(p == "" for p in parts):
                return None

            if len(parts) == 1:
                return float(parts[0])

            if len(parts) == 2:
                m = int(parts[0])
                sec = float(parts[1])
                return m * 60 + sec

            if len(parts) == 3:
                h = int(parts[0])
                m = int(parts[1])
                sec = float(parts[2])
                return h * 3600 + m * 60 + sec

            return None
        except Exception:
            return None

    def _on_show_scopes_frame_toggled(self, enabled: bool):
        """Frame-scrub scopes toggle (independent of playback scopes)."""
        try:
            self._on_show_scopes_frame_only(bool(enabled))
        except Exception:
            pass

    def _on_show_scopes_frame_only(self, enabled: bool) -> None:
        """Show/hide scopes in the Frame scrub tab only."""
        if hasattr(self, "scopes_frame_orig"):
            self.scopes_frame_orig.setVisible(enabled)
        if hasattr(self, "scopes_frame_proc"):
            self.scopes_frame_proc.setVisible(enabled)

    def _on_show_scopes_toggled(self, enabled: bool):
        """Playback scopes toggle (independent of frame-scrub scopes)."""
        enabled = bool(enabled)

        # Playback scopes widgets
        if hasattr(self, "scopes_playback_orig"):
            self.scopes_playback_orig.setVisible(enabled)
        if hasattr(self, "scopes_playback_proc"):
            self.scopes_playback_proc.setVisible(enabled)

        # When hiding playback scopes, stop any playback scopes timer to save CPU
        if not enabled:
            try:
                if getattr(self, "_play_scope_timer", None) is not None:
                    self._play_scope_timer.stop()
            except Exception:
                pass

    def _open_dir_for_path(self, path_text: str) -> str:
        try:
            p = Path(path_text).expanduser()
            if p.exists():
                return str(p if p.is_dir() else p.parent)
        except Exception:
            pass
        return str(Path.home())
    def _frame_render_current(self):
        """
        Slot for _frame_update_timer: render the currently selected preview frame
        (or re-render with current tuning). Keep it lightweight.
        """
        try:
            # If you have an explicit 'preview enabled' flag, gate here.
            # if not self.preview_enabled: return

            # Render one frame using existing logic.
            # This should NOT start playback, just update the still preview.
            if hasattr(self, "preview_render_once"):
                self.preview_render_once()
            elif hasattr(self, "render_preview_frame"):
                self.render_preview_frame()
            else:
                # Fallback: stop timer if nothing to call to avoid spinning.
                self._frame_update_timer.stop()
        except Exception as e:
            # Avoid crashing the GUI due to preview render errors.
            # Optionally: print(e) or log to your GUI console.
            self._frame_update_timer.stop()
        
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Colour Corrector Mod- MUUC")
        self.resize(1200, 820)
        self._seg_start_ms = 0
        self._seg_duration_ms = 0
        self._orig_duration_ms = 0
        self._proc_duration_ms = 0
        self._preview_window_ms = 0

        self.proc: Optional[QProcess] = None
        self.stage_pct = {"ANALYZE": 0.0, "PROCESS": 0.0}

        self._video_duration_sec = 0.0

        # Preview-related (created lazily)
        self._preview_initialized = False
        self._preview_available = False
        self._preview_block = False
        self._preview_duration_ms = 0

        self.player_orig = None
        self.player_proc = None
        self.audio_orig = None
        self.audio_proc = None
        self.video_orig = None
        self.video_proc = None
        self._sync_timer = None

        # Frame-scrub preview state (lazy init)
        self._frame_cap = None
        self._frame_req_timer = None
        self._frame_last_rel_ms = 0
        self._frame_loaded = False
        self._backend_mod = None
        self._frame_param_guard = False

        # Frame-scrub preview (cv2-based)
        self._frame_preview_initialized = False
        self._frame_cap = None
        self._frame_last_rel_ms = 0
        self._frame_update_timer = QTimer(self)
        self._frame_update_timer.setSingleShot(True)
        self._frame_update_timer.setInterval(80)
        self._frame_update_timer.timeout.connect(self._frame_render_current)
        self._frame_param_guard = False

        # Preset thumbnails (Frame scrub preview)
        self._preset_defs = None
        self._preset_btn_group = None
        self._preset_buttons = {}

        # Cache of (orig_bgr, processed_bgr) per preset, generated from the last rendered frame.
        # IMPORTANT: thumbnails are generated only when the base frame changes (frame scrub render),
        # never in response to clicking a preset or resizing the strip.
        self._preset_thumb_frames = {}
        self._preset_last_frame_bgr = None

        self._preset_active_name = None
        
        root = QVBoxLayout(self)

        # --- Top row: Inputs/Outputs and Tuning parameters side-by-side ---
        #paths_box = QGroupBox("Inputs/Outputs")
        paths_box = QGroupBox()
        paths_layout = QFormLayout(paths_box)

        self.script_path = QLineEdit(resource_path("app_backend_segment_speedflags.py"))
        btn_script = QPushButton("Browse")
        btn_script.clicked.connect(self.browse_script)
        script_row = QHBoxLayout()
        script_row.addWidget(self.script_path)
        script_row.addWidget(btn_script)
        # paths_layout.addRow("Script path:", script_row)

        # Input
        self.input_path = QLineEdit()
        btn_in = QPushButton("Browse")
        btn_in.clicked.connect(self.browse_input)
        in_row = QHBoxLayout()
        in_row.addWidget(self.input_path)
        in_row.addWidget(btn_in)
        paths_layout.addRow("Input file:", in_row)

        # Output behaviour
        self.same_folder = QCheckBox("Save next to input as <name>.corrected<ext>")
        self.same_folder.setChecked(True)
        self.same_folder.stateChanged.connect(self.on_same_folder_changed)
        paths_layout.addRow("", self.same_folder)

        self.output_path = QLineEdit()
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(self.browse_output)
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_path)
        out_row.addWidget(btn_out)
        paths_layout.addRow("Output file:", out_row)

        # Auto-reload preview when the input changes (file load / paste / browse).
        # This ensures preview always matches the currently selected file.
        # Reload on any text change (including Browse) but debounce to avoid thrashing.
        self._preview_reload_timer = QTimer(self)
        self._preview_reload_timer.setSingleShot(True)
        self._preview_reload_timer.setInterval(180)
        self._preview_reload_timer.timeout.connect(self.preview_reload)
        self.input_path.textChanged.connect(lambda _t: self._preview_reload_timer.start())

        # Progress
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        # Progress bar is shown alongside Reload preview (see Controls row below)
        self.progress.setTextVisible(True)
        self.progress.setMinimumWidth(140)

        # --- Segment selection ---
        seg_box = QGroupBox("Video segment (optional)")
        seg_layout = QFormLayout(seg_box)

        self.segment_only = QCheckBox("Process segment only")
        self.segment_only.setChecked(False)
        self.segment_only.setEnabled(False)
        self.segment_only.stateChanged.connect(self.on_segment_toggled)
        seg_layout.addRow("", self.segment_only)

        self.seg_start_slider = QSlider(Qt.Horizontal)
        self.seg_end_slider = QSlider(Qt.Horizontal)
        self.seg_start_slider.setEnabled(False)
        self.seg_end_slider.setEnabled(False)
        self.seg_start_slider.valueChanged.connect(self.on_segment_changed)
        self.seg_end_slider.valueChanged.connect(self.on_segment_changed)

        self.seg_start_label = QLabel("Start: 00:00")
        self.seg_end_label = QLabel("End: 00:00")

        seg_layout.addRow(self.seg_start_label, self.seg_start_slider)
        seg_layout.addRow(self.seg_end_label, self.seg_end_slider)
        paths_layout.addRow(seg_box)
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_preview_reload = QPushButton("Reload preview")
        self.btn_preview_reload.setEnabled(True)
        self.btn_preview_reload.clicked.connect(self.preview_reload)

        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self.start_job)
        self.btn_stop.clicked.connect(self.stop_job)

        button_row = QHBoxLayout()
        button_row.addWidget(self.btn_start)
        button_row.addWidget(self.btn_stop)
        button_row.addWidget(self.btn_preview_reload)
        button_row.addWidget(self.progress)
        button_row.addStretch(1)
        paths_layout.addRow("Controls:", button_row)

        # --- Parameters (grouped by functional impact) ---
        toolbox = QToolBox()
        toolbox.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        def dspin(minv, maxv, step, val, decimals=6):
            w = QDoubleSpinBox()
            w.setRange(minv, maxv)
            w.setSingleStep(step)
            w.setDecimals(decimals)
            w.setValue(val)
            return w

        def ispin(minv, maxv, step, val):
            w = QSpinBox()
            w.setRange(minv, maxv)
            w.setSingleStep(step)
            w.setValue(val)
            return w
        # ----------------------------
        # 1) Colour balance & sampling
        # ----------------------------
        page_color = QWidget()
        layout_color = QFormLayout(page_color)

        self.threshold_ratio = dspin(1, 1e7, 50, 2000, decimals=0)
        self.min_avg_red = dspin(0, 255, 1, 45, decimals=2)
        self.max_hue_shift = dspin(0, 10000, 10, 100, decimals=2)
        self.blue_magic_value = dspin(0, 10, 0.05, 1.2, decimals=2)
        self.sample_seconds = dspin(0.01, 3600, 0.25, 2, decimals=2)
        # Number of sampled frames used for windowed mean (odd number)
        self.sample_window_samples = ispin(1, 301, 2, 15)
        self.sample_window_samples.setToolTip(
            "Number of sampled frames averaged per colour-balance estimate.\n"
            "Must be odd.\n"
            "15 = 7 before + current + 7 after.\n"
            "Lower = more responsive, Higher = smoother."
        )
        layout_color.addRow("THRESHOLD_RATIO:", self.threshold_ratio)
        layout_color.addRow("MIN_AVG_RED (rec 40:blue water):", self.min_avg_red)
        layout_color.addRow("MAX_HUE_SHIFT (rec 200:blue water):", self.max_hue_shift)
        layout_color.addRow("BLUE_MAGIC_VALUE (rec 0.9:blue water- increase for deeper blue):", self.blue_magic_value)
        layout_color.addRow("SAMPLE_SECONDS (Increase to reduce sudden shifts):", self.sample_seconds)
        layout_color.addRow(
            "Sample window (frames):",
            self.sample_window_samples
        )
        toolbox.addItem(page_color, "Colour Balance and Sampling")

        # ---------------------------------
        # 2) Tone mapping (shadows/highlights)
        # ---------------------------------
        page_tone = QWidget()
        layout_tone = QFormLayout(page_tone)

        self.shadow_amount_percent = dspin(0, 1, 0.05, 0.7, decimals=2)
        self.shadow_tone_percent = dspin(0, 1, 0.05, 1, decimals=2)
        self.shadow_radius = ispin(0, 9999, 1, 0)

        self.highlight_amount_percent = dspin(0, 1, 0.05, 0.05, decimals=2)
        self.highlight_tone_percent = dspin(0, 1, 0.05, 0.05, decimals=2)
        self.highlight_radius = ispin(0, 9999, 1, 0)

        layout_tone.addRow("shadow_amount_percent (How much shadows are lifted- more detail/depth):", self.shadow_amount_percent)
        layout_tone.addRow("shadow_tone_percent (Which pixels count as shadows- smaller value selects only dark pixels):", self.shadow_tone_percent)

        layout_tone.addRow("highlight_amount_percent (How strongly highlights are reshaped):", self.highlight_amount_percent)
        layout_tone.addRow("highlight_tone_percent (Which pixels are highlights smaller value selects brighter pixels):", self.highlight_tone_percent)

        toolbox.addItem(page_tone, "Tone Mapping (Shadows/Highlights)")

        # ---------------------------------
        # 3) Temporal stability (anti-pumping)
        # ---------------------------------
        page_stab = QWidget()
        layout_stab = QFormLayout(page_stab)

        # Filter-matrix temporal stabilisation (reduces sudden colour shifts)
        self.filter_smooth_alpha = QDoubleSpinBox()
        self.filter_smooth_alpha.setRange(0.0, 1.0)
        self.filter_smooth_alpha.setSingleStep(0.01)
        self.filter_smooth_alpha.setDecimals(3)
        self.filter_smooth_alpha.setValue(0.05)
        self.filter_smooth_alpha.setToolTip(
            "EMA smoothing factor for per-frame colour filter matrices."
            "0 disables (original behaviour). Typical 0.01..0.20."
            "Higher = smoother but slower adaptation."
        )

        self.filter_max_delta = QDoubleSpinBox()
        self.filter_max_delta.setRange(0.0, 5.0)
        self.filter_max_delta.setSingleStep(0.005)
        self.filter_max_delta.setDecimals(4)
        self.filter_max_delta.setValue(0.02)
        self.filter_max_delta.setToolTip(
            "Clamp the per-frame change applied to filter coefficients (L-infinity)."
            "0 disables. Typical 0.005..0.05 (depends on footage)."
            "Lower = more stable, but can under-correct rapid lighting changes."
        )
        # params_layout.addRow("Filter max delta (clamp: If colour shifting rapidly lower max_delta then increase alpha slightly):", self.filter_max_delta)
        # params_layout.addRow("Filter smooth alpha (EMA):", self.filter_smooth_alpha)

        # Auto-contrast temporal stabilisation (EMA + clamp on alpha/beta)
        self.ac_smooth_alpha = QDoubleSpinBox()
        self.ac_smooth_alpha.setRange(0.0, 1.0)
        self.ac_smooth_alpha.setSingleStep(0.01)
        self.ac_smooth_alpha.setDecimals(3)
        self.ac_smooth_alpha.setValue(0.10)
        self.ac_smooth_alpha.setToolTip(
            "EMA smoothing factor for auto-contrast alpha/beta. "
            "0 disables. Typical 0.05..0.30. "
            "Higher = smoother but slower exposure adaptation."
        )

        self.ac_max_delta_alpha = QDoubleSpinBox()
        self.ac_max_delta_alpha.setRange(0.0, 5.0)
        self.ac_max_delta_alpha.setSingleStep(0.01)
        self.ac_max_delta_alpha.setDecimals(3)
        self.ac_max_delta_alpha.setValue(0.05)
        self.ac_max_delta_alpha.setToolTip(
            "Clamp per-update change in auto-contrast alpha (multiplicative). "
            "0 disables. Typical 0.01..0.10."
        )

        self.ac_max_delta_beta = QDoubleSpinBox()
        self.ac_max_delta_beta.setRange(0.0, 255.0)
        self.ac_max_delta_beta.setSingleStep(0.5)
        self.ac_max_delta_beta.setDecimals(1)
        self.ac_max_delta_beta.setValue(2.0)
        self.ac_max_delta_beta.setToolTip(
            "Clamp per-update change in auto-contrast beta (additive). "
            "0 disables. Typical 0.5..5."
        )
        layout_stab.addRow("Filter max delta (clamp):", self.filter_max_delta)
        layout_stab.addRow("Filter smooth alpha (EMA):", self.filter_smooth_alpha)
        layout_stab.addRow("Auto-contrast smooth alpha (EMA):", self.ac_smooth_alpha)
        layout_stab.addRow("Auto-contrast max delta alpha (clamp):", self.ac_max_delta_alpha)
        layout_stab.addRow("Auto-contrast max delta beta (clamp):", self.ac_max_delta_beta)

        toolbox.addItem(page_stab, "Temporal Stability (Anti-pumping)")

        # ----------------------------
        # 4) Contrast / exposure control
        # ----------------------------
        page_contrast = QWidget()
        layout_contrast = QFormLayout(page_contrast)

        self.clip_hist_percent_in = dspin(0, 100, 0.05, 0.1, decimals=2)

        self.disable_auto_contrast = QCheckBox("Disable auto brightness/contrast (more stable)")
        self.disable_auto_contrast.setChecked(False)
        self.disable_auto_contrast.setToolTip(
            "If enabled, skips the auto brightness/contrast stage that can cause rapid pumping with lighting changes."
        )

        self.auto_contrast_every_n_frames = QSpinBox()
        self.auto_contrast_every_n_frames.setRange(0, 1000000)
        self.auto_contrast_every_n_frames.setSingleStep(1)
        self.auto_contrast_every_n_frames.setValue(0)  # 0 = auto (~fps)
        self.auto_contrast_every_n_frames.setToolTip(
            "Recompute auto-contrast histogram every N frames. "
            "0 = auto (about once per second). Higher values = less reactive."
        )

        layout_contrast.addRow("clip_hist_percent_in (reduce if pumping observed):", self.clip_hist_percent_in)
        layout_contrast.addRow(self.disable_auto_contrast)
        layout_contrast.addRow("Auto-contrast every N frames (0= FPS):", self.auto_contrast_every_n_frames)

        toolbox.addItem(page_contrast, "Contrast & Exposure")

        # ----------------------------
        # 5) Performance / speed
        # ----------------------------
        page_perf = QWidget()
        layout_perf = QFormLayout(page_perf)

        self.precompute_filters = QCheckBox("Precompute filters (faster, uses more RAM)")
        self.precompute_filters.setChecked(True)

        self.fast_hs = QCheckBox("Fast shadows/highlights (approximate, much faster)")
        self.fast_hs.setChecked(True)
        self.fast_hs.setToolTip(
            "Computes shadow/highlight maps at lower resolution\n"
            "then applies them to full-resolution output.\n"
            "Large speed-up with minimal quality loss."
        )

        self.fast_hs_map_scale = QDoubleSpinBox()
        self.fast_hs_map_scale.setRange(0.10, 1.00)
        self.fast_hs_map_scale.setSingleStep(0.05)
        self.fast_hs_map_scale.setDecimals(2)
        self.fast_hs_map_scale.setValue(0.1)
        self.fast_hs_map_scale.setToolTip("Map scale for fast HS (0.10..1.00). Smaller = faster, slightly lower fidelity.")
        self.fast_hs_map_scale.setEnabled(True)
        self.fast_hs.toggled.connect(self.fast_hs_map_scale.setEnabled)

        self.downsample_factor = QComboBox()
        self.downsample_factor.addItems(["1 (full)", "2 (half)", "4 (quarter)", "8 (eighth)", "16 (sixteenth)"])
        self.downsample_factor.setCurrentIndex(0)
        self.downsample_factor.setToolTip(
            "Downsample factor for video processing. Higher values are faster but reduce output/preview resolution."
        )

        layout_perf.addRow(self.precompute_filters)
        layout_perf.addRow(self.fast_hs)
        layout_perf.addRow("Fast HS map scale (lower=faster)", self.fast_hs_map_scale)
        layout_perf.addRow("Downsample (processing/preview):", self.downsample_factor)

        toolbox.addItem(page_perf, "Performance")

        top_row = QHBoxLayout()

        top_row = QHBoxLayout()
        top_row.addWidget(paths_box, 3)
        # Stack tuning + performance on the right
        right_stack = QVBoxLayout()
        right_stack.addWidget(toolbox)
        right_wrap = QWidget()
        right_wrap.setLayout(right_stack)
        top_row.addWidget(right_wrap, 3)
        root.addLayout(top_row)

        # --- Controls ---
        # ctrl_row = QHBoxLayout()
        # self.btn_start = QPushButton("Start")
        # self.btn_stop = QPushButton("Stop")
        # self.btn_preview_init = QPushButton("Initialize preview")
        # self.btn_preview_init.clicked.connect(self.init_preview)
        # #ctrl_row.addWidget(self.btn_preview_init)
        # self.btn_stop.setEnabled(False)
        # self.btn_start.clicked.connect(self.start_job)
        # self.btn_stop.clicked.connect(self.stop_job)
        # ctrl_row.addWidget(self.btn_start)
        # ctrl_row.addWidget(self.btn_stop)
        # ctrl_row.addStretch(1)
        # root.addLayout(ctrl_row)

        self.preview_container = QWidget()
        self.preview_container_layout = QVBoxLayout(self.preview_container)

        # --- Logs ---
        # Command line / log box removed by request.
        self.log = None
        
        # ... top UI ...

        root.addWidget(self.preview_container, 1)
        # Init
        self.on_same_folder_changed()

        # Update original preview when input changes (if preview initialized)
        self.input_path.textChanged.connect(self._maybe_load_original_preview)
        self.output_path.textChanged.connect(self._maybe_load_processed_preview)

        # When tuning changes, refresh frame preview (if that tab is active)
        self._wire_frame_preview_param_signals()
        # Eagerly initialise preview on app start.
        # This removes the need for an explicit "Initialize preview" button/state.
        try:
            self.init_preview()
        except Exception:
            # Never allow preview init to prevent the app from starting.
            pass

    def _wire_frame_preview_param_signals(self):
        """Connect tuning widgets to trigger a frame-preview refresh.

        This is intentionally best-effort: if any widget is missing, we skip it.
        """
        def hook(w):
            if w is None:
                return
            # QDoubleSpinBox/QSpinBox
            if hasattr(w, "valueChanged"):
                try:
                    w.valueChanged.connect(self._on_frame_params_changed)
                    return
                except Exception:
                    pass
            # QCheckBox
            if hasattr(w, "stateChanged"):
                try:
                    w.stateChanged.connect(self._on_frame_params_changed)
                    return
                except Exception:
                    pass
            if hasattr(w, "toggled"):
                try:
                    w.toggled.connect(self._on_frame_params_changed)
                except Exception:
                    pass
            # QComboBox
            if hasattr(w, "currentIndexChanged"):
                try:
                    w.currentIndexChanged.connect(self._on_frame_params_changed)
                except Exception:
                    pass

        for widget in [
            # Colour balance
            self.threshold_ratio, self.min_avg_red, self.max_hue_shift, self.blue_magic_value, self.sample_seconds, self.sample_window_samples,
            # Tone mapping
            self.shadow_amount_percent, self.shadow_tone_percent, self.shadow_radius,
            self.highlight_amount_percent, self.highlight_tone_percent, self.highlight_radius,
            # Temporal stability (not used in single frame, but user expects changes to reflect)
            self.filter_smooth_alpha, self.filter_max_delta,
            self.ac_smooth_alpha, self.ac_max_delta_alpha, self.ac_max_delta_beta,
            # Contrast/exposure
            self.clip_hist_percent_in, self.disable_auto_contrast, self.auto_contrast_every_n_frames,
            # Performance
            self.fast_hs, self.fast_hs_map_scale, self.downsample_factor,
            self.sample_window_samples,  
        ]:
            hook(widget)

    def _on_frame_params_changed(self, *args):
        """Best-effort: if the Frame scrub tab is active, re-render the current frame."""
        if getattr(self, "_frame_param_guard", False):
            return
        if not getattr(self, "_preview_available", False):
            return
        if not getattr(self, "_frame_loaded", False):
            return
        tabs = getattr(self, "preview_tabs", None)
        if tabs is None:
            return
        if getattr(self, "tab_frame", None) is None:
            return
        if tabs.currentWidget() is self.tab_frame:
            self._schedule_frame_render(self._frame_last_rel_ms)

    def _schedule_frame_render(self, rel_ms: int):
        """Debounced frame (re)render."""
        self._frame_last_rel_ms = int(max(0, rel_ms))
        t = getattr(self, "_frame_req_timer", None)
        if t is None:
            return
        t.start()

    def _orig_base_ms(self) -> int:
        # Original should respect the selected segment timestamp
        return self._seg_start_ms if (self.segment_only.isChecked() and self._seg_duration_ms > 0) else 0

    def _proc_base_ms(self) -> int:
        # Processed should always start from 0 (start of processed file)
        # even when original is offset into a segment.
        return 0

    # ---------- Logging ----------
    def append_log(self, text: str) -> None:
        # Log UI removed; keep for backward compatibility (no-op).
        if self.log is None:
            return
        self.log.append(text.rstrip())

    # ---------- Browse ----------
    def browse_script(self):
        start = self._open_dir_for_path(self.script_path.text().strip())
        p, _ = QFileDialog.getOpenFileName(
            self, "Select backend script", start, "Python Files (*.py)"
        )
        if p:
            self.script_path.setText(p)


    def browse_input(self):
        start = self._open_dir_for_path(self.input_path.text().strip())
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Select input file",
            start,
            "Media (*.mp4 *.mov *.mkv *.avi *.png *.jpg *.jpeg *.tif *.tiff);;All (*.*)"
        )
        if not p:
            return

        self.input_path.setText(p)

        # keep your existing logic below unchanged


        ext = Path(p).suffix.lower()
        if ext in {".mp4", ".mov", ".mkv", ".avi"}:
            self._video_duration_sec = self._probe_video_duration_seconds(Path(p))
            dur_int = int(max(0, round(self._video_duration_sec)))
            self.segment_only.setEnabled(dur_int > 0)
            self.seg_start_slider.setRange(0, max(0, dur_int))
            self.seg_end_slider.setRange(0, max(0, dur_int))
            self.seg_start_slider.setValue(0)
            self.seg_end_slider.setValue(dur_int if dur_int > 0 else 0)
            self.seg_start_label.setText("Start: 00:00")
            self.seg_end_label.setText(f"End: {self._format_time(dur_int)}")
            self.on_segment_toggled()
        else:
            self._video_duration_sec = 0.0
            self.segment_only.setChecked(False)
            self.segment_only.setEnabled(False)
            self.seg_start_slider.setEnabled(False)
            self.seg_end_slider.setEnabled(False)

        if self.same_folder.isChecked():
            self.output_path.setText(self.default_output_for(Path(p)))

    def browse_output(self):
        inp = self.input_path.text().strip()
        start = str(Path(inp).parent) if inp else str(Path.home())
        p, _ = QFileDialog.getSaveFileName(self, "Select output file", start, "All (*.*)")
        if p:
            self.output_path.setText(p)

    # ---------- Output behaviour ----------
    def on_same_folder_changed(self):
        use_same = self.same_folder.isChecked()
        self.output_path.setEnabled(not use_same)
        if use_same:
            inp = self.input_path.text().strip()
            if inp:
                self.output_path.setText(self.default_output_for(Path(inp)))

    # ---------- Segment ----------
    def on_segment_toggled(self):
        enabled = self.segment_only.isChecked() and self._video_duration_sec > 0
        self.seg_start_slider.setEnabled(enabled)
        self.seg_end_slider.setEnabled(enabled)

        if not enabled:
            # Segment mode OFF: clear any segment offset and restore full-range UI
            self._seg_start_ms = 0
            self._seg_duration_ms = 0

            dur_int = int(max(0, round(self._video_duration_sec)))
            self.seg_start_slider.blockSignals(True)
            self.seg_end_slider.blockSignals(True)
            try:
                self.seg_start_slider.setValue(0)
                self.seg_end_slider.setValue(dur_int)
            finally:
                self.seg_start_slider.blockSignals(False)
                self.seg_end_slider.blockSignals(False)

            self.seg_start_label.setText("Start: 00:00")
            self.seg_end_label.setText(f"End: {self._format_time(dur_int)}")

            # If preview exists, recompute window and reset scrubbers to 0
            if self._preview_available:
                self._recompute_preview_window()
                try:
                    self.preview_slider.setValue(0)
                    self._update_preview_time_label(0)
                except Exception:
                    pass

                if hasattr(self, "frame_slider") and self.frame_slider is not None:
                    try:
                        self.frame_slider.setValue(0)
                        self._frame_last_rel_ms = 0
                        self._update_frame_time_label(0)
                    except Exception:
                        pass
            return

        # Segment mode ON: normal behavior
        self.on_segment_changed()

    def on_segment_changed(self):
        if self._video_duration_sec <= 0:
            return

        start = int(self.seg_start_slider.value())
        end = int(self.seg_end_slider.value())

        if end <= start:
            end = start + 1
            self.seg_end_slider.blockSignals(True)
            self.seg_end_slider.setValue(end)
            self.seg_end_slider.blockSignals(False)

        self._seg_start_ms = start * 1000
        self._seg_duration_ms = (end - start) * 1000

        self.seg_start_label.setText(f"Start: {self._format_time(start)}")
        self.seg_end_label.setText(f"End: {self._format_time(end)}")

        # # Update preview slider range if preview exists
        # if self._preview_available:
        #     self.preview_slider.setRange(0, self._seg_duration_ms)
        #     self.preview_slider.setValue(0)
        #     self._update_preview_time_label(0)
        if self._preview_available:
            self._recompute_preview_window()
            self.preview_slider.setValue(0)
            self._update_preview_time_label(0)


    # ---------- Helpers ----------
    @staticmethod
    def _format_time(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        m = int(seconds // 60)
        s = int(round(seconds - 60 * m))
        return f"{m:02d}:{s:02d}"

    @staticmethod
    def _probe_video_duration_seconds(path: Path) -> float:
        if cv2 is not None:
            cap = cv2.VideoCapture(str(path))
            try:
                fps = cap.get(getattr(cv2, "CAP_PROP_FPS", 5)) or 0.0
                frames = cap.get(getattr(cv2, "CAP_PROP_FRAME_COUNT", 7)) or 0.0
                if fps and frames:
                    return float(frames) / float(fps)
            finally:
                cap.release()
        return 0.0

    @staticmethod
    def default_output_for(inp: Path) -> str:
        return str(inp.with_name(f"{inp.stem}.corrected{inp.suffix}"))
    
    def _on_preview_duration_changed(self, dur_ms: int):
        if not self._preview_available:
            return
        self._orig_duration_ms = max(0, int(dur_ms))
        self._recompute_preview_window()

    def _on_proc_duration_changed(self, dur_ms: int):
        if not self._preview_available:
            return
        self._proc_duration_ms = max(0, int(dur_ms))
        self._recompute_preview_window()

    def _recompute_preview_window(self):
        """
        Defines the preview timeline length (slider range and stop point).
        Segment-only: exactly segment length.
        Otherwise: full playback window, bounded by available durations.
        """
        if not self._preview_available:
            return

        if self.segment_only.isChecked() and self._seg_duration_ms > 0:
            win = int(self._seg_duration_ms)
        else:
            # If processed not loaded yet, fall back to original duration
            if self._proc_duration_ms > 0:
                win = int(min(self._orig_duration_ms, self._proc_duration_ms))
            else:
                win = int(self._orig_duration_ms)

        win = max(0, win)
        self._preview_window_ms = win

        self.preview_slider.setRange(0, win)
        # Frame scrub should be based on ORIGINAL timeline (or segment length if segment-only),
        # not limited by processed preview duration.
        if hasattr(self, "frame_slider") and self.frame_slider is not None:
            try:
                if self.segment_only.isChecked() and self._seg_duration_ms > 0:
                    frame_max = int(self._seg_duration_ms)
                else:
                    frame_max = int(self._orig_duration_ms)

                self.frame_slider.setRange(0, max(0, frame_max))
                self._frame_last_rel_ms = min(int(self._frame_last_rel_ms), max(0, frame_max))
                self.frame_slider.setValue(self._frame_last_rel_ms)
                self._update_frame_time_label(self._frame_last_rel_ms)
            except Exception:
                pass
        # Do NOT force slider back to 0 here; keep current position if possible
        cur = min(self.preview_slider.value(), win)
        self.preview_slider.setValue(cur)
        self._update_preview_time_label(cur)
        
    def _on_frame_slider_value_changed(self, v: int):
        self._frame_last_rel_ms = int(max(0, v))
        self._update_frame_time_label(self._frame_last_rel_ms)
        self._schedule_frame_render(self._frame_last_rel_ms)

    def _on_play_scopes_mode_changed(self, state: int):
        """
        Slot: toggles continuous playback scopes refresh (RGB parade / waveform)
        during the Playback tab. Safe even if preview is not initialised.
        """
        self._play_scopes_continuous_enabled = bool(state)

        # If preview not ready, nothing to do.
        if not getattr(self, "_preview_available", False):
            return

        # If turning OFF, stop any scheduled refresh work.
        if not self._play_scopes_continuous_enabled:
            try:
                if getattr(self, "_play_scope_timer", None) is not None:
                    self._play_scope_timer.stop()
            except Exception:
                pass
            return

        # If turning ON, only run when the Playback tab is active.
        try:
            tabs = getattr(self, "preview_tabs", None)
            if tabs is not None and tabs.currentWidget() is getattr(self, "tab_playback", None):
                # This method exists in your file (it is referenced already).
                self._maybe_start_continuous_playback_scopes()
        except Exception:
            # Never allow scopes to crash the UI.
            pass

    # ---------- Preview ----------
    def init_preview(self):
        if self._preview_initialized:
            return
        self._preview_initialized = True

        try:
            from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
            from PySide6.QtMultimediaWidgets import QVideoWidget
        except Exception as e:
            # self.preview_status.setText(f"Preview unavailable (QtMultimedia import failed): {e}")
            self._preview_available = False
            return
        self._preview_available = True
        self.btn_preview_reload.setEnabled(True)

        # Build UI (tabs)
        self.preview_tabs = QTabWidget()
        self.preview_container_layout.addWidget(self.preview_tabs, 1)

        # --- Tab 1: Playback (existing behaviour) ---
        self.tab_playback = QWidget()
        tab_playback_layout = QVBoxLayout(self.tab_playback)

        videos_row = QHBoxLayout()
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()

        # left_col.addWidget(QLabel("Original", alignment=Qt.AlignHCenter))
        # right_col.addWidget(QLabel("Processed", alignment=Qt.AlignHCenter))

        self.video_orig = QVideoWidget()
        self.video_proc = QVideoWidget()

        # Ensure previews are large and scale with the window
        for vw in (self.video_orig, self.video_proc):
            vw.setMinimumSize(320, 180)
            # Downscale playback display to reduce render load; keep aspect via layout height.
            # vw.setMaximumWidth(210)
            # vw.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
            vw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scopes_playback_orig = AsyncScopesWidget(title="Original")
        self.scopes_playback_proc = AsyncScopesWidget(title="Processed")

        # Ensure initial visibility matches the playback checkbox (default OFF)
        self.scopes_playback_orig.setVisible(False)
        self.scopes_playback_proc.setVisible(False)

        orig_row = QHBoxLayout()
        orig_row.addWidget(self.video_orig, 1)
        orig_row.addWidget(self.scopes_playback_orig, 0)
        # Processed playback status (shown inline instead of popups)
        self.proc_playback_status = QLabel("")
        self.proc_playback_status.setWordWrap(True)
        self.proc_playback_status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        try:
            self.proc_playback_status.setStyleSheet('color: #b00020;')
        except Exception:
            pass

        proc_video_col = QVBoxLayout()
        proc_video_col.addWidget(self.video_proc, 1)
        proc_video_col.addWidget(self.proc_playback_status, 0)

        proc_row = QHBoxLayout()
        proc_row.addLayout(proc_video_col, 1)
        # proc_row = QHBoxLayout()
        # proc_row.addWidget(self.video_proc, 1)
        proc_row.addWidget(self.scopes_playback_proc, 0)

        left_col.addLayout(orig_row, 1)
        right_col.addLayout(proc_row, 1)

        videos_row.addLayout(left_col, 1)
        videos_row.addLayout(right_col, 1)
        tab_playback_layout.addLayout(videos_row, 1)

        controls = QHBoxLayout()
        self.btn_preview_play = QPushButton("▶/⏸")
        self.btn_preview_stop = QPushButton("⏹")
        self.btn_preview_play.clicked.connect(self.preview_play_pause)
        self.btn_preview_stop.clicked.connect(self.preview_stop)
        self.show_scopes = QCheckBox("Show scopes")
        self.show_scopes.setChecked(False)  # playback scopes default OFF
        self.show_scopes.toggled.connect(self._on_show_scopes_toggled)
        controls.addWidget(self.show_scopes)

        # Apply initial playback scopes state
        try:
            self._on_show_scopes_toggled(self.show_scopes.isChecked())
        except Exception:
            pass

        self.preview_slider = ClickSeekSlider(Qt.Horizontal)
        self.preview_slider.clickedValue.connect(self.preview_seek)

        self.preview_slider.setRange(0, 0)
        self.preview_slider.sliderMoved.connect(self.preview_seek)        # while dragging
        self.preview_slider.sliderReleased.connect(lambda: self.preview_seek(self.preview_slider.value()))

        self.preview_time = QLabel("00:00 / 00:00")
        self.preview_time.setToolTip("Click to go to time (MM:SS or HH:MM:SS). Prefix with 'orig ' for original timeline.")
        self.preview_time.setCursor(Qt.PointingHandCursor)
        self.preview_time.installEventFilter(self)

        self.preview_time.setMinimumWidth(120)

        self.mute_original = QCheckBox("Mute original")
        self.mute_processed = QCheckBox("Mute processed")
        self.mute_original.setChecked(True)
        self.mute_processed.setChecked(True)
        self.mute_original.stateChanged.connect(self._apply_preview_mute)
        self.mute_processed.stateChanged.connect(self._apply_preview_mute)

        controls.addWidget(self.btn_preview_play)
        controls.addWidget(self.btn_preview_stop)
        controls.addWidget(self.preview_slider, 1)
        controls.addWidget(self.preview_time)
        controls.addWidget(self.mute_original)
        controls.addWidget(self.mute_processed)
        # Playback scopes: continuous update toggle (default OFF)
        self.play_scopes_continuous = QCheckBox("Continuous scopes")
        self.play_scopes_continuous.setChecked(False)
        self.play_scopes_continuous.setToolTip(
            "If enabled, scopes refresh continuously during playback (higher CPU). \n"
            "If disabled, scopes refresh only on seeks/position updates.\n"
        )
        self.play_scopes_continuous.stateChanged.connect(self._on_play_scopes_mode_changed)
        # controls.addWidget(self.play_scopes_continuous)
        tab_playback_layout.addLayout(controls)

        

        # --- Tab 2: Frame scrub (single-frame before/after) ---
        self.tab_frame = QWidget()
        tab_frame_layout = QVBoxLayout(self.tab_frame)

        frame_row = QHBoxLayout()
        f_left = QVBoxLayout()
        f_right = QVBoxLayout()
        # f_left.addWidget(QLabel("Original frame", alignment=Qt.AlignHCenter))
        # f_right.addWidget(QLabel("Processed frame", alignment=Qt.AlignHCenter))

        self.frame_label_orig = QLabel("(no frame loaded)")
        self.frame_label_proc = QLabel("(no frame loaded)")
        for lab in (self.frame_label_orig, self.frame_label_proc):
            lab.setAlignment(Qt.AlignCenter)
            lab.setMinimumSize(320, 180)
            lab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            lab.setStyleSheet("QLabel { background: #111; color: #ddd; border: 1px solid #333; }")
        self.scopes_frame_orig = AsyncScopesWidget(title="Scopes (Original)")
        self.scopes_frame_proc = AsyncScopesWidget(title="Scopes (Processed)")

        f_orow = QHBoxLayout()
        f_orow.addWidget(self.frame_label_orig, 1)
        f_orow.addWidget(self.scopes_frame_orig, 0)
        f_prow = QHBoxLayout()
        f_prow.addWidget(self.frame_label_proc, 1)
        f_prow.addWidget(self.scopes_frame_proc, 0)

        f_left.addLayout(f_orow, 1)
        f_right.addLayout(f_prow, 1)
        frame_row.addLayout(f_left, 1)
        frame_row.addLayout(f_right, 1)
        tab_frame_layout.addLayout(frame_row, 1)

        # --- Preset thumbnails panel (applies parameter bundles to the selected frame) ---
        if self._preset_defs is None:
            self._preset_defs = self._build_preset_definitions()

        # Preset thumbnails are mounted directly within the Frame scrub tab (no enclosing group box).
        preset_panel = QWidget()
        preset_panel.setToolTip(
            "Select a preset to apply a parameter bundle to the global settings.\n"
            "This updates the tuning fields and will be used by downstream batch correction."
        )
        preset_layout = QVBoxLayout(preset_panel)
        preset_layout.setContentsMargins(2, 2, 2, 2)

        self._preset_btn_group = QButtonGroup(self)
        self._preset_btn_group.setExclusive(True)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # The strip height is dynamically tied to the video preview height.
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._preset_scroll = scroll

        inner = QWidget()
        self._preset_inner = inner
        inner_row = QHBoxLayout(inner)
        inner_row.setContentsMargins(2, 2, 2, 2)
        inner_row.setSpacing(4)

        # Create one button per preset; icons are populated when a frame is loaded.
        self._preset_buttons = {}
        for name in self._preset_defs.keys():
            b = QToolButton()
            b.setText(name)
            b.setCheckable(True)
            b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            # Geometry is set by _update_preset_strip_geometry() (driven by preview height).
            b.setAutoRaise(True)
            b.clicked.connect(lambda checked=False, n=name: self._on_preset_button_clicked(n))
            self._preset_btn_group.addButton(b)
            self._preset_buttons[name] = b
            inner_row.addWidget(b)

        inner_row.addStretch(1)
        scroll.setWidget(inner)
        preset_layout.addWidget(scroll)
        tab_frame_layout.addWidget(preset_panel, 0)

        # Keep preset thumbnails sized to the current preview video height.
        try:
            self._preset_resize_hook = _ResizeHook(self._update_preset_strip_geometry)
            self.tab_frame.installEventFilter(self._preset_resize_hook)
        except Exception:
            pass
        self._update_preset_strip_geometry()
        scrub_controls = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.sliderMoved.connect(self._on_frame_slider_moved)
        self.frame_slider.sliderReleased.connect(self._on_frame_slider_released)
        # Frame scrub: show/hide scopes (sync with playback checkbox)
        self.show_scopes_frame = QCheckBox("Show scopes")
        self.show_scopes_frame.setChecked(True)  # independent default for Frame scrub scopes
        self.show_scopes_frame.toggled.connect(self._on_show_scopes_frame_toggled)
        scrub_controls.addWidget(self.show_scopes_frame)
        self.frame_time = QLabel("00:00")
        self.frame_time.setMinimumWidth(70)

        self.btn_frame_refresh = QPushButton("Refresh")
        self.btn_frame_refresh.clicked.connect(self._refresh_frame_preview)
        self.frame_slider.valueChanged.connect(self._on_frame_slider_value_changed)


        scrub_controls.addWidget(QLabel("Scrub:"))
        scrub_controls.addWidget(self.frame_slider, 1)
        scrub_controls.addWidget(self.frame_time)
        # scrub_controls.addWidget(self.btn_frame_refresh)
        tab_frame_layout.addLayout(scrub_controls)

        self.preview_tabs.addTab(self.tab_frame, "Frame scrub")
        self.preview_tabs.addTab(self.tab_playback, "Playback")

        # When switching to Frame scrub, render the current slider position.
        self.preview_tabs.currentChanged.connect(self._on_preview_tab_changed)

        # Internal debounce timer for frame updates
        self._frame_req_timer = QTimer(self)
        self._frame_req_timer.setSingleShot(True)
        self._frame_req_timer.setInterval(80)
        self._frame_req_timer.timeout.connect(self._refresh_frame_preview)

        # Re-render processed frame argsing parameters change (only when frame tab is active)
        self._wire_frame_preview_param_signals()

        # Players
        self.player_orig = QMediaPlayer(self)
        self.player_proc = QMediaPlayer(self)
        self.audio_orig = QAudioOutput(self)
        self.audio_proc = QAudioOutput(self)
        self.player_orig.setAudioOutput(self.audio_orig)
        self.player_proc.setAudioOutput(self.audio_proc)
        self.player_orig.setVideoOutput(self.video_orig)
        self.player_proc.setVideoOutput(self.video_proc)

        self.player_orig.positionChanged.connect(self._on_preview_position_changed)
        self.player_orig.durationChanged.connect(self._on_preview_duration_changed)
        self.player_proc.durationChanged.connect(self._on_proc_duration_changed)

        

        self._sync_timer = QTimer(self)
        self._sync_timer.setInterval(50)
        self._sync_timer.timeout.connect(self._sync_preview_players)

        self._apply_preview_mute()

        # Playback scopes (throttled): render scopes next to videos during playback/seek
        self._play_scope_cap_orig = None
        self._play_scope_cap_proc = None
        self._play_scope_timer = QTimer(self)
        self._play_scope_timer.setSingleShot(True)
        self._play_scope_timer.setInterval(120)
        # Continuous mode uses a repeating timer; default is OFF
        self._play_scopes_continuous_enabled = False
        self._play_scope_timer.timeout.connect(self._on_play_scope_timer_timeout)
        self._play_scope_last_rel_ms = 0

        # Load current paths
        self._maybe_load_original_preview()
        self._maybe_load_processed_preview()

        # Prime the Frame scrub tab with the first frame immediately.
        # (QTabWidget.currentChanged does not fire for the initially-selected tab.)
        try:
            QTimer.singleShot(0, lambda: self._schedule_frame_render(0))
        except Exception:
            pass

    def _apply_preview_mute(self):
        if not self._preview_available:
            return
        try:
            self.audio_orig.setMuted(self.mute_original.isChecked())
            self.audio_proc.setMuted(self.mute_processed.isChecked())
        except Exception:
            pass

    def _maybe_load_original_preview(self):
        if not self._preview_available:
            return
        p = self.input_path.text().strip()
        if not p:
            return
        fp = Path(p)
        if fp.exists() and fp.is_file():
            self.player_orig.setSource(QUrl.fromLocalFile(str(fp.resolve())))
            self.player_orig.pause()

    def _maybe_load_processed_preview(self, force: bool = False):
        if not self._preview_available or self.player_proc is None:
            return False

        p = self.output_path.text().strip()
        if not p:
            return False

        fp = Path(p)
        if not (fp.exists() and fp.is_file()):
            return False

        url = QUrl.fromLocalFile(str(fp.resolve()))

        # If not forcing and already set, do nothing
        if (not force) and (not self.player_proc.source().isEmpty()) and (self.player_proc.source() == url):
            return True

        # Force a true reload: clear first, then set again
        self.player_proc.stop()
        self.player_proc.setSource(QUrl())   # clear
        self.player_proc.setSource(url)
        self.player_proc.pause()
        return True

    def _set_processed_playback_message(self, msg: str) -> None:
        """Show a non-blocking message in the Processed playback pane."""
        lab = getattr(self, 'proc_playback_status', None)
        if lab is None:
            return
        try:
            lab.setText(str(msg or ''))
            lab.setVisible(bool(msg))
        except Exception:
            pass

    def _clear_processed_playback_message(self) -> None:
        self._set_processed_playback_message('')

    # ---------- Frame-scrub preview ----------
    def _on_preview_tab_changed(self, idx: int):
        # If the user moves into the frame-scrub tab, render immediately.
        tabs = getattr(self, "preview_tabs", None)
        if tabs is None or getattr(self, "tab_frame", None) is None:
            return
        if tabs.currentWidget() is self.tab_frame:
            self._schedule_frame_render(getattr(self, "_frame_last_rel_ms", 0))
# If user switches into Playback tab, optionally start continuous scopes.
        if tabs.currentWidget() is getattr(self, "tab_playback", None):
            try:
                if getattr(self, "_play_scopes_continuous_enabled", False):
                    self._maybe_start_continuous_playback_scopes()
            except Exception:
                pass
        else:
            # Leaving playback: stop continuous scopes timer to reduce CPU.
            try:
                if getattr(self, "_play_scopes_continuous_enabled", False):
                    self._play_scope_timer.stop()
            except Exception:
                pass
    def _on_frame_slider_moved(self, v: int):
        self._frame_last_rel_ms = int(max(0, v))
        self._update_frame_time_label(self._frame_last_rel_ms)
        self._schedule_frame_render(self._frame_last_rel_ms)

    def _on_frame_slider_released(self):
        self._schedule_frame_render(self._frame_last_rel_ms)

    def _update_frame_time_label(self, rel_ms: int):
        self.frame_time.setText(self._fmt_ms(int(rel_ms)))

    def _fmt_ms(self, ms: int) -> str:
        ms = int(max(0, ms))
        s = ms // 1000
        m = s // 60
        s = s % 60
        return f"{m:02d}:{s:02d}"

    def _update_preset_strip_geometry(self) -> None:
        """Keep the preset thumbnail strip and icons sized in proportion to the video preview height."""
        scroll = getattr(self, "_preset_scroll", None)
        inner = getattr(self, "_preset_inner", None)
        buttons = getattr(self, "_preset_buttons", None) or {}
        if scroll is None or not buttons:
            return
        # Use the tab height as the driver (NOT the preview label height).
        # Using the label height creates a feedback loop because changing the strip height
        # changes the available label height, which can cause oscillation/jitter.
        try:
            tab_h = int(getattr(self, "tab_frame", None).height())
        except Exception:
            tab_h = 0
        if tab_h <= 0:
            return

        # Allocate a meaningful fraction of the tab height to the strip.
        strip_h = max(120, min(440, int(tab_h * 0.22)))
        scroll.setMinimumHeight(strip_h)
        scroll.setMaximumHeight(strip_h)
        if inner is not None:
            inner.setMinimumHeight(strip_h)

        # Reserve room for the text label under the icon.
        label_room = 44
        icon_h = max(48, strip_h - label_room)
        icon_w = max(64, int(icon_h * 16 / 9))
        # Avoid resize→thumbnail-refresh→resize jitter: only apply geometry when it changes.
        new_geom = (strip_h, icon_w, icon_h)
        if getattr(self, "_preset_geom_cache", None) == new_geom:
            return
        self._preset_geom_cache = new_geom

        for b in buttons.values():
            try:
                b.setIconSize(QSize(icon_w, icon_h))
                b.setMinimumWidth(icon_w + 20)
                b.setMinimumHeight(strip_h - 6)
                b.setMaximumHeight(strip_h - 2)
                b.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            except Exception:
                pass

        # Resize should NOT trigger a thumbnail re-render (it causes flicker / cross-thumbnail refresh).
        # Instead, just rescale icons from the already-cached thumbnail frames.
        self._update_preset_icons_from_cache()


    def _update_preset_icons_from_cache(self) -> None:
        """Rescale preset button icons from cached thumbnail frames (no re-render)."""
        buttons = getattr(self, "_preset_buttons", None) or {}
        cache = getattr(self, "_preset_thumb_frames", None) or {}
        if not buttons or not cache:
            return
        for name, btn in buttons.items():
            try:
                cached = cache.get(name)
                if cached is None or not isinstance(cached, tuple) or len(cached) != 2:
                    continue
                _orig_bgr, proc_bgr = cached
                if proc_bgr is None:
                    continue
                tw = int(btn.iconSize().width()) if btn.iconSize().width() > 0 else 160
                th = int(btn.iconSize().height()) if btn.iconSize().height() > 0 else 90
                pm = self._np_bgr_to_pixmap(proc_bgr, max_w=tw, max_h=th, fill=True)
                btn.setIcon(QIcon(pm))
            except Exception:
                # Best-effort; keep prior icon.
                pass


    def _np_bgr_to_pixmap(self, img_bgr: np.ndarray, *, max_w: int, max_h: int, fill: bool = False) -> QPixmap:
        if img_bgr is None or img_bgr.size == 0:
            return QPixmap()


        # BGR -> RGB
        rgb = img_bgr[..., ::-1]

        # QImage requires a contiguous buffer (slicing with ::-1 often is not)
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)

        h, w = rgb.shape[:2]

        # Create QImage from contiguous bytes; copy to detach from numpy lifetime safely
        qimg = QImage(rgb.data, w, h, int(rgb.strides[0]), QImage.Format_RGB888).copy()

        pm = QPixmap.fromImage(qimg)
        if max_w > 0 and max_h > 0:
            if fill:
                pm = pm.scaled(max_w, max_h, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                # Center-crop to exact target size.
                if pm.width() != max_w or pm.height() != max_h:
                    x = max(0, (pm.width() - max_w) // 2)
                    y = max(0, (pm.height() - max_h) // 2)
                    pm = pm.copy(x, y, max_w, max_h)
            else:
                pm = pm.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return pm

    def _downsample_bgr_max(self, frame_bgr: np.ndarray, *, max_w: int = 854, max_h: int = 480) -> np.ndarray:
        """Downsample a BGR frame to fit within (max_w,max_h) preserving aspect.

        Used to clamp the compute cost of preset thumbnail generation.
        """
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            return frame_bgr
        if cv2 is None:
            return frame_bgr
        try:
            h, w = frame_bgr.shape[:2]
            if w <= 0 or h <= 0:
                return frame_bgr
            # Scale so both dimensions are <= their respective maxima.
            scale = max(w / float(max_w), h / float(max_h), 1.0)
            if scale <= 1.0:
                return frame_bgr
            nw = max(2, int(w / scale))
            nh = max(2, int(h / scale))
            return cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        except Exception:
            return frame_bgr


    def _ensure_frame_cap(self) -> bool:
        if cv2 is None:
            return False
        p = self.input_path.text().strip()
        if not p:
            return False
        fp = Path(p)
        if not fp.exists():
            return False

        # (Re)open capture when input path changes
        if getattr(self, "_frame_cap_path", None) != str(fp.resolve()):
            try:
                if self._frame_cap is not None:
                    self._frame_cap.release()
            except Exception:
                pass
            self._frame_cap = cv2.VideoCapture(str(fp.resolve()))
            self._frame_cap_path = str(fp.resolve())

        return bool(self._frame_cap is not None and self._frame_cap.isOpened())
    def _ensure_play_scope_caps(self) -> bool:
        """Open cv2.VideoCapture handles for the current preview sources."""
        if cv2 is None:
            return False
        try:
            src_o = self.player_orig.source()
            src_p = self.player_proc.source()
            path_o = src_o.toLocalFile() if hasattr(src_o, "toLocalFile") else ""
            path_p = src_p.toLocalFile() if hasattr(src_p, "toLocalFile") else ""
        except Exception:
            return False

        ok = False
        if path_o:
            if getattr(self, "_play_scope_cap_orig_path", None) != path_o:
                try:
                    if self._play_scope_cap_orig is not None:
                        self._play_scope_cap_orig.release()
                except Exception:
                    pass
                self._play_scope_cap_orig = cv2.VideoCapture(path_o)
                self._play_scope_cap_orig_path = path_o
            ok = ok or (self._play_scope_cap_orig is not None and self._play_scope_cap_orig.isOpened())
        if path_p:
            if getattr(self, "_play_scope_cap_proc_path", None) != path_p:
                try:
                    if self._play_scope_cap_proc is not None:
                        self._play_scope_cap_proc.release()
                except Exception:
                    pass
                self._play_scope_cap_proc = cv2.VideoCapture(path_p)
                self._play_scope_cap_proc_path = path_p
            ok = ok or (self._play_scope_cap_proc is not None and self._play_scope_cap_proc.isOpened())
        return ok

    def _refresh_playback_scopes(self):
        """Render scopes for the current playback/seek position (best effort)."""
        if hasattr(self, "show_scopes") and (not self.show_scopes.isChecked()):
            return
        if not getattr(self, "_preview_available", False):
            return
        if not self._ensure_play_scope_caps():
            return

        rel_ms = int(getattr(self, "_play_scope_last_rel_ms", 0) or 0)

        # Convert rel -> absolute for each stream
        abs_o = self._orig_base_ms() + rel_ms
        abs_p = self._proc_base_ms() + rel_ms

        def grab(cap, abs_ms):
            if cap is None or not cap.isOpened():
                return None
            try:
                cap.set(cv2.CAP_PROP_POS_MSEC, float(abs_ms))
                ok, fr = cap.read()
                return fr if ok else None
            except Exception:
                return None
        
        def _downscale_for_scopes(fr, max_w=105):
            if fr is None:
                return None
            h, w = fr.shape[:2]
            if w <= max_w:
                return fr
            scale = max_w / float(w)
            new_w = max_w
            new_h = max(1, int(h * scale))
            try:
                return cv2.resize(fr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            except Exception:
                return fr

        fr_o = _downscale_for_scopes(grab(self._play_scope_cap_orig, abs_o))
        fr_p = _downscale_for_scopes(grab(self._play_scope_cap_proc, abs_p))

        try:
            if fr_o is not None:
                self.scopes_playback_orig.set_frame_bgr(fr_o)
            if fr_p is not None:
                self.scopes_playback_proc.set_frame_bgr(fr_p)
        except Exception:
            pass
        





    def _on_play_scope_timer_timeout(self):
        """Timer tick for playback scopes (single-shot or repeating)."""
        if not getattr(self, "_preview_available", False):
            return
        # Update last rel pos from current UI/player state.
        try:
            rel = int(self.preview_slider.value()) if hasattr(self, "preview_slider") else 0
        except Exception:
            rel = 0
        self._play_scope_last_rel_ms = max(0, int(rel))
        self._refresh_playback_scopes()

    def _is_preview_playing(self) -> bool:
        try:
            if not self.player_proc.source().isEmpty():
                self._clear_processed_playback_message()

            from PySide6.QtMultimedia import QMediaPlayer
            return self.player_orig.playbackState() == QMediaPlayer.PlayingState
        except Exception:
            return False

    def _trigger_playback_scopes_refresh(self):
        """Refresh scopes once (used when continuous mode is OFF and playback is paused)."""
        if not getattr(self, "_preview_available", False):
            return
        if getattr(self, "_play_scopes_continuous_enabled", False):
            return
        # Only refresh while paused/stopped to match UI contract.
        if self._is_preview_playing():
            return
        try:
            self._play_scope_timer.setSingleShot(True)
            # Restart the single-shot debounce.
            if self._play_scope_timer.isActive():
                self._play_scope_timer.stop()
            self._play_scope_timer.start()
        except Exception:
            pass

    def _maybe_start_continuous_playback_scopes(self):
        """Start repeating scopes updates during playback (used when continuous mode is ON)."""
        if not getattr(self, "_preview_available", False):
            return
        if not getattr(self, "_play_scopes_continuous_enabled", False):
            return
        # Only run in playback tab to avoid background CPU.
        try:
            tabs = getattr(self, "preview_tabs", None)
            if tabs is not None and tabs.currentWidget() is not getattr(self, "tab_playback", None):
                return
        except Exception:
            return
        if not self._is_preview_playing():
            return
        try:
            self._play_scope_timer.setSingleShot(False)
            if not self._play_scope_timer.isActive():
                self._play_scope_timer.start()
        except Exception:
            pass

    def _on_play_scopes_mode_changed(self, state: int):
        """Checkbox handler: ON = continuous scopes; OFF = scopes refresh only when paused."""
        self._play_scopes_continuous_enabled = bool(state)

        if not getattr(self, "_preview_available", False):
            return

        try:
            # Stop any existing timer activity first.
            if self._play_scope_timer.isActive():
                self._play_scope_timer.stop()
        except Exception:
            pass

        if self._play_scopes_continuous_enabled:
            # Continuous mode: repeating timer while playing.
            try:
                self._play_scope_timer.setSingleShot(False)
                self._play_scope_timer.setInterval(750)

            except Exception:
                pass
            self._maybe_start_continuous_playback_scopes()
        else:
            # Non-continuous mode: single-shot refreshes only while paused.
            try:
                self._play_scope_timer.setSingleShot(True)
            except Exception:
                pass
            # If already paused, refresh once immediately.
            self._trigger_playback_scopes_refresh()
    def _load_backend_module(self):
        if self._backend_mod is not None:
            return self._backend_mod
        try:
            import importlib.util
            backend_path = resource_path("app_backend_segment_speedflags.py")
            spec = importlib.util.spec_from_file_location("cc_backend", backend_path)
            if spec is None or spec.loader is None:
                raise RuntimeError("Failed to load backend spec")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._backend_mod = mod
            return mod
        except Exception as e:
            self.append_log(f"[Frame scrub] Failed to load backend module: {e}")
            return None

    def _apply_gui_params_to_backend_globals(self, mod) -> None:
        # Core tuning
        mod.THRESHOLD_RATIO = int(self.threshold_ratio.value())
        mod.MIN_AVG_RED = int(self.min_avg_red.value())
        mod.MAX_HUE_SHIFT = int(self.max_hue_shift.value())
        mod.BLUE_MAGIC_VALUE = float(self.blue_magic_value.value())
        mod.SAMPLE_SECONDS = float(self.sample_seconds.value())
        if hasattr(mod, "SAMPLE_WINDOW_SAMPLES") and hasattr(self, "sample_window_samples"):
            mod.SAMPLE_WINDOW_SAMPLES = int(self.sample_window_samples.value())
        # Contrast / tone mapping
        mod.clip_hist_percent_in = float(self.clip_hist_percent_in.value())
        mod.shadow_amount_percent = float(self.shadow_amount_percent.value())
        mod.shadow_tone_percent = float(self.shadow_tone_percent.value())
        mod.shadow_radius = int(self.shadow_radius.value())
        mod.highlight_amount_percent = float(self.highlight_amount_percent.value())
        mod.highlight_tone_percent = float(self.highlight_tone_percent.value())
        mod.highlight_radius = int(self.highlight_radius.value())

        # Performance
        mod.USE_FAST_HS = bool(self.fast_hs.isChecked())
        mod.FAST_HS_MAP_SCALE = float(self.fast_hs_map_scale.value())
        mod.DISABLE_AUTO_CONTRAST = bool(self.disable_auto_contrast.isChecked())

    # -----------------------------
    # Preset thumbnails (Frame scrub)
    # -----------------------------
    def _build_preset_definitions(self) -> dict:
        """Define named preset parameter bundles.

        These are applied by setting GUI fields, which in turn drives downstream
        colour correction (single-frame preview, playback preview and batch).
        """
        return {
            "Neutral": {
                # Keep defaults / current neutral-ish settings
                "threshold_ratio": 2000,
                "min_avg_red": 45,
                "max_hue_shift": 100,
                "blue_magic_value": 1.20,
                "sample_seconds": 2.0,
                "sample_window_samples": 5,
                "clip_hist_percent_in": 0.30,
                "shadow_amount_percent": 0.70,
                "shadow_tone_percent": 1.00,
                "shadow_radius": 0,
                "highlight_amount_percent": 0.05,
                "highlight_tone_percent": 0.05,
                "highlight_radius": 0,
                "disable_auto_contrast": False,
                "fast_hs": False,
                "fast_hs_map_scale": 0.50,
            },
            "No auto contrast/brightness": {
                # Same base as Neutral, but disables the auto brightness/contrast stage
                # for maximum stability.
                "threshold_ratio": 2000,
                "min_avg_red": 45,
                "max_hue_shift": 100,
                "blue_magic_value": 1.20,
                "sample_seconds": 2.0,
                "sample_window_samples": 5,
                "clip_hist_percent_in": 0.30,
                "shadow_amount_percent": 0.70,
                "shadow_tone_percent": 1.00,
                "shadow_radius": 0,
                "highlight_amount_percent": 0.05,
                "highlight_tone_percent": 0.05,
                "highlight_radius": 0,
                "disable_auto_contrast": True,
                "fast_hs": False,
                "fast_hs_map_scale": 0.50,
            },
            "Vibrant": {
                # More pop: slightly stronger contrast shaping and slightly punchier blues.
                "threshold_ratio": 2000,
                "min_avg_red": 42,
                "max_hue_shift": 160,
                "blue_magic_value": 0.95,
                "sample_seconds": 2.0,
                "sample_window_samples": 5,
                "clip_hist_percent_in": 0.25,
                "shadow_amount_percent": 0.60,
                "shadow_tone_percent": 0.95,
                "shadow_radius": 0,
                "highlight_amount_percent": 0.18,
                "highlight_tone_percent": 0.18,
                "highlight_radius": 0,
                "disable_auto_contrast": False,
                "fast_hs": True,
                "fast_hs_map_scale": 0.50,
            },
            "Low light (lift)": {
                # Designed for darker scenes: more shadow lift, gentler highlights.
                "threshold_ratio": 2200,
                "min_avg_red": 48,
                "max_hue_shift": 120,
                "blue_magic_value": 1.10,
                "sample_seconds": 3.0,
                "sample_window_samples": 5,
                "clip_hist_percent_in": 0.35,
                "shadow_amount_percent": 0.80,
                "shadow_tone_percent": 0.95,
                "shadow_radius": 0,
                "highlight_amount_percent": 0.06,
                "highlight_tone_percent": 0.06,
                "highlight_radius": 0,
                "disable_auto_contrast": False,
                "fast_hs": True,
                "fast_hs_map_scale": 0.50,
            },
            "Blue water": {
                "threshold_ratio": 2000,
                "min_avg_red": 40,
                "max_hue_shift": 200,
                "blue_magic_value": 0.90,
                "sample_seconds": 2.0,
                "sample_window_samples": 5,
                "clip_hist_percent_in": 0.30,
                "shadow_amount_percent": 0.60,
                "shadow_tone_percent": 1.00,
                "shadow_radius": 0,
                "highlight_amount_percent": 0.20,
                "highlight_tone_percent": 0.20,
                "highlight_radius": 0,
                "disable_auto_contrast": False,
                "fast_hs": True,
                "fast_hs_map_scale": 0.50,
            },
            "Green water": {
                # Counter green/cyan casts typical in shallow water or algae-heavy scenes.
                "threshold_ratio": 2000,
                "min_avg_red": 44,
                "max_hue_shift": 190,
                "blue_magic_value": 1.05,
                "sample_seconds": 2.0,
                "sample_window_samples": 5,
                "clip_hist_percent_in": 0.28,
                "shadow_amount_percent": 0.60,
                "shadow_tone_percent": 1.00,
                "shadow_radius": 0,
                "highlight_amount_percent": 0.22,
                "highlight_tone_percent": 0.22,
                "highlight_radius": 0,
                "disable_auto_contrast": False,
                "fast_hs": True,
                "fast_hs_map_scale": 0.50,
            },
            "Clarity": {
                # Adds perceived clarity via a stronger, but still controlled, contrast curve.
                "threshold_ratio": 1900,
                "min_avg_red": 43,
                "max_hue_shift": 140,
                "blue_magic_value": 1.10,
                "sample_seconds": 2.0,
                "sample_window_samples": 5,
                "clip_hist_percent_in": 0.18,
                "shadow_amount_percent": 0.55,
                "shadow_tone_percent": 0.95,
                "shadow_radius": 0,
                "highlight_amount_percent": 0.28,
                "highlight_tone_percent": 0.28,
                "highlight_radius": 0,
                "disable_auto_contrast": False,
                "fast_hs": True,
                "fast_hs_map_scale": 0.50,
            },
            "Warm lift": {
                "threshold_ratio": 1800,
                "min_avg_red": 55,
                "max_hue_shift": 90,
                "blue_magic_value": 1.10,
                "sample_seconds": 2.0,
                "sample_window_samples": 5,
                "clip_hist_percent_in": 0.20,
                "shadow_amount_percent": 0.55,
                "shadow_tone_percent": 0.85,
                "shadow_radius": 0,
                "highlight_amount_percent": 0.08,
                "highlight_tone_percent": 0.08,
                "highlight_radius": 0,
                "disable_auto_contrast": False,
                "fast_hs": True,
                "fast_hs_map_scale": 0.50,
            },
            "More blue": {
                # Bias towards deeper/cleaner blues (lower BLUE_MAGIC_VALUE).
                "threshold_ratio": 2100,
                "min_avg_red": 38,
                "max_hue_shift": 220,
                "blue_magic_value": 1.4,
                "sample_seconds": 2.0,
                "sample_window_samples": 5,
                "clip_hist_percent_in": 0.35,
                "shadow_amount_percent": 0.62,
                "shadow_tone_percent": 1.00,
                "shadow_radius": 0,
                "highlight_amount_percent": 0.18,
                "highlight_tone_percent": 0.18,
                "highlight_radius": 0,
                "disable_auto_contrast": False,
                "fast_hs": True,
                "fast_hs_map_scale": 0.50,
            },
            "Stable": {
                "threshold_ratio": 2200,
                "min_avg_red": 45,
                "max_hue_shift": 110,
                "blue_magic_value": 1.15,
                "sample_seconds": 3.0,
                "sample_window_samples": 31,
                "clip_hist_percent_in": 0.35,
                "shadow_amount_percent": 0.65,
                "shadow_tone_percent": 1.00,
                "shadow_radius": 0,
                "highlight_amount_percent": 0.10,
                "highlight_tone_percent": 0.10,
                "highlight_radius": 0,
                "disable_auto_contrast": False,
                "fast_hs": True,
                "fast_hs_map_scale": 0.50,
            },
            "More blue (stable)": {
                # Deeper blues + reduced pumping: disable auto-contrast and smooth sampling.
                "threshold_ratio": 2100,
                "min_avg_red": 38,
                "max_hue_shift": 220,
                "blue_magic_value": 1.4,
                "sample_seconds": 3.0,
                "sample_window_samples": 31,
                "clip_hist_percent_in": 0.35,
                "shadow_amount_percent": 0.62,
                "shadow_tone_percent": 1.00,
                "shadow_radius": 0,
                "highlight_amount_percent": 0.18,
                "highlight_tone_percent": 0.18,
                "highlight_radius": 0,
                "disable_auto_contrast": False,
                "fast_hs": True,
                "fast_hs_map_scale": 0.50,
            },
        }

    def _apply_preset_to_widgets(self, preset_name: str, *, schedule_render: bool = True) -> None:
        defs = getattr(self, "_preset_defs", None) or {}
        p = defs.get(preset_name)
        if not p:
            return

        self._preset_active_name = preset_name

        # Apply values into the same GUI controls used for batch correction.
        self._frame_param_guard = True
        try:
            if hasattr(self, "threshold_ratio"):
                self.threshold_ratio.setValue(float(p.get("threshold_ratio", self.threshold_ratio.value())))
            if hasattr(self, "min_avg_red"):
                self.min_avg_red.setValue(float(p.get("min_avg_red", self.min_avg_red.value())))
            if hasattr(self, "max_hue_shift"):
                self.max_hue_shift.setValue(float(p.get("max_hue_shift", self.max_hue_shift.value())))
            if hasattr(self, "blue_magic_value"):
                self.blue_magic_value.setValue(float(p.get("blue_magic_value", self.blue_magic_value.value())))
            if hasattr(self, "sample_seconds"):
                self.sample_seconds.setValue(float(p.get("sample_seconds", self.sample_seconds.value())))
            if hasattr(self, "sample_window_samples"):
                self.sample_window_samples.setValue(int(p.get("sample_window_samples", self.sample_window_samples.value())))

            if hasattr(self, "clip_hist_percent_in"):
                self.clip_hist_percent_in.setValue(float(p.get("clip_hist_percent_in", self.clip_hist_percent_in.value())))

            if hasattr(self, "shadow_amount_percent"):
                self.shadow_amount_percent.setValue(float(p.get("shadow_amount_percent", self.shadow_amount_percent.value())))
            if hasattr(self, "shadow_tone_percent"):
                self.shadow_tone_percent.setValue(float(p.get("shadow_tone_percent", self.shadow_tone_percent.value())))
            if hasattr(self, "shadow_radius"):
                self.shadow_radius.setValue(int(p.get("shadow_radius", self.shadow_radius.value())))

            if hasattr(self, "highlight_amount_percent"):
                self.highlight_amount_percent.setValue(float(p.get("highlight_amount_percent", self.highlight_amount_percent.value())))
            if hasattr(self, "highlight_tone_percent"):
                self.highlight_tone_percent.setValue(float(p.get("highlight_tone_percent", self.highlight_tone_percent.value())))
            if hasattr(self, "highlight_radius"):
                self.highlight_radius.setValue(int(p.get("highlight_radius", self.highlight_radius.value())))

            if hasattr(self, "disable_auto_contrast"):
                self.disable_auto_contrast.setChecked(bool(p.get("disable_auto_contrast", self.disable_auto_contrast.isChecked())))

            if hasattr(self, "fast_hs"):
                self.fast_hs.setChecked(bool(p.get("fast_hs", self.fast_hs.isChecked())))
            if hasattr(self, "fast_hs_map_scale"):
                self.fast_hs_map_scale.setValue(float(p.get("fast_hs_map_scale", self.fast_hs_map_scale.value())))
        finally:
            self._frame_param_guard = False

        # Optionally refresh the single-frame preview. When presets are clicked we
        # prefer cached frames to avoid regenerating all thumbnails.
        if schedule_render:
            try:
                self._schedule_frame_render(int(getattr(self, "_frame_last_rel_ms", 0) or 0))
            except Exception:
                pass

    def _on_preset_button_clicked(self, preset_name: str) -> None:
        """Apply preset params and update the frame preview using cached thumbnail frames when available."""
        # Stop any pending frame renders; clicking a preset should be instantaneous.
        try:
            if getattr(self, "_frame_req_timer", None) is not None:
                self._frame_req_timer.stop()
        except Exception:
            pass
        try:
            if getattr(self, "_frame_update_timer", None) is not None:
                self._frame_update_timer.stop()
        except Exception:
            pass

        # Apply preset to widgets, but guard against triggering an async frame regen via valueChanged handlers.
        try:
            self._frame_param_guard = True
            self._apply_preset_to_widgets(preset_name, schedule_render=False)
        finally:
            self._frame_param_guard = False

        # Prefer using the already-rendered thumbnail frames for preview (no re-seek/decode).
        cached = getattr(self, "_preset_thumb_frames", {}).get(preset_name)
        if cached is not None and isinstance(cached, tuple) and len(cached) == 2:
            orig_bgr, proc_bgr = cached
            if orig_bgr is not None and proc_bgr is not None:
                try:
                    self._frame_loaded = True
                except Exception:
                    pass

                try:
                    pm_o = self._np_bgr_to_pixmap(orig_bgr, max_w=self.frame_label_orig.width(), max_h=self.frame_label_orig.height())
                    self.frame_label_orig.setPixmap(pm_o)
                except Exception:
                    pass
                try:
                    pm_p = self._np_bgr_to_pixmap(proc_bgr, max_w=self.frame_label_proc.width(), max_h=self.frame_label_proc.height())
                    self.frame_label_proc.setPixmap(pm_p)
                except Exception:
                    pass

                # Update scopes if enabled
                try:
                    if hasattr(self, "show_scopes_frame") and self.show_scopes_frame.isChecked():
                        if hasattr(self, "scopes_frame_orig"):
                            self.scopes_frame_orig.set_frame_bgr(orig_bgr)
                        if hasattr(self, "scopes_frame_proc"):
                            self.scopes_frame_proc.set_frame_bgr(proc_bgr)
                except Exception:
                    pass
                return

        # Fallback: regenerate using normal pipeline if no cache is available.
        try:
            self._refresh_frame_preview()
        except Exception:
            pass

    def _refresh_preset_thumbnails(self) -> None:
        """Re-render preset thumbnail buttons from the last captured original frame.

        Important: use the cached frame already captured for Frame scrub preview (self._preset_last_frame_bgr)
        instead of re-reading/decoding a frame for each preset.
        """
        base_bgr = getattr(self, "_preset_last_frame_bgr", None)
        if base_bgr is None:
            return
        if not getattr(self, "_preview_available", False):
            return
        if not getattr(self, "_frame_loaded", False):
            return

        mod = self._load_backend_module()
        if mod is None:
            return

        # Optional: compute a stabilised filter RGB estimate once (shared across presets).
        # NOTE: we downsample sampled frames to ~720p before accumulating to keep this cheap.
        filter_rgb_override = None
        try:
            win = int(self.sample_window_samples.value()) if hasattr(self, "sample_window_samples") else 1
        except Exception:
            win = 1

        if win and win > 1 and cv2 is not None and getattr(self, "_frame_cap", None) is not None:
            if win % 2 == 0:
                win += 1
            try:
                cap = self._frame_cap
                rel_ms = int(max(0, getattr(self, "_frame_last_rel_ms", 0)))
                abs_ms = int(self._orig_base_ms()) + rel_ms

                dt_ms = float(self.sample_seconds.value()) * 1000.0 if hasattr(self, "sample_seconds") else 0.0
                if dt_ms <= 0:
                    dt_ms = 1000.0  # fallback 1s

                half = win // 2
                acc = None
                n = 0
                # Save current position so we can restore it
                try:
                    pos_ms_before = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                except Exception:
                    pos_ms_before = float(abs_ms)

                for j in range(-half, half + 1):
                    t_ms = float(abs_ms) + (float(j) * dt_ms)
                    if t_ms < 0:
                        continue
                    try:
                        cap.set(cv2.CAP_PROP_POS_MSEC, t_ms)
                        ok2, fr2 = cap.read()
                    except Exception:
                        ok2, fr2 = False, None

                    if (not ok2) or fr2 is None:
                        continue

                    fr2 = self._downsample_bgr_max(fr2, max_w=1280, max_h=720)
                    rgb2 = fr2[..., ::-1]  # BGR->RGB
                    rgb2 = np.ascontiguousarray(rgb2, dtype=np.float32)
                    if acc is None:
                        acc = rgb2
                    else:
                        acc += rgb2
                    n += 1

                if acc is not None and n > 0:
                    filter_rgb_override = (acc / float(n)).astype(np.uint8)

                # Restore original position for any subsequent reads
                try:
                    cap.set(cv2.CAP_PROP_POS_MSEC, pos_ms_before)
                except Exception:
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(abs_ms))
            except Exception:
                filter_rgb_override = None

        defs = getattr(self, "_preset_defs", None) or {}

        # Clamp preset thumbnail compute cost: downsample once to ~720p before colour correction.
        thumb_base_bgr = self._downsample_bgr_max(base_bgr, max_w=854, max_h=480)


        # Ensure cache dict exists
        if not hasattr(self, "_preset_thumb_frames") or getattr(self, "_preset_thumb_frames", None) is None:
            self._preset_thumb_frames = {}

        for name, btn in list(getattr(self, "_preset_buttons", {}).items()):
            try:
                p = defs.get(name)
                if not p:
                    continue

                # Temporarily apply globals for filter_matrix computation
                # (get_filter_matrix reads MIN_AVG_RED / MAX_HUE_SHIFT / BLUE_MAGIC_VALUE / THRESHOLD_RATIO etc.)
                old = {
                    "THRESHOLD_RATIO": getattr(mod, "THRESHOLD_RATIO", None),
                    "MIN_AVG_RED": getattr(mod, "MIN_AVG_RED", None),
                    "MAX_HUE_SHIFT": getattr(mod, "MAX_HUE_SHIFT", None),
                    "BLUE_MAGIC_VALUE": getattr(mod, "BLUE_MAGIC_VALUE", None),
                    "SAMPLE_SECONDS": getattr(mod, "SAMPLE_SECONDS", None),
                    "SAMPLE_WINDOW_SAMPLES": getattr(mod, "SAMPLE_WINDOW_SAMPLES", None),
                    "clip_hist_percent_in": getattr(mod, "clip_hist_percent_in", None),
                    "shadow_amount_percent": getattr(mod, "shadow_amount_percent", None),
                    "shadow_tone_percent": getattr(mod, "shadow_tone_percent", None),
                    "shadow_radius": getattr(mod, "shadow_radius", None),
                    "highlight_amount_percent": getattr(mod, "highlight_amount_percent", None),
                    "highlight_tone_percent": getattr(mod, "highlight_tone_percent", None),
                    "highlight_radius": getattr(mod, "highlight_radius", None),
                    "USE_FAST_HS": getattr(mod, "USE_FAST_HS", None),
                    "FAST_HS_MAP_SCALE": getattr(mod, "FAST_HS_MAP_SCALE", None),
                }

                mod.THRESHOLD_RATIO = int(p.get("threshold_ratio", old["THRESHOLD_RATIO"] or 2000))
                mod.MIN_AVG_RED = int(p.get("min_avg_red", old["MIN_AVG_RED"] or 45))
                mod.MAX_HUE_SHIFT = int(p.get("max_hue_shift", old["MAX_HUE_SHIFT"] or 100))
                mod.BLUE_MAGIC_VALUE = float(p.get("blue_magic_value", old["BLUE_MAGIC_VALUE"] or 1.2))
                mod.SAMPLE_SECONDS = float(p.get("sample_seconds", old["SAMPLE_SECONDS"] or 2.0))
                if hasattr(mod, "SAMPLE_WINDOW_SAMPLES"):
                    mod.SAMPLE_WINDOW_SAMPLES = int(p.get("sample_window_samples", old["SAMPLE_WINDOW_SAMPLES"] or 15))
                mod.clip_hist_percent_in = float(p.get("clip_hist_percent_in", old["clip_hist_percent_in"] or 0.3))
                mod.shadow_amount_percent = float(p.get("shadow_amount_percent", old["shadow_amount_percent"] or 0.7))
                mod.shadow_tone_percent = float(p.get("shadow_tone_percent", old["shadow_tone_percent"] or 1.0))
                mod.shadow_radius = int(p.get("shadow_radius", old["shadow_radius"] or 0))
                mod.highlight_amount_percent = float(p.get("highlight_amount_percent", old["highlight_amount_percent"] or 0.05))
                mod.highlight_tone_percent = float(p.get("highlight_tone_percent", old["highlight_tone_percent"] or 0.05))
                mod.highlight_radius = int(p.get("highlight_radius", old["highlight_radius"] or 0))
                mod.USE_FAST_HS = bool(p.get("fast_hs", old["USE_FAST_HS"] or False))
                mod.FAST_HS_MAP_SCALE = float(p.get("fast_hs_map_scale", old["FAST_HS_MAP_SCALE"] or 0.5))

                out_bgr = mod.correct_frame_full(
                    thumb_base_bgr,
                    disable_auto_contrast=bool(p.get("disable_auto_contrast", False)),
                    clip_hist_percent=float(p.get("clip_hist_percent_in", 0.3)),
                    use_fast_hs=bool(p.get("fast_hs", False)),
                    fast_hs_map_scale=float(p.get("fast_hs_map_scale", 0.5)),
                    filter_rgb_override=filter_rgb_override,
                )

                # Cache frames for instant preview when this preset is selected
                try:
                    self._preset_thumb_frames[name] = (thumb_base_bgr.copy(), out_bgr.copy())
                except Exception:
                    pass

                tw = int(btn.iconSize().width()) if btn.iconSize().width() > 0 else 160
                th = int(btn.iconSize().height()) if btn.iconSize().height() > 0 else 90
                pm = self._np_bgr_to_pixmap(out_bgr, max_w=tw, max_h=th, fill=True)
                btn.setIcon(QIcon(pm))
            except Exception:
                # Best-effort; leave prior icon.
                pass
            finally:
                # Restore globals to whatever the GUI is currently set to (so preview render stays correct)
                try:
                    self._apply_gui_params_to_backend_globals(mod)
                except Exception:
                    pass

    def _refresh_frame_preview(self):
        # Only refresh if tab is active
        tabs = getattr(self, "preview_tabs", None)
        if tabs is None or getattr(self, "tab_frame", None) is None:
            return
        if tabs.currentWidget() is not self.tab_frame:
            return

        if not self._ensure_frame_cap():
            return

        cap = self._frame_cap
        rel_ms = int(max(0, getattr(self, "_frame_last_rel_ms", 0)))
        abs_ms = int(self._orig_base_ms()) + rel_ms

        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(abs_ms))
            ok, frame_bgr = cap.read()
        except Exception:
            ok, frame_bgr = False, None

        if not ok or frame_bgr is None:
            self.frame_label_orig.setText("(failed to read frame)")
            self.frame_label_proc.setText("(failed to read frame)")
            return
        # Ensure rel_ms cannot exceed current preview window
        if self._preview_window_ms > 0:
            rel_ms = min(rel_ms, int(self._preview_window_ms))

        self._frame_loaded = True
        # Update preset thumbnails using this original frame (ONLY when the base frame changes).
        try:
            self._preset_last_frame_bgr = frame_bgr.copy()
            self._refresh_preset_thumbnails()
        except Exception:
            pass

        # # Render original
        try:
            pm_o = self._np_bgr_to_pixmap(frame_bgr, max_w=self.frame_label_orig.width(), max_h=self.frame_label_orig.height())
            self.frame_label_orig.setPixmap(pm_o)
        except Exception as e:
            self.frame_label_orig.setText(f"(render failed: {e})")
            self.frame_label_proc.setText(f"(render failed: {e})")
            return
        # pm_o = self._np_bgr_to_pixmap(
        #     frame_bgr,
        #     max_w=self.frame_label_orig.width(),
        #     max_h=self.frame_label_orig.height(),
        # )

        # try:
        #     pm_o = self._np_bgr_to_pixmap(frame_bgr, max_w=..., max_h=...)
        #     self.frame_label_orig.setPixmap(pm_o)
        # except Exception as e:
        #     self.frame_label_orig.setText(f"(render failed: {e})")
        #     self.frame_label_proc.setText(f"(render failed: {e})")
        #     return

        # Render processed
        mod = self._load_backend_module()
        if mod is None:
            self.frame_label_proc.setText("(backend module unavailable)")
            return

        try:
            self._frame_param_guard = True
            self._apply_gui_params_to_backend_globals(mod)
        finally:
            self._frame_param_guard = False

        # Compute an RGB mean over a local window of *sampled* frames to stabilise the filter estimate (best effort).
        # This mirrors the backend logic: mean over an odd number of samples centred on the current timepoint.
        # We sample frames at offsets of SAMPLE_SECONDS rather than decoding consecutive frames.
        filter_rgb_override = None
        try:
            win = int(self.sample_window_samples.value()) if hasattr(self, "sample_window_samples") else 1
        except Exception:
            win = 1

        if win and win > 1:
            # Force odd window size
            if win % 2 == 0:
                win += 1

            try:
                dt_ms = float(self.sample_seconds.value()) * 1000.0 if hasattr(self, "sample_seconds") else 0.0
                if dt_ms <= 0:
                    dt_ms = 1000.0  # fallback 1s

                half = win // 2
                acc = None
                n = 0

                # Save current position so we can restore it
                try:
                    pos_ms_before = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                except Exception:
                    pos_ms_before = float(abs_ms)

                for j in range(-half, half + 1):
                    t_ms = float(abs_ms) + (float(j) * dt_ms)
                    if t_ms < 0:
                        continue
                    try:
                        cap.set(cv2.CAP_PROP_POS_MSEC, t_ms)
                        ok2, fr2 = cap.read()
                    except Exception:
                        ok2, fr2 = False, None

                    if (not ok2) or fr2 is None:
                        continue

                    rgb2 = fr2[..., ::-1]  # BGR->RGB
                    rgb2 = np.ascontiguousarray(rgb2, dtype=np.float32)
                    if acc is None:
                        acc = rgb2
                    else:
                        acc += rgb2
                    n += 1

                if acc is not None and n > 0:
                    filter_rgb_override = (acc / float(n)).astype(np.uint8)

                # Restore original position for any subsequent reads
                try:
                    cap.set(cv2.CAP_PROP_POS_MSEC, pos_ms_before)
                except Exception:
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(abs_ms))
            except Exception:
                filter_rgb_override = None

        try:
            corrected = mod.correct_frame_full(
                frame_bgr,
                disable_auto_contrast=bool(self.disable_auto_contrast.isChecked()),
                clip_hist_percent=float(self.clip_hist_percent_in.value()),
                use_fast_hs=bool(self.fast_hs.isChecked()),
                fast_hs_map_scale=float(self.fast_hs_map_scale.value()),
                filter_rgb_override=filter_rgb_override,
            )
        except Exception as e:
            self.frame_label_proc.setText(f"(process failed: {e})")
            return
        try:
            pm_p = self._np_bgr_to_pixmap(corrected, max_w=self.frame_label_proc.width(), max_h=self.frame_label_proc.height())
            self.frame_label_proc.setPixmap(pm_p)
        except Exception as e:
            self.frame_label_orig.setText(f"(render failed: {e})")
            self.frame_label_proc.setText(f"(render failed: {e})")
            return
        # Update scopes
        try:
            self.scopes_frame_orig.set_frame_bgr(frame_bgr)
            self.scopes_frame_proc.set_frame_bgr(corrected)
            if (not hasattr(self, "show_scopes")) or self.show_scopes.isChecked():
                self.scopes_frame_orig.set_frame_bgr(frame_bgr)
                self.scopes_frame_proc.set_frame_bgr(corrected)
        except Exception:
            pass


    # def _on_preview_duration_changed(self, dur_ms: int):
    #     if not self._preview_available:
    #         return

    #     if self.segment_only.isChecked() and self._seg_duration_ms > 0:
    #         self.preview_slider.setRange(0, self._seg_duration_ms)
    #         self._preview_duration_ms = self._seg_duration_ms
    #     else:
    #         self.preview_slider.setRange(0, dur_ms)
    #         self._preview_duration_ms = dur_ms

    #     self._update_preview_time_label(0)


    # def _on_preview_position_changed(self, pos_ms: int):
    #     if not self._preview_available or self._preview_block:
    #         return
    #     self._preview_block = True
    #     try:
    #         self.preview_slider.setValue(int(pos_ms))
    #         self._update_preview_time_label(int(pos_ms))
    #     finally:
    #         self._preview_block = False
    def _on_preview_position_changed(self, pos_ms: int):
        if not self._preview_available or self._preview_block:
            return

        self._preview_block = True
        try:
            # Convert absolute original position -> relative position within preview window
            rel = int(pos_ms) - int(self._orig_base_ms())
            if rel < 0:
                rel = 0
            rel = min(rel, self.preview_slider.maximum())

            self.preview_slider.setValue(rel)
            self._update_preview_time_label(rel)
            # Render scopes during playback/seek without starving the timer:
            # only (re)start if not already active.
            try:
                self._play_scope_last_rel_ms = int(rel)
                if not getattr(self, "_play_scopes_continuous_enabled", False):
                    if not self._play_scope_timer.isActive():
                        self._play_scope_timer.start()
            except Exception:
                pass
        finally:
            self._preview_block = False

    # def _update_preview_time_label(self, pos_ms: int):
    #     total = int(getattr(self, "_preview_window_ms", 0) or 0)
    #     pos_s = max(0, int(pos_ms // 1000))
    #     tot_s = max(0, int(total // 1000))
    #     self.preview_time.setText(f"{self._format_time(pos_s)} / {self._format_time(tot_s)}")
    def _update_preview_time_label(self, rel_ms: int):
        # rel_ms is always relative to the preview window (0..window)
        total = int(getattr(self, "_preview_window_ms", 0) or 0)

        rel_s = max(0, int(rel_ms // 1000))
        tot_s = max(0, int(total // 1000))

        # Absolute original timestamp = base + rel
        abs_orig_ms = int(self._orig_base_ms()) + int(rel_ms)
        abs_orig_s = max(0, int(abs_orig_ms // 1000))

        # Display: rel / total (Orig absolute)
        self.preview_time.setText(
            f"{self._format_time(rel_s)} / {self._format_time(tot_s)}   (Orig {self._format_time(abs_orig_s)})"
        )

    # def preview_seek(self, value_ms: int):
    #     if not self._preview_available or self._preview_block:
    #         return

    #     self._preview_block = True
    #     try:
    #         base = self._seg_start_ms if self.segment_only.isChecked() else 0
    #         abs_pos = base + value_ms
    #         self.player_orig.setPosition(abs_pos)
    #         self.player_proc.setPosition(abs_pos)
    #     finally:
    #         self._preview_block = False
    def preview_seek(self, value_ms: int):
        if not self._preview_available or self._preview_block:
            return

        self._preview_block = True
        try:
            rel = int(value_ms)
            self.player_orig.setPosition(self._orig_base_ms() + rel)
            self.player_proc.setPosition(self._proc_base_ms() + rel)  # always 0 + rel
            # In non-continuous mode, scopes should update only while paused.
            self._trigger_playback_scopes_refresh()
        finally:
            self._preview_block = False


    def preview_play_pause(self):
        if not self._preview_available:
            return
        # Ensure sources are loaded (processed may not be loaded yet)
        if self.player_orig.source().isEmpty():
            self._maybe_load_original_preview()
        if self.player_proc.source().isEmpty():
            self._maybe_load_processed_preview()
        # If processed still has no source, do not try to play it
        if self.player_proc.source().isEmpty():
            self._set_processed_playback_message(
                'Processed file not found at the output path. '
                'Run processing or verify the output path, then click Reload preview.'
            )
            return
        # Clear any previous processed playback error once we have a source.
        if not self.player_proc.source().isEmpty():
            self._clear_processed_playback_message()
        from PySide6.QtMultimedia import QMediaPlayer

        if self.player_orig.playbackState() == QMediaPlayer.PlayingState:
            self.player_orig.pause()
            self.player_proc.pause()
            self._sync_timer.stop()
            # Stop continuous scopes if running, then refresh once if non-continuous.
            try:
                if hasattr(self, "_play_scope_timer") and self._play_scope_timer.isActive():
                    self._play_scope_timer.stop()
            except Exception:
                pass
            self._trigger_playback_scopes_refresh()
            return

        # rel_pos = self.preview_slider.value()
        rel_pos = int(self.preview_slider.value())
        base = self._seg_start_ms if self.segment_only.isChecked() else 0
        abs_pos = base + rel_pos

        self.player_orig.setPosition(self._orig_base_ms() + rel_pos)
        self.player_proc.setPosition(self._proc_base_ms() + rel_pos)

        self.player_orig.play()
        if not self.player_proc.source().isEmpty():
            self.player_proc.play()
        self._sync_timer.start()
        # Scopes refresh policy:
        # - Continuous scopes ON: refresh periodically during playback.
        # - Continuous scopes OFF: do not refresh during playback (only while paused).
        try:
            if getattr(self, "_play_scopes_continuous_enabled", False):
                self._maybe_start_continuous_playback_scopes()
            else:
                if self._play_scope_timer.isActive():
                    self._play_scope_timer.stop()
        except Exception:
            pass


    def preview_stop(self):
        if not self._preview_available:
            return
        self.player_orig.stop()
        self.player_proc.stop()
        self._sync_timer.stop()
        try:
            if hasattr(self, "_play_scope_timer") and self._play_scope_timer.isActive():
                self._play_scope_timer.stop()
        except Exception:
            pass
        try:
            self.scopes_playback_orig.clear()
            self.scopes_playback_proc.clear()
        except Exception:
            pass
        self.preview_slider.setValue(0)
        self._update_preview_time_label(0)

    
    def preview_reload(self):
        if not self._preview_available:
            return

        # Stop playback first (clean reset)
        self._sync_timer.stop()
        try:
            self.player_orig.stop()
            self.player_proc.stop()
        except Exception:
            pass

        # Re-load sources from current paths
        self._maybe_load_original_preview()
        ok_proc = self._maybe_load_processed_preview(force=True)

        if ok_proc:
            self._clear_processed_playback_message()

        if not ok_proc:
            self._set_processed_playback_message(
                'Processed file does not exist at the output path yet. '
                'Run processing first, or check the output path.'
            )

        # Re-apply segment window to slider range if segment-only is enabled
        if self.segment_only.isChecked() and self._seg_duration_ms > 0:
            self._preview_duration_ms = self._seg_duration_ms
            self.preview_slider.setRange(0, self._seg_duration_ms)
            self.preview_slider.setValue(0)
            self._update_preview_time_label(0)

            # # Ensure players start at segment start
            # self.player_orig.setPosition(self._seg_start_ms)
            # self.player_proc.setPosition(self._seg_start_ms)
            # Ensure players start correctly after reload
            self.player_orig.setPosition(self._orig_base_ms())
            self.player_proc.setPosition(self._proc_base_ms())
        else:
            # If not segment-only, duration will be updated via durationChanged
            self.preview_slider.setValue(0)
            self._update_preview_time_label(0)
            self.player_orig.setPosition(0)
            self.player_proc.setPosition(0)
        # Also reload Frame scrub state after preview reload.
        # 1) Recompute window (this updates frame slider range too).
        try:
            self._recompute_preview_window()
        except Exception:
            pass

        # 2) Force frame scrub to reopen capture (in case input path changed).
        try:
            if getattr(self, "_frame_cap", None) is not None:
                try:
                    self._frame_cap.release()
                except Exception:
                    pass
            self._frame_cap = None
            self._frame_cap_path = None
        except Exception:
            pass

        # 3) Reset to start and render a fresh frame.
        if hasattr(self, "frame_slider") and self.frame_slider is not None:
            try:
                self.frame_slider.setValue(0)
                self._frame_last_rel_ms = 0
                self._update_frame_time_label(0)
                self._refresh_frame_preview()
            except Exception:
                pass

    # def _sync_preview_players(self):
    #     if not self._preview_available:
    #         return

    #     from PySide6.QtMultimedia import QMediaPlayer

    #     if self.player_orig.playbackState() != QMediaPlayer.PlayingState:
    #         return

    #     pos = self.player_orig.position()

    #     if self.segment_only.isChecked():
    #         if pos >= self._seg_start_ms + self._seg_duration_ms:
    #             self.preview_stop()
    #             return

    #     # Keep players aligned
    #     delta = abs(pos - self.player_proc.position())
    #     if delta > 120:
    #         self.player_proc.setPosition(pos)
    def _sync_preview_players(self):
        if not self._preview_available:
            return

        # Clear any previous processed playback error once we have a source.
        if not self.player_proc.source().isEmpty():
            self._clear_processed_playback_message()

        from PySide6.QtMultimedia import QMediaPlayer

        if self.player_orig.playbackState() != QMediaPlayer.PlayingState:
            return

        o_abs = int(self.player_orig.position())
        rel = o_abs - int(self._orig_base_ms())
        if rel < 0:
            rel = 0

        # Stop at end of preview window if segment-only
        # if self.segment_only.isChecked() and self._seg_duration_ms > 0:
        #     if rel >= self._seg_duration_ms:
        #         self.preview_stop()
        #         return
        # if self._preview_window_ms > 0 and rel >= self._preview_window_ms:
        #     self.preview_stop()
        #     return
        pass
        p_target = int(self._proc_base_ms()) + int(rel)  # base=0 + rel
        p_abs = int(self.player_proc.position())

        if abs(p_abs - p_target) > 40:
            self.player_proc.setPosition(p_target)


    # ---------- Args / job ----------
    def build_args(self, mode: str, inp: Path, out: Path) -> list:
        script = self.script_path.text().strip()
        if not script or not Path(script).exists():
            raise ValueError("Script path is invalid.")

        # IMPORTANT: when launching via sys.executable, the first argument must be the script to run.
        # We call the backend script directly (not the GUI in --backend mode).
        args = [
            "--backend",
            mode,
            str(inp),
            str(out),
            "--threshold-ratio", str(self.threshold_ratio.value()),
            "--min-avg-red", str(self.min_avg_red.value()),
            "--max-hue-shift", str(self.max_hue_shift.value()),
            "--blue-magic-value", str(self.blue_magic_value.value()),
            "--sample-seconds", str(self.sample_seconds.value()),
            "--sample-window-samples", str(int(self.sample_window_samples.value())),
            "--clip-hist-percent-in", str(self.clip_hist_percent_in.value()),
            "--shadow-amount-percent", str(self.shadow_amount_percent.value()),
            "--shadow-tone-percent", str(self.shadow_tone_percent.value()),
            "--shadow-radius", str(self.shadow_radius.value()),
            "--highlight-amount-percent", str(self.highlight_amount_percent.value()),
            "--highlight-tone-percent", str(self.highlight_tone_percent.value()),
            "--highlight-radius", str(self.highlight_radius.value()),
        ]

        # Performance flags
        args += ["--auto-contrast-every-n-frames", str(self.auto_contrast_every_n_frames.value())]
        if not self.precompute_filters.isChecked():
            args += ["--no-precompute-filters"]
        if self.disable_auto_contrast.isChecked():
            args += ["--disable-auto-contrast"]
        if self.fast_hs.isChecked():
            args += ["--fast-hs"]
            args += ["--fast-hs-map-scale", f"{self.fast_hs_map_scale.value():.2f}"]

        # Single-tab only: optional video downsample (affects both processing and preview output)
        if mode == "video":
            try:
                ds = int(self.downsample_factor.currentText().split()[0])
            except Exception:
                ds = 1
            ds = max(1, ds)
            args += ["--downsample", str(ds)]
        # Optional: process only a selected segment (video mode only)
        # if (
        #     mode == "video"
        #     and self.segment_only.isEnabled()
        #     and self.segment_only.isChecked()
        #     and getattr(self, "_video_duration_sec", 0) > 0
        # ):
        #     start_sec = int(self.seg_start_slider.value())
        #     end_sec = int(self.seg_end_slider.value())
        #     if end_sec > start_sec:
        #         args += ["--start-sec", str(start_sec), "--duration-sec", str(end_sec - start_sec)]

        # return args
        if mode == "video" and self.segment_only.isEnabled() and self.segment_only.isChecked() and self._video_duration_sec > 0:
            start_sec = int(self.seg_start_slider.value())
            end_sec = int(self.seg_end_slider.value())
            if end_sec > start_sec:
                args += ["--start-sec", str(start_sec), "--duration-sec", str(end_sec - start_sec)]
        return args

    def start_job(self):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            QMessageBox.information(self, "Job running", "A job is already running. Stop it before starting a new one.")
            return

        inp_txt = self.input_path.text().strip()
        if not inp_txt:
            QMessageBox.warning(self, "Input required", "Please select an input file.")
            return

        inp = Path(inp_txt)
        if not inp.exists():
            QMessageBox.warning(self, "Invalid input", "Input path does not exist.")
            return

        if self.same_folder.isChecked():
            out_txt = self.default_output_for(inp)
            self.output_path.setText(out_txt)
        else:
            out_txt = self.output_path.text().strip()

        if not out_txt:
            QMessageBox.warning(self, "Output required", "Please select an output file (or enable same-folder output).")
            return

        out = Path(out_txt)
        out.parent.mkdir(parents=True, exist_ok=True)

        ext = inp.suffix.lower()
        mode = "video" if ext in {".mp4", ".mov", ".mkv", ".avi"} else "image"

        try:
            args = self.build_args(mode, inp, out)
        except Exception as e:
            QMessageBox.critical(self, "Argument error", str(e))
            return

        self.stage_pct = {"ANALYZE": 0.0, "PROCESS": 0.0}
        self.progress.setValue(0)

        self.proc = QProcess(self)
        self.proc.setProgram(sys.executable)
        self.proc.setArguments(args)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self.on_stdout)
        self.proc.finished.connect(self.on_finished)

        cmdline = f"{sys.executable} " + " ".join(shlex.quote(str(a)) for a in args)
        self.append_log(f"START {cmdline}")

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.proc.start()
        if not self.proc.waitForStarted(3000):
            self.append_log("Failed to start process.")
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.proc = None

    def stop_job(self):
        if not self.proc:
            return
        if self.proc.state() != QProcess.NotRunning:
            self.append_log("Stopping job...")
            self.proc.kill()

    def _update_progress(self, stage: str, pct: float):
        stage = stage.upper()
        pct = max(0.0, min(100.0, float(pct)))
        if stage in self.stage_pct:
            self.stage_pct[stage] = pct
        overall = 0.2 * self.stage_pct.get("ANALYZE", 0.0) + 0.8 * self.stage_pct.get("PROCESS", 0.0)
        self.progress.setValue(int(round(overall)))

    def on_stdout(self):
        if not self.proc:
            return
        data = self.proc.readAllStandardOutput().data().decode(errors="replace")
        if not data:
            return
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("PROGRESS "):
                parts = line.split()
                if len(parts) == 3:
                    stage = parts[1]
                    try:
                        pct = float(parts[2])
                    except ValueError:
                        continue
                    self._update_progress(stage, pct)
            else:
                self.append_log(line)

    def on_finished(self, exit_code: int, _exit_status):
        if exit_code == 0:
            self.progress.setValue(100)
            self.append_log(f"DONE (exit {exit_code})")
        else:
            self.append_log(f"FAILED (exit {exit_code})")

        # Refresh processed preview if initialized
        self._maybe_load_processed_preview(force=True)

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.proc = None
        



class BatchProcessorTab(QWidget):
    """
    Batch, parallel processing of full videos.
    Uses the tuning/performance settings from the provided BlueCorrectorSingleGUI instance,
    but ALWAYS processes full videos (no segment trimming).
    """
    VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}
    def _refresh_table(self):
        # Forces immediate repaint of table widgets (progress bars, status)
        self.table.viewport().update()

    def __init__(self, single: "BlueCorrectorSingleGUI"):
        super().__init__()
        self.single = single
        self.setObjectName("batch_tab")

        self._jobs = []  # list[dict]
        self._active = {}  # row -> QProcess

        root = QVBoxLayout(self)

        # --- Controls row ---
        ctl = QHBoxLayout()

        self.btn_add_files = QPushButton("Add files…")
        self.btn_add_folder = QPushButton("Add folder…")
        self.btn_clear = QPushButton("Clear list")

        self.btn_add_files.clicked.connect(self.add_files)
        self.btn_add_folder.clicked.connect(self.add_folder)
        self.btn_clear.clicked.connect(self.clear_jobs)

        self.out_dir = QLineEdit()
        self.out_dir.setPlaceholderText("Output folder for corrected videos")
        self.out_dir.textChanged.connect(self._refresh_output_paths)
        self.btn_out_dir = QPushButton("Browse…")
        self.btn_out_dir.clicked.connect(self.browse_out_dir)

        self.max_parallel = QSpinBox()
        self.max_parallel.setRange(1, 64)
        self.max_parallel.setValue(4)

        # Batch downsample (processing) - reduces resolution to speed up batch processing.
        # This is independent from the Single/Segment downsample.
        self.batch_downsample = QComboBox()
        self.batch_downsample.addItems([
            "1 (full)",
            "2 (half)",
            "4 (quarter)",
            "8 (eighth)",
            "16 (sixteenth)",
            "32 (1/32)",
        ])
        self.batch_downsample.setCurrentIndex(0)
        self.batch_downsample.setToolTip(
            "Downsample factor for BATCH processing. Higher values are faster but reduce output resolution."
        )

        self.btn_start = QPushButton("Start batch")
        self.btn_stop = QPushButton("Stop all")
        self.btn_stop.setEnabled(False)

        self.btn_start.clicked.connect(self.start_batch)
        self.btn_stop.clicked.connect(self.stop_all)

        ctl.addWidget(self.btn_add_files)
        ctl.addWidget(self.btn_add_folder)
        ctl.addWidget(self.btn_clear)
        ctl.addSpacing(12)
        ctl.addWidget(QLabel("Output folder:"))
        ctl.addWidget(self.out_dir, 1)
        ctl.addWidget(self.btn_out_dir)
        ctl.addSpacing(12)
        ctl.addWidget(QLabel("Max parallel:"))
        ctl.addWidget(self.max_parallel)
        ctl.addSpacing(12)
        ctl.addWidget(QLabel("Downsample:"))
        ctl.addWidget(self.batch_downsample)
        ctl.addSpacing(12)
        ctl.addWidget(self.btn_start)
        ctl.addWidget(self.btn_stop)

        root.addLayout(ctl)

        # --- Overall progress ---
        overall_row = QHBoxLayout()
        self.batch_progress = QProgressBar()
        self.batch_progress.setRange(0, 100)
        self.batch_progress.setValue(0)
        self.batch_progress.setFormat("%p%")
        self.batch_status = QLabel("No jobs.")
        overall_row.addWidget(QLabel("Batch progress:"))
        overall_row.addWidget(self.batch_progress, 1)
        overall_row.addWidget(self.batch_status)
        root.addLayout(overall_row)

        # --- Jobs table ---
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Status", "Progress", "Input", "Output"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        root.addWidget(self.table, 2)

        # --- Log ---
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        root.addWidget(self.log, 1)

    # ---------- UI helpers ----------
    def append_log(self, text: str) -> None:
        self.log.append(text.rstrip())

    def browse_out_dir(self):
        start = str(Path(self.out_dir.text()).expanduser()) if self.out_dir.text().strip() else str(Path.home())
        p = QFileDialog.getExistingDirectory(self, "Select output folder", start)
        if p:
            self.out_dir.setText(p)

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select video files",
            str(Path.home()),
            "Media (*.mp4 *.mov *.mkv *.avi);;All (*.*)"
        )
        if not files:
            return
        self._add_inputs([Path(f) for f in files])

    def add_folder(self):
        p = QFileDialog.getExistingDirectory(self, "Select folder containing videos", str(Path.home()))
        if not p:
            return
        folder = Path(p)
        inputs = []
        for ext in sorted(self.VIDEO_EXTS):
            inputs.extend(folder.glob(f"*{ext}"))
        self._add_inputs(inputs)

    def clear_jobs(self):
        self.stop_all()
        self._jobs.clear()
        self.table.setRowCount(0)
        self.batch_progress.setValue(0)
        self.batch_status.setText("No jobs.")
        self.append_log("Cleared job list.")

    def _default_output_path(self, inp: Path) -> Path:
        out_dir_txt = self.out_dir.text().strip()
        out_dir = Path(out_dir_txt) if out_dir_txt else inp.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{inp.stem}.corrected{inp.suffix}"

    def _refresh_output_paths(self):
        """Refresh the Output column in the batch table when the output folder changes."""
        out_dir_txt = self.out_dir.text().strip()
        out_dir = Path(out_dir_txt) if out_dir_txt else None
        for job in self._jobs:
            # Do not rewrite output path for running jobs
            proc = job.get("proc")
            if proc is not None and proc.state() != QProcess.NotRunning:
                continue
            inp = job["inp"]
            target_dir = out_dir if out_dir is not None else inp.parent
            new_out = target_dir / f"{inp.stem}.corrected{inp.suffix}"
            job["out"] = new_out
            self.table.setItem(job["row"], 3, QTableWidgetItem(str(new_out)))

    def _add_inputs(self, inputs: list[Path]):
        added = 0
        for inp in inputs:
            if not inp.exists() or not inp.is_file():
                continue
            if inp.suffix.lower() not in self.VIDEO_EXTS:
                continue

            outp = self._default_output_path(inp)

            # De-dupe by input path
            if any(j["inp"] == inp for j in self._jobs):
                continue

            row = self.table.rowCount()
            self.table.insertRow(row)

            status_item = QTableWidgetItem("Queued")
            self.table.setItem(row, 0, status_item)

            pb = QProgressBar()
            pb.setRange(0, 100)
            pb.setValue(0)
            pb.setFormat("%p%")
            self.table.setCellWidget(row, 1, pb)

            self.table.setItem(row, 2, QTableWidgetItem(str(inp)))
            self.table.setItem(row, 3, QTableWidgetItem(str(outp)))

            self._jobs.append({
                "row": row,
                "inp": inp,
                "out": outp,
                "proc": None,
                "stage_pct": {"ANALYZE": 0.0, "PROCESS": 0.0},
                "status": "Queued",
            })
            added += 1

        if added:
            self.append_log(f"Added {added} video(s) to batch.")
            self._update_batch_status()

    # ---------- Process management ----------
    def _build_video_args_full(self, inp: Path, out: Path) -> list:
        """
        Copy of BlueCorrectorSingleGUI.build_args(), but intentionally excludes
        any segment-related arguments. Batch always runs full videos.
        """
        script = self.single.script_path.text().strip()
        if not script or not Path(script).exists():
            raise ValueError("Script path is invalid.")

        args = [
            "--backend",
            "video",
            str(inp),
            str(out),
            "--threshold-ratio", str(self.single.threshold_ratio.value()),
            "--min-avg-red", str(self.single.min_avg_red.value()),
            "--max-hue-shift", str(self.single.max_hue_shift.value()),
            "--blue-magic-value", str(self.single.blue_magic_value.value()),
            "--sample-seconds", str(self.single.sample_seconds.value()),
            "--sample-window-samples", str(int(self.single.sample_window_samples.value())),
            "--clip-hist-percent-in", str(self.single.clip_hist_percent_in.value()),
            "--shadow-amount-percent", str(self.single.shadow_amount_percent.value()),
            "--shadow-tone-percent", str(self.single.shadow_tone_percent.value()),
            "--shadow-radius", str(self.single.shadow_radius.value()),
            "--highlight-amount-percent", str(self.single.highlight_amount_percent.value()),
            "--highlight-tone-percent", str(self.single.highlight_tone_percent.value()),
            "--highlight-radius", str(self.single.highlight_radius.value()),
        ]

        if self.single.fast_hs.isChecked():
            args += ["--fast-hs"]
            args += ["--fast-hs-map-scale", f"{self.single.fast_hs_map_scale.value():.2f}"]

        # Performance flags
        args += ["--auto-contrast-every-n-frames", str(self.single.auto_contrast_every_n_frames.value())]
        # Temporal stabilisation flags (shared with Single tab)
        args += ["--filter-smooth-alpha", str(self.single.filter_smooth_alpha.value())]
        args += ["--filter-max-delta", str(self.single.filter_max_delta.value())]

        # Auto-contrast temporal stabilisation (EMA + clamp)
        if hasattr(self.single, "ac_smooth_alpha"):
            args += ["--ac-smooth-alpha", str(self.single.ac_smooth_alpha.value())]
            args += ["--ac-max-delta-alpha", str(self.single.ac_max_delta_alpha.value())]
            args += ["--ac-max-delta-beta", str(self.single.ac_max_delta_beta.value())]

        if hasattr(self.single, "disable_auto_contrast") and self.single.disable_auto_contrast.isChecked():
            args += ["--disable-auto-contrast"]
        if not self.single.precompute_filters.isChecked():
            args += ["--no-precompute-filters"]

        # Batch-only downsample factor (independent of Single/Segment tab)
        try:
            ds = int(self.batch_downsample.currentText().split()[0])
        except Exception:
            ds = 1
        ds = max(1, ds)
        args += ["--downsample", str(ds)]

        return args

    def start_batch(self):
        if not self._jobs:
            QMessageBox.information(self, "No jobs", "Add some videos first.")
            return

        out_dir_txt = self.out_dir.text().strip()
        if not out_dir_txt:
            # Default to "next to input" behaviour, but keep the UI explicit.
            QMessageBox.information(
                self,
                "Output folder required",
                "Please select an output folder for the batch tab (so results do not overwrite unexpected locations)."
            )
            return

        try:
            Path(out_dir_txt).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Invalid output folder", str(e))
            return

        if any(j.get("proc") is not None for j in self._jobs):
            QMessageBox.information(self, "Batch running", "Batch is already running. Stop it before restarting.")
            return

        self.append_log("Starting batch…")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_add_files.setEnabled(False)
        self.btn_add_folder.setEnabled(False)
        self.btn_clear.setEnabled(False)

        self._kick()

    def _kick(self):
        # Start jobs up to max_parallel
        max_par = int(self.max_parallel.value())
        running = sum(1 for j in self._jobs if j.get("proc") is not None and j["proc"].state() != QProcess.NotRunning)
        capacity = max(0, max_par - running)
        if capacity <= 0:
            self._update_batch_status()
            return

        for job in self._jobs:
            if capacity <= 0:
                break
            if job["status"] != "Queued":
                continue
            self._start_job(job)
            capacity -= 1

        self._update_batch_status()

    def _set_row_status(self, row: int, status: str):
        item = self.table.item(row, 0)
        if item is None:
            item = QTableWidgetItem(status)
            self.table.setItem(row, 0, item)
        else:
            item.setText(status)

    def _row_progressbar(self, row: int) -> QProgressBar:
        w = self.table.cellWidget(row, 1)
        return w if isinstance(w, QProgressBar) else None

    def _update_row_progress(self, job: dict):
        stage_pct = job.get("stage_pct", {})
        overall = 0.2 * float(stage_pct.get("ANALYZE", 0.0)) + 0.8 * float(stage_pct.get("PROCESS", 0.0))
        pb = self._row_progressbar(job["row"])
        if pb:
            pb.setValue(int(round(max(0.0, min(100.0, overall)))))
        self._refresh_table()

    def _start_job(self, job: dict):
        inp = job["inp"]
        outp = self._default_output_path(inp)
        # Keep table output in sync with current out folder
        self.table.item(job["row"], 3).setText(str(outp))
        job["out"] = outp

        outp.parent.mkdir(parents=True, exist_ok=True)

        try:
            args = self._build_video_args_full(inp, outp)
        except Exception as e:
            job["status"] = "Error"
            self._set_row_status(job["row"], f"Error: {e}")
            self.append_log(f"[{inp.name}] ARG ERROR: {e}")
            return

        job["stage_pct"] = {"ANALYZE": 0.0, "PROCESS": 0.0}
        self._update_row_progress(job)
        job["status"] = "Running"
        self._set_row_status(job["row"], "Running")

        proc = QProcess(self)
        proc.setProgram(sys.executable)
        proc.setArguments(args)
        proc.setProcessChannelMode(QProcess.MergedChannels)

        # Capture row/job in closures
        def _on_stdout():
            data = proc.readAllStandardOutput().data().decode(errors="replace")
            if not data:
                return
            for line in data.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("PROGRESS "):
                    parts = line.split()
                    if len(parts) == 3:
                        stage = parts[1].upper()
                        try:
                            pct = float(parts[2])
                        except ValueError:
                            continue
                        pct = max(0.0, min(100.0, pct))
                        if stage in job["stage_pct"]:
                            job["stage_pct"][stage] = pct
                        self._update_row_progress(job)
                        self._update_batch_status()
                else:
                    self.append_log(f"[{inp.name}] {line}")

        def _on_finished(exit_code: int, _exit_status):
            if exit_code == 0:
                job["status"] = "Done"
                self._set_row_status(job["row"], "Done")
                # Force row progress to 100 on success
                pb = self._row_progressbar(job["row"])
                self._refresh_table()
                if pb:
                    pb.setValue(100)
                self.append_log(f"[{inp.name}] DONE")
            else:
                job["status"] = f"Failed ({exit_code})"
                self._set_row_status(job["row"], job["status"])
                self.append_log(f"[{inp.name}] FAILED (exit {exit_code})")
                self._refresh_table()
            job["proc"] = None
            self._update_batch_status()

            # Start next job if any queued
            self._kick()

            # If everything is finished, unlock UI
            if all(j["status"] in {"Done"} or j["status"].startswith("Failed") or j["status"].startswith("Error") for j in self._jobs):
                self._finish_batch()

        proc.readyReadStandardOutput.connect(_on_stdout)
        proc.finished.connect(_on_finished)

        job["proc"] = proc
        proc.start()
        if not proc.waitForStarted(3000):
            job["status"] = "Failed to start"
            self._set_row_status(job["row"], job["status"])
            job["proc"] = None
            self.append_log(f"[{inp.name}] Failed to start process.")
            self._kick()

    def _finish_batch(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_add_files.setEnabled(True)
        self.btn_add_folder.setEnabled(True)
        self.btn_clear.setEnabled(True)
        self.append_log("Batch finished.")
        self._update_batch_status()

    def stop_all(self):
        any_running = False
        for job in self._jobs:
            proc = job.get("proc")
            if proc and proc.state() != QProcess.NotRunning:
                any_running = True
                try:
                    proc.kill()
                except Exception:
                    pass
                job["status"] = "Stopped"
                self._set_row_status(job["row"], "Stopped")
                job["proc"] = None

        if any_running:
            self.append_log("Stopped all running jobs.")

        # Reset UI state
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_add_files.setEnabled(True)
        self.btn_add_folder.setEnabled(True)
        self.btn_clear.setEnabled(True)
        self._update_batch_status()

    def _update_batch_status(self):
        total = len(self._jobs)
        if total == 0:
            self.batch_progress.setValue(0)
            self.batch_status.setText("No jobs.")
            return

        done = sum(1 for j in self._jobs if j["status"] == "Done")
        failed = sum(1 for j in self._jobs if j["status"].startswith("Failed") or j["status"].startswith("Error"))
        running = sum(1 for j in self._jobs if j["status"] == "Running")
        queued = sum(1 for j in self._jobs if j["status"] == "Queued")

        # Average row progress as a simple, stable batch indicator
        progress_vals = []
        for j in self._jobs:
            pb = self._row_progressbar(j["row"])
            if pb:
                progress_vals.append(int(pb.value()))
        avg = int(round(sum(progress_vals) / max(1, len(progress_vals))))

        self.batch_progress.setValue(avg)
        self.batch_status.setText(f"{done}/{total} done, {running} running, {queued} queued, {failed} failed")


class MainTabs(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Colour Corrector Mod- MUUC")
        self.resize(1300, 860)

        root = QVBoxLayout(self)
        tabs = QTabWidget()
        root.addWidget(tabs)

        self.single = BlueCorrectorSingleGUI()
        self.batch = BatchProcessorTab(self.single)

        tabs.addTab(self.single, "Single / Segment Preview")
        tabs.addTab(self.batch, "Batch (Parallel Full Videos)")


if __name__ == "__main__":
    # Backend mode: run the backend script in-process and exit.
    if len(sys.argv) > 1 and sys.argv[1] == "--backend":
        backend_path = resource_path("app_backend_segment_speedflags.py")
        sys.argv = [backend_path] + sys.argv[2:]
        runpy.run_path(backend_path, run_name="__main__")
        raise SystemExit(0)

    app = QApplication(sys.argv)
    w = MainTabs()
    w.show()
    sys.exit(app.exec())
