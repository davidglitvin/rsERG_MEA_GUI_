"""
manual_segment_eyeavg_click_popup_v10.py

Qt GUI to:
- Load an epoched rsERG pickle file (3D array or dict of 3D arrays).
- Compute Eye1/Eye2 averages (Eye1=ch 1–8, Eye2=ch 9–16 by default).
- Manually mark bad segments separately in Eye1/Eye2 by clicking
  start/end in the plot, then pressing "Add segment".
- Manage segments in a separate dialog (inspect & delete).
- Export a *cleaned* pickle where:
    * We start from the Eye averages already in memory:
          data_eye : (2, n_times)   [0 -> Eye1_avg, 1 -> Eye2_avg]
    * Replace selected Eye1/Eye2 segments with NaNs in those two channels.
    * Reshape back to a 3D array:
          (n_epochs, 2, epoch_len_samples)
      and save that array as the cleaned pickle.

v10 changes
-----------
- Export logic is now strictly for the 2 synthetic channels (Eye1_avg & Eye2_avg).
- We do NOT touch or try to recreate the original 16-channel structure.
- The saved object is always a NumPy array with shape
      (n_epochs, 2, epoch_len_samples),
  with NaNs in the marked segments.
- Keeps v9’s global keyboard handling:
      ← / →  : scroll
      '1'/'2': switch Eye1/Eye2
      'U'    : undo last (current eye)
      'M'    : open Manage/Export dialog
      'Q'    : quit

Run from a terminal (NOT inside Jupyter):

    python manual_segment_eyeavg_click_popup_v10.py "/path/to/file_epoched.pkl" --sfreq 2000 --epoch-len 5
"""

import sys
import os
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Dict, Optional

import numpy as np

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QLabel,
    QDialog,
    QTableWidget,
    QTableWidgetItem,
    QFileDialog,
    QMessageBox,
    QDoubleSpinBox,
)

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# -------------------------------------------------------------------------
# Data loading helpers
# -------------------------------------------------------------------------

def load_pickle_as_continuous(
    file_path: str,
    sfreq: float = 2000.0,
    epoch_len_s: float = 5.0,
) -> Tuple[np.ndarray, int, List[str], Dict[str, int]]:
    """Load pickle and return continuous array (n_channels, n_times).

    Supports:
      - dict of arrays: concatenates along epoch axis in sorted(key) order
      - 3D array: (n_epochs, n_channels, n_times)
      - 2D array: (n_channels, n_times)

    Returns
    -------
    data_cont : ndarray, shape (n_channels, n_times)
    epoch_len_samples : int
        Epoch length in samples. For 3D/dict this is n_times per epoch.
        For 2D this is int(epoch_len_s * sfreq).
    ch_names : list of str
    block_n_epochs : dict
        For dict input, maps key -> n_epochs for that block.
        For non-dict input, empty dict.
    """
    with open(file_path, "rb") as f:
        obj = pickle.load(f)

    block_n_epochs: Dict[str, int] = {}

    if isinstance(obj, dict):
        arrays = []
        for key in sorted(obj.keys()):
            arr = np.asarray(obj[key])
            if arr.ndim != 3:
                raise ValueError(
                    f"Dict entry '{key}' has shape {arr.shape}, expected 3D "
                    "(n_epochs, n_channels, n_times)."
                )
            block_n_epochs[key] = arr.shape[0]
            arrays.append(arr)
        if not arrays:
            raise ValueError("Dictionary in pickle is empty.")
        data = np.concatenate(arrays, axis=0)
    else:
        data = np.asarray(obj)

    if data.ndim == 3:
        n_epochs, n_channels, n_times = data.shape
        data_cont = data.transpose(1, 0, 2).reshape(n_channels, n_epochs * n_times)
        epoch_len_samples = n_times
    elif data.ndim == 2:
        n_channels, n_times = data.shape
        data_cont = data
        epoch_len_samples = int(round(epoch_len_s * sfreq))
        if epoch_len_samples <= 0:
            raise ValueError("epoch_len_s must be positive.")
    else:
        raise ValueError(
            f"Unsupported pickle array shape {data.shape}, expected 2D or 3D."
        )

    ch_names = [f"Ch{i+1}" for i in range(data_cont.shape[0])]
    return data_cont.astype(float), epoch_len_samples, ch_names, block_n_epochs


# -------------------------------------------------------------------------
# Segment structure
# -------------------------------------------------------------------------

@dataclass
class Segment:
    eye: int          # 1 or 2
    start: int        # sample index (inclusive)
    end: int          # sample index (exclusive)


# -------------------------------------------------------------------------
# Matplotlib canvas
# -------------------------------------------------------------------------

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        fig = Figure(figsize=(8, 4))
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        fig.tight_layout()


# -------------------------------------------------------------------------
# Manage / Export dialog
# -------------------------------------------------------------------------

class ManageExportDialog(QDialog):
    def __init__(
        self,
        parent,
        segments_eye1: List[Tuple[int, int]],
        segments_eye2: List[Tuple[int, int]],
        sfreq: float,
        epoch_len_samples: int,
    ):
        super().__init__(parent)
        self.setWindowTitle("Manage Segments & Export Cleaned Pickle")

        # share lists with parent (live updates)
        self.segments_eye1 = segments_eye1
        self.segments_eye2 = segments_eye2
        self.sfreq = sfreq
        self.epoch_len_samples = epoch_len_samples

        self.table = QTableWidget(self)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["Eye", "Start (s)", "End (s)", "Duration (s)", "Samples"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)

        btn_delete = QPushButton("Delete selected")
        btn_export = QPushButton("Export cleaned pickle…")

        btn_delete.clicked.connect(self.on_delete_selected)
        btn_export.clicked.connect(self.on_export)

        layout = QVBoxLayout()
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        btn_row.addWidget(btn_delete)
        btn_row.addWidget(btn_export)
        layout.addLayout(btn_row)

        self.setLayout(layout)
        self.resize(600, 400)

        self.refresh_table()

    def _iter_segments_with_eye(self) -> List[Segment]:
        segs: List[Segment] = []
        for (s, e) in self.segments_eye1:
            segs.append(Segment(eye=1, start=s, end=e))
        for (s, e) in self.segments_eye2:
            segs.append(Segment(eye=2, start=s, end=e))
        segs.sort(key=lambda seg: (seg.start, seg.eye))
        return segs

    def refresh_table(self):
        segs = self._iter_segments_with_eye()
        self.table.setRowCount(len(segs))
        for row, seg in enumerate(segs):
            start_s = seg.start / self.sfreq
            end_s = seg.end / self.sfreq
            dur_s = end_s - start_s
            self.table.setItem(row, 0, QTableWidgetItem(str(seg.eye)))
            self.table.setItem(row, 1, QTableWidgetItem(f"{start_s:.3f}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{end_s:.3f}"))
            self.table.setItem(row, 3, QTableWidgetItem(f"{dur_s:.3f}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{seg.start}–{seg.end}"))

    def on_delete_selected(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            return

        segs = self._iter_segments_with_eye()
        for r in rows:
            if 0 <= r < len(segs):
                seg = segs[r]
                if seg.eye == 1:
                    self.segments_eye1[:] = [
                        s for s in self.segments_eye1
                        if not (s[0] == seg.start and s[1] == seg.end)
                    ]
                else:
                    self.segments_eye2[:] = [
                        s for s in self.segments_eye2
                        if not (s[0] == seg.start and s[1] == seg.end)
                    ]

        self.refresh_table()
        if hasattr(self.parent(), "update_plot"):
            self.parent().update_plot()

    def _export_from_eyedata(self, out_path: str):
        """Export cleaned 2-channel eye-average data from parent.data_eye."""
        parent = self.parent()
        if parent is None or not hasattr(parent, "data_eye"):
            raise RuntimeError("Parent window with data_eye not found.")

        data_eye = np.asarray(parent.data_eye, dtype=float)  # (2, n_times)
        if data_eye.shape[0] != 2:
            raise ValueError(
                f"Expected data_eye to have 2 channels, got shape {data_eye.shape}."
            )

        _, n_times = data_eye.shape
        data_clean = data_eye.copy()

        def _apply_seg(seg_list: List[Tuple[int, int]], ch_index: int):
            for s, e in seg_list:
                s_cl = max(0, min(n_times, int(s)))
                e_cl = max(0, min(n_times, int(e)))
                if e_cl <= s_cl:
                    continue
                data_clean[ch_index, s_cl:e_cl] = np.nan

        # Eye1 (index 0), Eye2 (index 1)
        _apply_seg(self.segments_eye1, 0)
        _apply_seg(self.segments_eye2, 1)

        if self.epoch_len_samples <= 0:
            raise ValueError("epoch_len_samples must be positive.")
        if n_times % self.epoch_len_samples != 0:
            raise ValueError(
                f"Total samples n_times={n_times} is not divisible by "
                f"epoch_len_samples={self.epoch_len_samples}."
            )

        n_epochs = n_times // self.epoch_len_samples
        # (2, n_times) -> (2, n_epochs, epoch_len_samples) -> (n_epochs, 2, n_times_epoch)
        data_3d = data_clean.reshape(
            2, n_epochs, self.epoch_len_samples
        ).transpose(1, 0, 2)

        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(out_path, "wb") as f_out:
            pickle.dump(data_3d, f_out, protocol=pickle.HIGHEST_PROTOCOL)

        print(
            f"[Export] Saved cleaned 3D Eye-average array with shape {data_3d.shape} "
            f"to: {out_path}"
        )

    def on_export(self):
        parent = self.parent()
        default_dir = os.path.dirname(os.path.abspath(parent.input_pkl_path))
        default_name = (
            os.path.splitext(os.path.basename(parent.input_pkl_path))[0]
            + "_EyeAvg_manualNaN.pkl"
        )
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save cleaned Eye-average pickle",
            os.path.join(default_dir, default_name),
            "Pickle files (*.pkl);;All files (*)",
        )
        if not out_path:
            return

        try:
            self._export_from_eyedata(out_path)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Export failed",
                f"An error occurred while exporting cleaned pickle:\n{exc}",
            )
            return

        QMessageBox.information(
            self,
            "Export complete",
            f"Cleaned Eye-average pickle saved to:\n{out_path}",
        )


# -------------------------------------------------------------------------
# Main window
# -------------------------------------------------------------------------

class ManualSegmentWindow(QMainWindow, QtCore.QObject):
    def __init__(
        self,
        input_pkl_path: str,
        sfreq: float = 2000.0,
        epoch_len_s: float = 5.0,
        eye1_indices: Sequence[int] = tuple(range(0, 8)),
        eye2_indices: Sequence[int] = tuple(range(8, 16)),
    ):
        super().__init__()
        self.setWindowTitle("Manual Artifact Segmentation (Eye Averages)")

        self.input_pkl_path = input_pkl_path
        self.sfreq = float(sfreq)
        self.epoch_len_s = float(epoch_len_s)
        self.eye1_indices = list(eye1_indices)
        self.eye2_indices = list(eye2_indices)

        # Load data as continuous (all channels)
        data_cont, epoch_len_samples, ch_names, block_n_epochs = load_pickle_as_continuous(
            input_pkl_path,
            sfreq=self.sfreq,
            epoch_len_s=self.epoch_len_s,
        )
        self.data_cont_full = data_cont  # (n_channels, n_times)
        self.n_channels, self.n_times = self.data_cont_full.shape
        self.epoch_len_samples = epoch_len_samples
        self.block_n_epochs = block_n_epochs

        # Create 2-channel eye-average data
        self.data_eye = self._compute_eye_averages()  # (2, n_times)
        self.segments_eye1: List[Tuple[int, int]] = []
        self.segments_eye2: List[Tuple[int, int]] = []

        self.current_eye = 0  # 0 -> Eye1, 1 -> Eye2
        self.window_sec = 20.0
        self.center_sample = self.n_times // 2

        self.temp_start_sample: Optional[int] = None
        self.temp_end_sample: Optional[int] = None

        self._init_ui()
        self.update_plot()

    # ------------------------------------------------------------------
    # Eye averaging
    # ------------------------------------------------------------------
    def _compute_eye_averages(self) -> np.ndarray:
        data = self.data_cont_full
        eye1 = [idx for idx in self.eye1_indices if 0 <= idx < data.shape[0]]
        eye2 = [idx for idx in self.eye2_indices if 0 <= idx < data.shape[0]]

        if not eye1 and not eye2:
            raise RuntimeError("No valid Eye1 or Eye2 channel indices.")

        out = []
        if eye1:
            out.append(data[eye1, :].mean(axis=0))
        else:
            out.append(np.zeros(data.shape[1], dtype=float))

        if eye2:
            out.append(data[eye2, :].mean(axis=0))
        else:
            out.append(np.zeros(data.shape[1], dtype=float))

        return np.vstack(out)  # (2, n_times)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _init_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)

        layout = QVBoxLayout()
        central.setLayout(layout)

        self.canvas = MplCanvas(self)
        layout.addWidget(self.canvas)

        ctrl_row1 = QHBoxLayout()
        self.label_eye = QLabel("Eye: 1 (cyan) / 2 (pink)")
        ctrl_row1.addWidget(self.label_eye)

        ctrl_row1.addSpacing(20)
        ctrl_row1.addWidget(QLabel("Window (s):"))
        self.spin_window = QDoubleSpinBox()
        self.spin_window.setDecimals(1)
        self.spin_window.setRange(1.0, 120.0)
        self.spin_window.setSingleStep(5.0)
        self.spin_window.setValue(self.window_sec)
        self.spin_window.valueChanged.connect(self.on_window_changed)
        ctrl_row1.addWidget(self.spin_window)

        ctrl_row1.addSpacing(20)
        self.label_pos = QLabel("Center: 0.0 s")
        ctrl_row1.addWidget(self.label_pos)

        ctrl_row1.addStretch()
        layout.addLayout(ctrl_row1)

        ctrl_row2 = QHBoxLayout()

        self.btn_add = QPushButton("Add segment (between clicks)")
        self.btn_add.clicked.connect(self.on_add_segment)
        ctrl_row2.addWidget(self.btn_add)

        self.btn_undo = QPushButton("Undo last (current eye)")
        self.btn_undo.clicked.connect(self.on_undo_last)
        ctrl_row2.addWidget(self.btn_undo)

        ctrl_row2.addStretch()

        self.btn_manage = QPushButton("Manage / Export…")
        self.btn_manage.clicked.connect(self.on_manage)
        ctrl_row2.addWidget(self.btn_manage)

        layout.addLayout(ctrl_row2)

        instructions = (
            "Mouse: left-click start and end of a bad segment, then click "
            "'Add segment' to commit.\n"
            "Keyboard: ← / → to scroll, '1' or '2' to switch Eye, 'U' to undo "
            "last segment (current eye), 'M' to open Manage/Export, 'Q' to quit."
        )
        self.label_help = QLabel(instructions)
        self.label_help.setWordWrap(True)
        layout.addWidget(self.label_help)

        # Matplotlib click events
        self.canvas.mpl_connect("button_press_event", self.on_mpl_click)

    # ------------------------------------------------------------------
    # Global key handling via eventFilter
    # ------------------------------------------------------------------
    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress:
            key = event.key()
            if key in (
                QtCore.Qt.Key_Left,
                QtCore.Qt.Key_Right,
                QtCore.Qt.Key_1,
                QtCore.Qt.Key_2,
                QtCore.Qt.Key_U,
                QtCore.Qt.Key_M,
                QtCore.Qt.Key_Q,
            ):
                self._handle_key(key)
                return True
        return super().eventFilter(obj, event)

    def _handle_key(self, key):
        step = int(0.25 * self.window_sec * self.sfreq)
        if step < 1:
            step = 1

        if key == QtCore.Qt.Key_Left:
            self.center_sample = max(0, self.center_sample - step)
            self.update_plot()
        elif key == QtCore.Qt.Key_Right:
            self.center_sample = min(self.n_times - 1, self.center_sample + step)
            self.update_plot()
        elif key == QtCore.Qt.Key_1:
            self.current_eye = 0
            self.update_plot()
        elif key == QtCore.Qt.Key_2:
            self.current_eye = 1
            self.update_plot()
        elif key == QtCore.Qt.Key_U:
            self.on_undo_last()
        elif key == QtCore.Qt.Key_M:
            self.on_manage()
        elif key == QtCore.Qt.Key_Q:
            self.close()

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    def update_plot(self):
        ax = self.canvas.ax
        ax.clear()

        half_win = int(0.5 * self.window_sec * self.sfreq)
        start = max(0, self.center_sample - half_win)
        end = min(self.n_times, self.center_sample + half_win)
        if end <= start:
            start = max(0, self.n_times - 1)
            end = self.n_times

        t = np.arange(start, end) / self.sfreq
        y = self.data_eye[self.current_eye, start:end]
        ax.plot(t, y, color="black", linewidth=0.8)

        # temp markers
        if self.temp_start_sample is not None:
            if start <= self.temp_start_sample < end:
                ts = self.temp_start_sample / self.sfreq
                ax.axvline(ts, color="red", linestyle="--", linewidth=1.0)
        if self.temp_end_sample is not None:
            if start <= self.temp_end_sample < end:
                te = self.temp_end_sample / self.sfreq
                ax.axvline(te, color="red", linestyle="--", linewidth=1.0)

        # Eye1 segments (cyan)
        for (s, e) in self.segments_eye1:
            if e <= start or s >= end:
                continue
            ss = max(s, start) / self.sfreq
            ee = min(e, end) / self.sfreq
            ax.axvspan(ss, ee, facecolor=(0.7, 1.0, 1.0, 0.4), edgecolor=None)

        # Eye2 segments (pink)
        for (s, e) in self.segments_eye2:
            if e <= start or s >= end:
                continue
            ss = max(s, start) / self.sfreq
            ee = min(e, end) / self.sfreq
            ax.axvspan(ss, ee, facecolor=(1.0, 0.7, 0.8, 0.4), edgecolor=None)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Eye avg (µV)")
        ax.set_title(f"Eye {self.current_eye + 1} (cyan=Eye1, pink=Eye2)")

        self.label_eye.setText(
            f"Eye: {self.current_eye + 1} (cyan=Eye1, pink=Eye2)"
        )
        center_sec = self.center_sample / self.sfreq
        self.label_pos.setText(f"Center: {center_sec:.3f} s")

        ax.grid(True, alpha=0.2)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def on_window_changed(self, val: float):
        self.window_sec = float(val)
        self.update_plot()

    def on_mpl_click(self, event):
        if event.inaxes != self.canvas.ax:
            return
        if event.button != 1:
            return
        if event.xdata is None:
            return

        sample_idx = int(round(event.xdata * self.sfreq))
        sample_idx = max(0, min(self.n_times - 1, sample_idx))

        if self.temp_start_sample is None:
            self.temp_start_sample = sample_idx
            self.temp_end_sample = None
        else:
            self.temp_end_sample = sample_idx

        self.update_plot()

    def _segments_for_current_eye(self) -> List[Tuple[int, int]]:
        return self.segments_eye1 if self.current_eye == 0 else self.segments_eye2

    @staticmethod
    def _merge_segments(seg_list: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not seg_list:
            return []
        seg_list = sorted(seg_list, key=lambda x: x[0])
        merged = [seg_list[0]]
        for s, e in seg_list[1:]:
            last_s, last_e = merged[-1]
            if s <= last_e:
                merged[-1] = (last_s, max(last_e, e))
            else:
                merged.append((s, e))
        return merged

    def on_add_segment(self):
        if self.temp_start_sample is None or self.temp_end_sample is None:
            return
        s = min(self.temp_start_sample, self.temp_end_sample)
        e = max(self.temp_start_sample, self.temp_end_sample)
        if e <= s:
            return
        seg_list = self._segments_for_current_eye()
        seg_list.append((s, e))
        seg_list[:] = self._merge_segments(seg_list)
        self.temp_start_sample = None
        self.temp_end_sample = None
        self.update_plot()

    def on_undo_last(self):
        seg_list = self._segments_for_current_eye()
        if seg_list:
            seg_list.pop()
            self.update_plot()

    def on_manage(self):
        dlg = ManageExportDialog(
            parent=self,
            segments_eye1=self.segments_eye1,
            segments_eye2=self.segments_eye2,
            sfreq=self.sfreq,
            epoch_len_samples=self.epoch_len_samples,
        )
        dlg.exec_()
        self.update_plot()


# -------------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python manual_segment_eyeavg_click_popup_v10.py "
              "\"/path/to/file_epoched.pkl\" [--sfreq 2000] [--epoch-len 5]")
        sys.exit(1)

    input_pkl_path = sys.argv[1]
    sfreq = 2000.0
    epoch_len_s = 5.0

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--sfreq" and i + 1 < len(args):
            sfreq = float(args[i + 1])
            i += 2
        elif args[i] == "--epoch-len" and i + 1 < len(args):
            epoch_len_s = float(args[i + 1])
            i += 2
        else:
            i += 1

    if not os.path.isfile(input_pkl_path):
        print(f"ERROR: Input pickle not found: {input_pkl_path}")
        sys.exit(1)

    app = QApplication(sys.argv)
    win = ManualSegmentWindow(
        input_pkl_path=input_pkl_path,
        sfreq=sfreq,
        epoch_len_s=epoch_len_s,
    )
    # Global key handling: arrows, 1/2, U, M, Q
    app.installEventFilter(win)
    win.resize(1000, 600)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
