
"""
manual_segment_eyeavg_eeg_click_popup_v15.py

Qt GUI to:
- Load an epoched rsERG pickle file (3D array or dict of 3D arrays).
- Compute Eye1/Eye2 averages (Eye1=ch 1–8, Eye2=ch 9–16 by default).
- Optionally compute an EEG average from any remaining channels beyond Eye1/Eye2.
  * If 1 EEG channel is present, the EEG "average" is that channel.
  * If multiple EEG channels are present, they are averaged.
- Manually mark bad segments by clicking start/end in the plot, then pressing "Add segment".
  Segments are tracked separately for each displayed signal (Eye1, Eye2, EEG if present).
- Manage segments in a separate dialog (inspect & delete).
- Export a *cleaned* pickle where:
    * We start from the synthetic signals already in memory:
          data_sig : (n_sig, n_times)
          [0 -> Eye1_avg, 1 -> Eye2_avg, 2 -> EEG_avg (optional)]
    * Replace selected segments with NaNs in the corresponding synthetic channel(s).
    * Reshape back to a 3D array:
          (n_epochs, n_sig, epoch_len_samples)
      and save that array as the cleaned pickle.

This update is designed to work with outputs that include optional EEG channels
from file_load_v4_autoXML_chanmap_EEG.py.

Run from a terminal (NOT inside Jupyter):

    python manual_segment_eyeavg_eeg_click_popup_v15.py "/path/to/file_epoched.pkl" --sfreq 2000 --epoch-len 5

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
    QComboBox,
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
    sig: int          # 1=Eye1, 2=Eye2, 3=EEG
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
        segments_eeg: Optional[List[Tuple[int, int]]],
        sfreq: float,
        epoch_len_samples: int,
        has_eeg: bool,
    ):
        super().__init__(parent)
        self.setWindowTitle("Manage Segments & Export Cleaned Pickle")

        # share lists with parent (live updates)
        self.segments_eye1 = segments_eye1
        self.segments_eye2 = segments_eye2
        self.segments_eeg = segments_eeg if segments_eeg is not None else []
        self.sfreq = float(sfreq)
        self.epoch_len_samples = int(epoch_len_samples)
        self.has_eeg = bool(has_eeg)

        self.table = QTableWidget(self)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["Signal", "Start (s)", "End (s)", "Duration (s)", "Samples"]
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
        self.resize(620, 420)

        self.refresh_table()

    def _iter_segments(self) -> List[Segment]:
        segs: List[Segment] = []
        for (s, e) in self.segments_eye1:
            segs.append(Segment(sig=1, start=s, end=e))
        for (s, e) in self.segments_eye2:
            segs.append(Segment(sig=2, start=s, end=e))
        if self.has_eeg:
            for (s, e) in self.segments_eeg:
                segs.append(Segment(sig=3, start=s, end=e))
        segs.sort(key=lambda seg: (seg.start, seg.sig))
        return segs

    def _sig_name(self, sig: int) -> str:
        if sig == 1:
            return "Eye1"
        if sig == 2:
            return "Eye2"
        if sig == 3:
            return "EEG"
        return str(sig)

    def refresh_table(self):
        segs = self._iter_segments()
        self.table.setRowCount(len(segs))
        for row, seg in enumerate(segs):
            start_s = seg.start / self.sfreq
            end_s = seg.end / self.sfreq
            dur_s = end_s - start_s
            self.table.setItem(row, 0, QTableWidgetItem(self._sig_name(seg.sig)))
            self.table.setItem(row, 1, QTableWidgetItem(f"{start_s:.3f}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{end_s:.3f}"))
            self.table.setItem(row, 3, QTableWidgetItem(f"{dur_s:.3f}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{seg.start}–{seg.end}"))

    def on_delete_selected(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            return

        segs = self._iter_segments()
        for r in rows:
            if 0 <= r < len(segs):
                seg = segs[r]
                if seg.sig == 1:
                    self.segments_eye1[:] = [
                        s for s in self.segments_eye1
                        if not (s[0] == seg.start and s[1] == seg.end)
                    ]
                elif seg.sig == 2:
                    self.segments_eye2[:] = [
                        s for s in self.segments_eye2
                        if not (s[0] == seg.start and s[1] == seg.end)
                    ]
                elif seg.sig == 3:
                    self.segments_eeg[:] = [
                        s for s in self.segments_eeg
                        if not (s[0] == seg.start and s[1] == seg.end)
                    ]

        self.refresh_table()
        if hasattr(self.parent(), "update_plot"):
            self.parent().update_plot()

    def _export_from_sigdata(self, out_path: str):
        """Export cleaned synthetic signal data from parent.data_sig."""
        parent = self.parent()
        if parent is None or not hasattr(parent, "data_sig"):
            raise RuntimeError("Parent window with data_sig not found.")

        data_sig = np.asarray(parent.data_sig, dtype=float)  # (n_sig, n_times)
        if data_sig.ndim != 2:
            raise ValueError(
                f"Expected data_sig to be 2D (n_sig, n_times), got shape {data_sig.shape}."
            )

        n_sig, n_times = data_sig.shape
        data_clean = data_sig.copy()

        seg_lists = [
            self.segments_eye1,
            self.segments_eye2,
        ]
        if self.has_eeg and n_sig >= 3:
            seg_lists.append(self.segments_eeg)

        def _apply_seg(seg_list: List[Tuple[int, int]], ch_index: int):
            for s, e in seg_list:
                s_cl = max(0, min(n_times, int(s)))
                e_cl = max(0, min(n_times, int(e)))
                if e_cl <= s_cl:
                    continue
                data_clean[ch_index, s_cl:e_cl] = np.nan

        for i, seg_list in enumerate(seg_lists):
            if i < n_sig:
                _apply_seg(seg_list, i)

        if self.epoch_len_samples <= 0:
            raise ValueError("epoch_len_samples must be positive.")
        if n_times % self.epoch_len_samples != 0:
            raise ValueError(
                f"Total samples n_times={n_times} is not divisible by "
                f"epoch_len_samples={self.epoch_len_samples}."
            )

        n_epochs = n_times // self.epoch_len_samples

        # (n_sig, n_times) -> (n_sig, n_epochs, epoch_len) -> (n_epochs, n_sig, epoch_len)
        data_3d = data_clean.reshape(
            n_sig, n_epochs, self.epoch_len_samples
        ).transpose(1, 0, 2)

        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(out_path, "wb") as f_out:
            pickle.dump(data_3d, f_out, protocol=pickle.HIGHEST_PROTOCOL)

        print(
            f"[Export] Saved cleaned synthetic array with shape {data_3d.shape} "
            f"to: {out_path}"
        )

    def on_export(self):
        parent = self.parent()
        default_dir = os.path.dirname(os.path.abspath(parent.input_pkl_path))
        base = os.path.splitext(os.path.basename(parent.input_pkl_path))[0]
        suffix = "_EyeEEGAvg_manualNaN.pkl" if self.has_eeg else "_EyeAvg_manualNaN.pkl"
        default_name = base + suffix

        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save cleaned synthetic pickle",
            os.path.join(default_dir, default_name),
            "Pickle files (*.pkl);;All files (*)",
        )
        if not out_path:
            return

        try:
            self._export_from_sigdata(out_path)
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
            f"Cleaned synthetic pickle saved to:\n{out_path}",
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
        eeg_indices: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.setWindowTitle("Manual Artifact Segmentation (Eye + optional EEG averages)")

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
        self.epoch_len_samples = int(epoch_len_samples)
        self.block_n_epochs = block_n_epochs
        self.ch_names = ch_names

        # Default EEG indices: any channels not in Eye1/Eye2
        used = set(self.eye1_indices) | set(self.eye2_indices)
        if eeg_indices is None:
            eeg_indices = [i for i in range(self.n_channels) if i not in used]
        self.eeg_indices = [i for i in eeg_indices if 0 <= i < self.n_channels]

        # Create synthetic signals
        self.data_sig, self.sig_names = self._compute_synthetic_signals()
        self.has_eeg = ("EEG" in self.sig_names)

        # segment lists per signal
        self.segments_eye1: List[Tuple[int, int]] = []
        self.segments_eye2: List[Tuple[int, int]] = []
        self.segments_eeg: List[Tuple[int, int]] = []

        # Display state
        self.current_sig = 0  # index into data_sig
        self.window_sec = 20.0
        # Amplitude zoom factor for display (higher = zoom in)
        self.amp_zoom = 1.0
        # Amplitude centering mode: 'Auto' uses robust center; 'Zero' locks center at 0; 'Manual' uses amp_center_value
        self.amp_center_mode = 'Auto'
        self.amp_center_value = 0.0
        self.center_sample = self.n_times // 2

        self.temp_start_sample: Optional[int] = None
        self.temp_end_sample: Optional[int] = None

        self._init_ui()
        self.update_plot()

    # ------------------------------------------------------------------
    # Synthetic signals
    # ------------------------------------------------------------------
    def _compute_synthetic_signals(self) -> Tuple[np.ndarray, List[str]]:
        data = self.data_cont_full

        eye1 = [idx for idx in self.eye1_indices if 0 <= idx < data.shape[0]]
        eye2 = [idx for idx in self.eye2_indices if 0 <= idx < data.shape[0]]
        eeg = [idx for idx in self.eeg_indices if 0 <= idx < data.shape[0]]

        if not eye1 and not eye2:
            raise RuntimeError("No valid Eye1 or Eye2 channel indices.")

        sig_list = []
        names = []

        # Eye1 avg
        if eye1:
            sig_list.append(data[eye1, :].mean(axis=0))
        else:
            sig_list.append(np.zeros(data.shape[1], dtype=float))
        names.append("Eye1")

        # Eye2 avg
        if eye2:
            sig_list.append(data[eye2, :].mean(axis=0))
        else:
            sig_list.append(np.zeros(data.shape[1], dtype=float))
        names.append("Eye2")

        # EEG avg (optional)
        if eeg:
            if len(eeg) == 1:
                sig_list.append(data[eeg[0], :])
            else:
                sig_list.append(data[eeg, :].mean(axis=0))
            names.append("EEG")

        data_sig = np.vstack(sig_list)  # (n_sig, n_times)
        # ensure float for NaN edits
        return data_sig.astype(float), names

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
        self.label_sig = QLabel("Signal: Eye1 / Eye2 / EEG")
        ctrl_row1.addWidget(self.label_sig)

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
        ctrl_row1.addWidget(QLabel("Amp zoom:"))
        self.spin_amp_zoom = QDoubleSpinBox()
        self.spin_amp_zoom.setDecimals(2)
        self.spin_amp_zoom.setRange(0.1, 50.0)
        self.spin_amp_zoom.setSingleStep(0.1)
        self.spin_amp_zoom.setValue(self.amp_zoom)
        self.spin_amp_zoom.setToolTip(
            "Controls vertical zoom for the currently displayed signal. "
            "Higher values zoom in; lower values zoom out."
        )
        self.spin_amp_zoom.valueChanged.connect(self.on_amp_zoom_changed)
        ctrl_row1.addWidget(self.spin_amp_zoom)

        ctrl_row1.addSpacing(16)
        ctrl_row1.addWidget(QLabel("Amp center:"))
        self.combo_amp_center = QComboBox()
        self.combo_amp_center.addItems(["Auto", "Zero", "Manual"])
        self.combo_amp_center.setCurrentText(self.amp_center_mode)
        self.combo_amp_center.setToolTip(
            "Auto: center y-limits on robust signal center. "
            "Zero: lock center at 0. "
            "Manual: use the value in 'Center value'."
        )
        self.combo_amp_center.currentTextChanged.connect(self.on_amp_center_mode_changed)
        ctrl_row1.addWidget(self.combo_amp_center)

        ctrl_row1.addSpacing(8)
        ctrl_row1.addWidget(QLabel("Center value:"))
        self.spin_amp_center_val = QDoubleSpinBox()
        self.spin_amp_center_val.setDecimals(3)
        self.spin_amp_center_val.setRange(-1e6, 1e6)
        self.spin_amp_center_val.setSingleStep(1.0)
        self.spin_amp_center_val.setValue(self.amp_center_value)
        self.spin_amp_center_val.setEnabled(self.amp_center_mode == "Manual")
        self.spin_amp_center_val.setToolTip(
            "Manual y-center for display (same units as the signal)."
        )
        self.spin_amp_center_val.valueChanged.connect(self.on_amp_center_value_changed)
        ctrl_row1.addWidget(self.spin_amp_center_val)

        ctrl_row1.addSpacing(20)
        self.label_pos = QLabel("Center: 0.0 s")
        ctrl_row1.addWidget(self.label_pos)

        ctrl_row1.addStretch()
        layout.addLayout(ctrl_row1)

        ctrl_row2 = QHBoxLayout()

        self.btn_add = QPushButton("Add segment (between clicks)")
        self.btn_add.clicked.connect(self.on_add_segment)
        ctrl_row2.addWidget(self.btn_add)

        self.btn_undo = QPushButton("Undo last (current signal)")
        self.btn_undo.clicked.connect(self.on_undo_last)
        ctrl_row2.addWidget(self.btn_undo)

        ctrl_row2.addStretch()

        self.btn_manage = QPushButton("Manage / Export…")
        self.btn_manage.clicked.connect(self.on_manage)
        ctrl_row2.addWidget(self.btn_manage)

        layout.addLayout(ctrl_row2)

        key_hint = "Keyboard: ← / → scroll, '1'/'2'"
        if self.has_eeg:
            key_hint += "/'3'"
        key_hint += " switch signal, 'U' undo (current), 'M' manage/export, 'Q' quit."

        instructions = (
            "Mouse: left-click start and end of a bad segment, then click "
            "'Add segment' to commit.\n"
            "Use 'Window (s)' to zoom time. Use 'Amp zoom' and 'Amp center' to control amplitude display.\n"
            + key_hint
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
            allowed = {
                QtCore.Qt.Key_Left,
                QtCore.Qt.Key_Right,
                QtCore.Qt.Key_1,
                QtCore.Qt.Key_2,
                QtCore.Qt.Key_U,
                QtCore.Qt.Key_M,
                QtCore.Qt.Key_Q,
            }
            if self.has_eeg:
                allowed.add(QtCore.Qt.Key_3)

            if key in allowed:
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
            self.current_sig = 0
            self.update_plot()
        elif key == QtCore.Qt.Key_2:
            self.current_sig = 1 if len(self.sig_names) > 1 else 0
            self.update_plot()
        elif key == QtCore.Qt.Key_3 and self.has_eeg:
            # EEG is always the last synthetic channel if present
            self.current_sig = len(self.sig_names) - 1
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
        y = self.data_sig[self.current_sig, start:end]
        ax.plot(t, y, color="black", linewidth=0.8)

        # Apply amplitude zoom (robust auto-range scaled by amp_zoom)
        finite = y[np.isfinite(y)]
        if finite.size > 1:
            try:
                y_lo, y_hi = np.percentile(finite, [1, 99])
            except Exception:
                y_lo, y_hi = float(np.min(finite)), float(np.max(finite))
            if y_hi == y_lo:
                y_hi = y_lo + 1.0
            robust_center = float(np.median(finite))
            robust_half_range = 0.5 * (y_hi - y_lo)
        elif finite.size == 1:
            robust_center = float(finite[0])
            robust_half_range = 1.0
        else:
            robust_center = 0.0
            robust_half_range = 1.0

        # Determine center based on user mode
        mode = getattr(self, "amp_center_mode", "Auto")
        if mode == "Zero":
            center_y = 0.0
        elif mode == "Manual":
            center_y = float(getattr(self, "amp_center_value", 0.0))
        else:
            # Auto
            center_y = robust_center

        zoom = max(float(getattr(self, "amp_zoom", 1.0)), 0.1)
        half_range = robust_half_range / zoom

        if not np.isfinite(half_range) or half_range <= 0:
            half_range = 1.0

        # Add a small floor to avoid over-collapsing when signal is extremely flat
        min_floor = 1e-6
        if half_range < min_floor:
            half_range = min_floor

        ax.set_ylim(center_y - half_range, center_y + half_range)

        # temp markers
        if self.temp_start_sample is not None and start <= self.temp_start_sample < end:
            ts = self.temp_start_sample / self.sfreq
            ax.axvline(ts, color="red", linestyle="--", linewidth=1.0)
        if self.temp_end_sample is not None and start <= self.temp_end_sample < end:
            te = self.temp_end_sample / self.sfreq
            ax.axvline(te, color="red", linestyle="--", linewidth=1.0)

        # segment overlays
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

        # EEG segments (light yellow)
        if self.has_eeg:
            for (s, e) in self.segments_eeg:
                if e <= start or s >= end:
                    continue
                ss = max(s, start) / self.sfreq
                ee = min(e, end) / self.sfreq
                ax.axvspan(ss, ee, facecolor=(1.0, 1.0, 0.6, 0.35), edgecolor=None)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (µV)")
        sig_label = self.sig_names[self.current_sig] if self.current_sig < len(self.sig_names) else f"Sig{self.current_sig+1}"
        ax.set_title(f"{sig_label} (cyan=Eye1 segs, pink=Eye2 segs" + (", yellow=EEG segs)" if self.has_eeg else ")"))

        self.label_sig.setText(
            f"Signal: {sig_label} (keys 1/2" + ("/3" if self.has_eeg else "") + ")"
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

    def on_amp_zoom_changed(self, val: float):
        """Update amplitude zoom factor for display."""
        self.amp_zoom = float(val)
        if not np.isfinite(self.amp_zoom) or self.amp_zoom <= 0:
            self.amp_zoom = 1.0
        self.update_plot()

    def on_amp_center_mode_changed(self, text: str):
        """Change amplitude centering mode."""
        self.amp_center_mode = str(text)
        try:
            self.spin_amp_center_val.setEnabled(self.amp_center_mode == "Manual")
        except Exception:
            pass
        self.update_plot()

    def on_amp_center_value_changed(self, val: float):
        """Update manual amplitude center value."""
        self.amp_center_value = float(val)
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

    def _segments_for_current_sig(self) -> List[Tuple[int, int]]:
        if self.current_sig == 0:
            return self.segments_eye1
        if self.current_sig == 1:
            return self.segments_eye2
        return self.segments_eeg

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
        seg_list = self._segments_for_current_sig()
        seg_list.append((s, e))
        seg_list[:] = self._merge_segments(seg_list)
        self.temp_start_sample = None
        self.temp_end_sample = None
        self.update_plot()

    def on_undo_last(self):
        seg_list = self._segments_for_current_sig()
        if seg_list:
            seg_list.pop()
            self.update_plot()

    def on_manage(self):
        dlg = ManageExportDialog(
            parent=self,
            segments_eye1=self.segments_eye1,
            segments_eye2=self.segments_eye2,
            segments_eeg=self.segments_eeg if self.has_eeg else None,
            sfreq=self.sfreq,
            epoch_len_samples=self.epoch_len_samples,
            has_eeg=self.has_eeg,
        )
        dlg.exec_()
        self.update_plot()


# -------------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python manual_segment_eyeavg_eeg_click_popup_v15.py "
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
    # Global key handling: arrows, 1/2/3, U, M, Q
    app.installEventFilter(win)
    win.resize(1000, 620)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
