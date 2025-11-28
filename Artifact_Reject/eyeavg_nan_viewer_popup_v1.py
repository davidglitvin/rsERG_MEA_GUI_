"""
eyeavg_nan_viewer_popup_v1.py

Qt-based viewer for Eye-average NaN-cleaned pickles created by
manual_segment_eyeavg_click_popup_v10.py.

Assumes the pickle contains a 3D NumPy array with shape:
    (n_epochs, 2, epoch_len_samples)
where:
    channel 0 -> Eye1_avg
    channel 1 -> Eye2_avg

What it does
------------
- Loads the 3D array and flattens to continuous (2, n_times).
- Displays one Eye at a time (Eye1 or Eye2).
- Lets you scroll along time with LEFT/RIGHT arrows.
- Switch Eye with '1' / '2'.
- Shows NaNs as:
    * Gaps in the trace (Matplotlib breaks lines at NaNs).
    * Light grey shaded spans where the current Eye's data is NaN.
- Displays fraction of NaNs in the current window.

Keyboard controls
-----------------
- LEFT / RIGHT : scroll in time (¼ of current window).
- '1'          : switch to Eye1.
- '2'          : switch to Eye2.
- 'Q'          : quit.

Mouse
-----
- No editing here, just viewing.

Run from a terminal (NOT in Jupyter), e.g.

    python eyeavg_nan_viewer_popup_v1.py \
        "/path/to/1_C57_220_Isoflurane0p5per_up_epoched_EyeAvg_manualNaN.pkl" \
        --sfreq 2000
"""

import sys
import os
import pickle
from typing import Tuple, Dict, Optional

import numpy as np

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
    QDoubleSpinBox,
)

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# -------------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------------

def load_eyeavg_nan_pickle(
    file_path: str,
    sfreq: float = 2000.0,
) -> Tuple[np.ndarray, int]:
    """Load EyeAvg NaN-cleaned pickle.

    Expects:
        - A pickle containing a 3D NumPy array with shape
              (n_epochs, 2, n_times_epoch)

    Returns
    -------
    data_cont : ndarray, shape (2, n_times)
        Continuous eye-average data (Eye1, Eye2).
    n_times_epoch : int
        Number of samples per epoch.
    """
    with open(file_path, "rb") as f:
        arr = pickle.load(f)

    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(
            f"Expected 3D array in EyeAvg pickle, got shape {arr.shape}."
        )
    n_epochs, n_ch, n_times_epoch = arr.shape
    if n_ch != 2:
        raise ValueError(
            f"Expected 2 channels (Eye1, Eye2) in EyeAvg array, got {n_ch}."
        )

    # reshape to continuous (2, n_epochs * n_times_epoch)
    data_cont = arr.transpose(1, 0, 2).reshape(2, n_epochs * n_times_epoch)
    return data_cont.astype(float), n_times_epoch


# -------------------------------------------------------------------------
# Qt canvas
# -------------------------------------------------------------------------

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        fig = Figure(figsize=(8, 4))
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        fig.tight_layout()


# -------------------------------------------------------------------------
# Main window
# -------------------------------------------------------------------------

class EyeAvgNanViewer(QMainWindow, QtCore.QObject):
    def __init__(
        self,
        input_pkl_path: str,
        sfreq: float = 2000.0,
        default_window_s: float = 20.0,
    ):
        super().__init__()
        self.setWindowTitle("EyeAvg NaN Viewer")

        self.input_pkl_path = input_pkl_path
        self.sfreq = float(sfreq)

        # Load EyeAvg data (2, n_times)
        data_cont, n_times_epoch = load_eyeavg_nan_pickle(
            input_pkl_path,
            sfreq=self.sfreq,
        )
        self.data_cont = data_cont
        self.n_epochs = data_cont.shape[1] // n_times_epoch
        self.epoch_len_samples = n_times_epoch

        self.n_ch, self.n_times = self.data_cont.shape
        if self.n_ch != 2:
            raise RuntimeError(
                f"Internal error: expected 2 channels, got {self.n_ch}."
            )

        # State
        self.current_eye = 0  # 0 -> Eye1, 1 -> Eye2
        self.window_sec = float(default_window_s)
        self.center_sample = self.n_times // 2

        self._init_ui()
        self.update_plot()

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

        self.label_eye = QLabel("Eye: 1 / 2")
        ctrl_row1.addWidget(self.label_eye)

        ctrl_row1.addSpacing(20)
        self.label_nan = QLabel("NaN in window: 0.0%")
        ctrl_row1.addWidget(self.label_nan)

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

        instructions = (
            "Keyboard: ← / → to scroll, '1' or '2' to switch Eye, 'Q' to quit.\n"
            "NaNs show as gaps in the trace and light grey shaded spans."
        )
        self.label_help = QLabel(instructions)
        self.label_help.setWordWrap(True)
        layout.addWidget(self.label_help)

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
        y = self.data_cont[self.current_eye, start:end]

        # NaN mask for current eye & window
        nan_mask = np.isnan(y)

        # Shade NaN segments
        if nan_mask.any():
            idx = np.where(nan_mask)[0]
            # group contiguous indices
            seg_starts = [idx[0]]
            seg_ends = []
            for i in range(1, len(idx)):
                if idx[i] != idx[i - 1] + 1:
                    seg_ends.append(idx[i - 1])
                    seg_starts.append(idx[i])
            seg_ends.append(idx[-1])

            for s_i, e_i in zip(seg_starts, seg_ends):
                ts = t[s_i]
                te = t[e_i]
                ax.axvspan(ts, te, facecolor=(0.8, 0.8, 0.8, 0.5), edgecolor=None)

        # Plot the Eye trace (NaNs will appear as gaps)
        ax.plot(t, y, color="black", linewidth=0.8)

        # Update labels
        eye_label = "Eye 1" if self.current_eye == 0 else "Eye 2"
        self.label_eye.setText(f"{eye_label}")

        nan_frac = float(nan_mask.mean()) * 100.0 if len(y) > 0 else 0.0
        self.label_nan.setText(f"NaN in window: {nan_frac:.1f}%")

        center_sec = self.center_sample / self.sfreq
        self.label_pos.setText(f"Center: {center_sec:.3f} s")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Eye avg (µV)")
        ax.set_title(f"{eye_label} (NaNs = gaps + grey spans)")
        ax.grid(True, alpha=0.2)

        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def on_window_changed(self, val: float):
        self.window_sec = float(val)
        self.update_plot()


# -------------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python eyeavg_nan_viewer_popup_v1.py "
              "\"/path/to/EyeAvg_manualNaN.pkl\" [--sfreq 2000]")
        sys.exit(1)

    input_pkl_path = sys.argv[1]
    sfreq = 2000.0

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--sfreq" and i + 1 < len(args):
            sfreq = float(args[i + 1])
            i += 2
        else:
            i += 1

    if not os.path.isfile(input_pkl_path):
        print(f"ERROR: Input pickle not found: {input_pkl_path}")
        sys.exit(1)

    app = QApplication(sys.argv)
    win = EyeAvgNanViewer(
        input_pkl_path=input_pkl_path,
        sfreq=sfreq,
        default_window_s=20.0,
    )
    # Global key handling: arrows, 1/2, Q
    app.installEventFilter(win)
    win.resize(1000, 600)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
