"""artifact_excision_eyeavg_viewer_v1.py

Interactive Jupyter GUI to visualize and tune artifact detections from
`artifact_excision_resegment_5s_v1`, with optional Eye1/Eye2 averaging.

Key features
------------
- Uses the same artifact metrics (i–iv) as your excision pipeline:
    (i)   amplitude |z|
    (ii)  derivative |z|
    (iii) low-frequency drift power
    (iv)  broadband 1/f intercept
- Interactive sliders for:
    - amp_z_hard, deriv_z_hard, drift_z_thresh, intercept_z_thresh
    - min_dur_ms_amp, min_dur_ms_deriv
    - per-type padding (ms) around detected segments (i–iv)
- Color legend:
    - amplitude  -> light magenta
    - derivative -> light cyan
    - drift      -> light green
    - 1/f        -> light yellow
- NEW: optional averaging of Eye1 (ch 0–7) and Eye2 (ch 8–15) into 2 synthetic
  channels, with the ability to drop specific channels before averaging.

Typical usage
-------------
In a Jupyter notebook (with artifact_excision_resegment_5s_v1.py on your PYTHONPATH):

    from artifact_excision_eyeavg_viewer_v1 import launch_artifact_viewer

    input_pkl = r"D:\Open Ephys\Isoflurane\C57 220\20251114\Analysis\MNE Array\1_C57_220_Isoflurane0p5per_up_epoched.pkl"

    # Eye-averaged view with channel dropping:
    launch_artifact_viewer(
        input_pkl_path=input_pkl,
        sfreq=2000.0,
        epoch_len_s=5.0,
        amp_z_hard=6.0,
        deriv_z_hard=6.0,
        drift_z_thresh=6.0,
        intercept_z_thresh=7.0,
        min_dur_ms_amp=5.0,
        min_dur_ms_deriv=5.0,
        average_eye_groups=True,
        drop_indices_eye1=[0, 1, 2],   # drop channels 0,1,2 before averaging Eye1
        # drop_indices_eye2=[...]      # optional, analogous for Eye2
    )

You will then see a channel dropdown with:
    - Eye1_avg (idx 0)
    - Eye2_avg (idx 1)
and can slide a window across time to see which regions each criterion (i–iv)
would mark for excision, with adjustable padding.
"""

import os
from typing import Dict, Tuple, Sequence, Optional, List

import numpy as np
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display, clear_output

# Import core functions from the artifact excision module
from artifact_excision_resegment_5s_v1 import (
    load_pickle_as_continuous,
    build_artifact_mask_i_to_iv,
)


# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------
def _find_segments(mask_1d: np.ndarray) -> Tuple[Tuple[int, int], ...]:
    """Return (start, end) pairs for contiguous True regions in a 1D bool mask."""
    mask_1d = np.asarray(mask_1d, dtype=bool)
    idx = np.where(mask_1d)[0]
    if idx.size == 0:
        return tuple()

    segments: List[Tuple[int, int]] = []
    start = idx[0]
    prev = idx[0]

    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            segments.append((start, prev + 1))
            start = i
            prev = i

    segments.append((start, prev + 1))
    return tuple(segments)


def _pad_mask(mask_1d: np.ndarray, sfreq: float, pad_ms: float) -> np.ndarray:
    """Pad a boolean mask in time by a fixed duration (ms) before/after each run."""
    mask_1d = np.asarray(mask_1d, dtype=bool)
    n_times = mask_1d.size
    pad_samples = int(round(pad_ms * sfreq / 1000.0))
    if pad_samples <= 0 or not mask_1d.any():
        return mask_1d.copy()

    padded = np.zeros_like(mask_1d, dtype=bool)
    segments = _find_segments(mask_1d)
    for start, end in segments:
        s = max(0, start - pad_samples)
        e = min(n_times, end + pad_samples)
        padded[s:e] = True
    return padded


def _compute_eye_averages(
    data_cont: np.ndarray,
    ch_names: Sequence[str],
    eye1_indices: Sequence[int],
    eye2_indices: Sequence[int],
    drop_indices_eye1: Optional[Sequence[int]] = None,
    drop_indices_eye2: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Compute Eye1/Eye2 average channels with optional dropping.

    Parameters
    ----------
    data_cont : ndarray, shape (n_channels, n_times)
        Original continuous data.
    ch_names : sequence of str
        Original channel names.
    eye1_indices : sequence of int
        Indices for Eye1 group (e.g., 0–7).
    eye2_indices : sequence of int
        Indices for Eye2 group (e.g., 8–15).
    drop_indices_eye1 : sequence of int | None
        Channel indices (in original space) to drop from Eye1 before averaging.
    drop_indices_eye2 : sequence of int | None
        Channel indices (in original space) to drop from Eye2 before averaging.

    Returns
    -------
    data_avg : ndarray, shape (2, n_times)
        data_avg[0] = average of Eye1 channels, data_avg[1] = average of Eye2 channels.
    ch_names_avg : list of str
        Names of the synthetic channels: ["Eye1_avg", "Eye2_avg"].
    """
    n_channels, n_times = data_cont.shape

    eye1 = np.array(eye1_indices, dtype=int)
    eye2 = np.array(eye2_indices, dtype=int)

    # Restrict to valid indices
    eye1 = eye1[(eye1 >= 0) & (eye1 < n_channels)]
    eye2 = eye2[(eye2 >= 0) & (eye2 < n_channels)]

    # Drop requested indices
    if drop_indices_eye1 is not None:
        drop1 = np.array(list(drop_indices_eye1), dtype=int)
        eye1 = np.array([idx for idx in eye1 if idx not in drop1])
    if drop_indices_eye2 is not None:
        drop2 = np.array(list(drop_indices_eye2), dtype=int)
        eye2 = np.array([idx for idx in eye2 if idx not in drop2])

    if eye1.size == 0:
        raise ValueError("Eye1 group has no channels left after dropping.")
    if eye2.size == 0:
        raise ValueError("Eye2 group has no channels left after dropping.")

    eye1_avg = data_cont[eye1].mean(axis=0)
    eye2_avg = data_cont[eye2].mean(axis=0)

    data_avg = np.vstack([eye1_avg, eye2_avg])
    ch_names_avg = ["Eye1_avg", "Eye2_avg"]

    print("Eye averaging configuration:")
    print(f"  Eye1 indices used: {eye1.tolist()}")
    print(f"  Eye2 indices used: {eye2.tolist()}")

    return data_avg, ch_names_avg


# -------------------------------------------------------------------------
# GUI builder
# -------------------------------------------------------------------------
def _build_gui(
    data_cont: np.ndarray,
    sfreq: float,
    epoch_len_samples: int,
    component_masks_init: Dict[str, np.ndarray],
    ch_names: List[str],
    amp_z_hard_init: float,
    deriv_z_hard_init: float,
    drift_z_thresh_init: float,
    intercept_z_thresh_init: float,
    min_dur_ms_amp_init: float,
    min_dur_ms_deriv_init: float,
) -> None:
    """Build and display the interactive artifact viewer GUI."""
    n_channels, n_times = data_cont.shape
    total_dur_s = n_times / sfreq
    times = np.arange(n_times) / sfreq

    # Colors for each artifact type (light shades)
    color_map = {
        "amp": "#ffb3ff",        # light magenta
        "deriv": "#b3ffff",      # light cyan
        "drift": "#b3ffb3",      # light green
        "intercept": "#ffffb3",  # light yellow
    }

    # State: current (possibly padded) masks
    component_masks_raw: Dict[str, np.ndarray] = {
        k: v.copy() for k, v in component_masks_init.items()
    }
    component_masks_padded: Dict[str, np.ndarray] = {
        k: v.copy() for k, v in component_masks_raw.items()
    }

    # --- Widgets: channel & window controls ---
    ch_options = [(f"{ch_names[i]} (idx {i})", i) for i in range(n_channels)]
    ch_select = widgets.Dropdown(
        options=ch_options,
        value=0,
        description="Channel:",
        layout=widgets.Layout(width="280px"),
    )

    win_len_slider = widgets.FloatSlider(
        value=5.0,
        min=0.5,
        max=min(30.0, total_dur_s),
        step=0.5,
        description="Window (s):",
        readout_format=".1f",
        continuous_update=False,
        layout=widgets.Layout(width="400px"),
    )

    center_slider = widgets.FloatSlider(
        value=min(5.0, total_dur_s / 2.0),
        min=0.0,
        max=total_dur_s,
        step=0.5,
        description="Center (s):",
        readout_format=".1f",
        continuous_update=False,
        layout=widgets.Layout(width="400px"),
    )

    show_amp = widgets.Checkbox(
        value=True, description="Amplitude (i)", indent=False
    )
    show_deriv = widgets.Checkbox(
        value=True, description="Derivative (ii)", indent=False
    )
    show_drift = widgets.Checkbox(
        value=True, description="Drift (iii)", indent=False
    )
    show_intercept = widgets.Checkbox(
        value=True, description="1/f intercept (iv)", indent=False
    )

    # --- Widgets: thresholds & durations (editable) ---
    amp_thresh_slider = widgets.FloatSlider(
        value=float(amp_z_hard_init),
        min=3.0,
        max=20.0,
        step=0.5,
        description="amp z≥",
        continuous_update=False,
        readout_format=".1f",
        layout=widgets.Layout(width="250px"),
    )
    deriv_thresh_slider = widgets.FloatSlider(
        value=float(deriv_z_hard_init),
        min=3.0,
        max=20.0,
        step=0.5,
        description="deriv z≥",
        continuous_update=False,
        readout_format=".1f",
        layout=widgets.Layout(width="250px"),
    )
    drift_thresh_slider = widgets.FloatSlider(
        value=float(drift_z_thresh_init),
        min=3.0,
        max=20.0,
        step=0.5,
        description="drift z≥",
        continuous_update=False,
        readout_format=".1f",
        layout=widgets.Layout(width="250px"),
    )
    intercept_thresh_slider = widgets.FloatSlider(
        value=float(intercept_z_thresh_init),
        min=3.0,
        max=20.0,
        step=0.5,
        description="1/f z≥",
        continuous_update=False,
        readout_format=".1f",
        layout=widgets.Layout(width="250px"),
    )

    min_dur_amp_slider = widgets.FloatSlider(
        value=float(min_dur_ms_amp_init),
        min=1.0,
        max=50.0,
        step=1.0,
        description="amp min ms",
        continuous_update=False,
        readout_format=".0f",
        layout=widgets.Layout(width="250px"),
    )
    min_dur_deriv_slider = widgets.FloatSlider(
        value=float(min_dur_ms_deriv_init),
        min=1.0,
        max=50.0,
        step=1.0,
        description="deriv min ms",
        continuous_update=False,
        readout_format=".0f",
        layout=widgets.Layout(width="250px"),
    )

    # --- Widgets: per-type padding (ms) ---
    pad_amp_slider = widgets.FloatSlider(
        value=0.0,
        min=0.0,
        max=100.0,
        step=2.0,
        description="amp pad ms",
        continuous_update=False,
        readout_format=".0f",
        layout=widgets.Layout(width="250px"),
    )
    pad_deriv_slider = widgets.FloatSlider(
        value=0.0,
        min=0.0,
        max=100.0,
        step=2.0,
        description="deriv pad ms",
        continuous_update=False,
        readout_format=".0f",
        layout=widgets.Layout(width="250px"),
    )
    pad_drift_slider = widgets.FloatSlider(
        value=0.0,
        min=0.0,
        max=500.0,
        step=10.0,
        description="drift pad ms",
        continuous_update=False,
        readout_format=".0f",
        layout=widgets.Layout(width="250px"),
    )
    pad_intercept_slider = widgets.FloatSlider(
        value=0.0,
        min=0.0,
        max=500.0,
        step=10.0,
        description="1/f pad ms",
        continuous_update=False,
        readout_format=".0f",
        layout=widgets.Layout(width="250px"),
    )

    # Summary label with counts (seconds)
    n_total = n_times

    def _sec(n):
        return n / sfreq

    summary = widgets.HTML()

    # Color legend widget
    legend_html = widgets.HTML(
        value=(
            "<b>Artifact color legend (padded regions that would be excised):</b><br>"
            f"<span style='background-color:{color_map['amp']};"
            " padding:2px 12px; margin-right:8px;'>Amplitude (i)</span>"
            f"<span style='background-color:{color_map['deriv']};"
            " padding:2px 12px; margin-right:8px;'>Derivative (ii)</span><br>"
            f"<span style='background-color:{color_map['drift']};"
            " padding:2px 12px; margin-right:8px;'>Drift (iii)</span>"
            f"<span style='background-color:{color_map['intercept']};"
            " padding:2px 12px; margin-right:8px;'>1/f intercept (iv)</span>"
        )
    )

    plot_output = widgets.Output()

    def recompute_masks(*_):
        """Recompute artifact masks based on current threshold / duration / padding settings."""
        nonlocal component_masks_raw, component_masks_padded

        amp_thr = float(amp_thresh_slider.value)
        deriv_thr = float(deriv_thresh_slider.value)
        drift_thr = float(drift_thresh_slider.value)
        intercept_thr = float(intercept_thresh_slider.value)
        min_amp_ms = float(min_dur_amp_slider.value)
        min_deriv_ms = float(min_dur_deriv_slider.value)

        # Recompute raw masks using core detection function
        _, cm = build_artifact_mask_i_to_iv(
            data_cont,
            sfreq=sfreq,
            epoch_len_samples=epoch_len_samples,
            amp_z_hard=amp_thr,
            deriv_z_hard=deriv_thr,
            drift_z_thresh=drift_thr,
            intercept_z_thresh=intercept_thr,
            min_dur_ms_amp=min_amp_ms,
            min_dur_ms_deriv=min_deriv_ms,
        )
        component_masks_raw = {k: v.copy() for k, v in cm.items()}

        # Apply per-type padding
        component_masks_padded = {
            "amp": _pad_mask(component_masks_raw["amp"], sfreq, float(pad_amp_slider.value)),
            "deriv": _pad_mask(component_masks_raw["deriv"], sfreq, float(pad_deriv_slider.value)),
            "drift": _pad_mask(component_masks_raw["drift"], sfreq, float(pad_drift_slider.value)),
            "intercept": _pad_mask(component_masks_raw["intercept"], sfreq, float(pad_intercept_slider.value)),
        }

        # Update summary (padded counts)
        amp_count = int(component_masks_padded["amp"].sum())
        deriv_count = int(component_masks_padded["deriv"].sum())
        drift_count = int(component_masks_padded["drift"].sum())
        intercept_count = int(component_masks_padded["intercept"].sum())

        summary.value = (
            f"<b>Total duration:</b> {total_dur_s:.2f} s ({n_total} samples)<br>"
            f"Amplitude (i) padded: {amp_count} samples (~{_sec(amp_count):.2f} s)<br>"
            f"Derivative (ii) padded: {deriv_count} samples (~{_sec(deriv_count):.2f} s)<br>"
            f"Drift (iii) padded: {drift_count} samples (~{_sec(drift_count):.2f} s)<br>"
            f"1/f intercept (iv) padded: {intercept_count} samples (~{_sec(intercept_count):.2f} s)"
        )

        update_plot()

    def update_plot(*_):
        """Redraw the plot for current widget settings."""
        with plot_output:
            clear_output(wait=True)

            win_len = max(0.1, float(win_len_slider.value))
            center = float(center_slider.value)
            t0 = max(0.0, center - win_len / 2.0)
            t1 = min(total_dur_s, center + win_len / 2.0)
            if t1 <= t0:
                t1 = min(total_dur_s, t0 + 0.1)

            i0 = int(round(t0 * sfreq))
            i1 = int(round(t1 * sfreq))
            if i1 <= i0:
                i1 = min(i0 + 1, n_times)

            t_seg = times[i0:i1]
            ch_idx = int(ch_select.value)
            y = data_cont[ch_idx, i0:i1]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(t_seg, y, linewidth=0.8)

            mask_flags = []
            if show_amp.value:
                mask_flags.append("amp")
            if show_deriv.value:
                mask_flags.append("deriv")
            if show_drift.value:
                mask_flags.append("drift")
            if show_intercept.value:
                mask_flags.append("intercept")

            for key in mask_flags:
                full_mask = component_masks_padded[key]
                seg_mask = full_mask[i0:i1]
                segments = _find_segments(seg_mask)
                for s, e in segments:
                    ax.axvspan(
                        t_seg[s],
                        t_seg[e - 1],
                        color=color_map[key],
                        alpha=0.4,
                        linewidth=0,
                    )

            ax.set_xlim(t_seg[0], t_seg[-1])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (a.u.)")
            ax.set_title(
                f"Channel {ch_names[ch_idx]} | "
                f"Window {t0:.2f}–{t1:.2f} s (len {win_len:.2f} s)"
            )
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    # Wire up callbacks
    controls_plot_only = [
        ch_select,
        win_len_slider,
        center_slider,
        show_amp,
        show_deriv,
        show_drift,
        show_intercept,
    ]
    for w in controls_plot_only:
        w.observe(update_plot, names="value")

    controls_recompute = [
        amp_thresh_slider,
        deriv_thresh_slider,
        drift_thresh_slider,
        intercept_thresh_slider,
        min_dur_amp_slider,
        min_dur_deriv_slider,
        pad_amp_slider,
        pad_deriv_slider,
        pad_drift_slider,
        pad_intercept_slider,
    ]
    for w in controls_recompute:
        w.observe(recompute_masks, names="value")

    # Initial summary + plot
    recompute_masks()

    # Layout
    controls_row1 = widgets.HBox([ch_select])
    controls_row2 = widgets.HBox([win_len_slider, center_slider])
    controls_row3 = widgets.HBox([show_amp, show_deriv, show_drift, show_intercept])

    thresh_row1 = widgets.HBox([amp_thresh_slider, deriv_thresh_slider])
    thresh_row2 = widgets.HBox([drift_thresh_slider, intercept_thresh_slider])
    dur_row = widgets.HBox([min_dur_amp_slider, min_dur_deriv_slider])
    pad_row1 = widgets.HBox([pad_amp_slider, pad_deriv_slider])
    pad_row2 = widgets.HBox([pad_drift_slider, pad_intercept_slider])

    gui = widgets.VBox(
        [
            widgets.HTML(
                value="<h3>Artifact Viewer (i–iv) with Threshold, Padding, and Eye Averaging</h3>"
            ),
            legend_html,
            summary,
            controls_row1,
            controls_row2,
            controls_row3,
            widgets.HTML(value="<b>Detection thresholds & durations:</b>"),
            thresh_row1,
            thresh_row2,
            dur_row,
            widgets.HTML(value="<b>Per-type padding (ms) around detected segments:</b>"),
            pad_row1,
            pad_row2,
            plot_output,
        ]
    )

    display(gui)


# -------------------------------------------------------------------------
# Public entry point
# -------------------------------------------------------------------------
def launch_artifact_viewer(
    input_pkl_path: str,
    sfreq: float = 2000.0,
    epoch_len_s: float = 5.0,
    amp_z_hard: float = 8.0,
    deriv_z_hard: float = 8.0,
    drift_z_thresh: float = 8.0,
    intercept_z_thresh: float = 8.0,
    min_dur_ms_amp: float = 10.0,
    min_dur_ms_deriv: float = 8.0,
    *,
    average_eye_groups: bool = False,
    eye1_indices: Sequence[int] = tuple(range(0, 8)),
    eye2_indices: Sequence[int] = tuple(range(8, 16)),
    drop_indices_eye1: Optional[Sequence[int]] = None,
    drop_indices_eye2: Optional[Sequence[int]] = None,
) -> None:
    """Launch the interactive artifact viewer for a given .pkl file.

    Parameters
    ----------
    input_pkl_path : str
        Path to the input pickle file.
    sfreq : float
        Sampling frequency (Hz).
    epoch_len_s : float
        Epoch length used for spectral (drift / 1-f) windowing, in seconds.
    amp_z_hard, deriv_z_hard, drift_z_thresh, intercept_z_thresh : float
        Initial thresholds for criteria (i–iv).
    min_dur_ms_amp, min_dur_ms_deriv : float
        Initial minimum durations (ms) for criteria (i) and (ii).
    average_eye_groups : bool
        If True, average Eye1 (eye1_indices) and Eye2 (eye2_indices) into
        two synthetic channels before artifact detection and plotting.
    eye1_indices, eye2_indices : sequence of int
        Original channel indices for Eye1 and Eye2.
    drop_indices_eye1, drop_indices_eye2 : sequence of int | None
        Original channel indices to drop from Eye1/Eye2 before averaging.
    """
    if not os.path.isfile(input_pkl_path):
        raise FileNotFoundError(f"Input pickle file not found: {input_pkl_path}")

    # Load and build continuous data (n_channels, n_times)
    data_cont, epoch_len_samples, ch_names = load_pickle_as_continuous(
        input_pkl_path,
        sfreq=sfreq,
        epoch_len_s=epoch_len_s,
    )

    # Optionally compute Eye1/Eye2 averages (and drop bad channels)
    if average_eye_groups:
        data_for_view, ch_names_for_view = _compute_eye_averages(
            data_cont=data_cont,
            ch_names=ch_names,
            eye1_indices=eye1_indices,
            eye2_indices=eye2_indices,
            drop_indices_eye1=drop_indices_eye1,
            drop_indices_eye2=drop_indices_eye2,
        )
    else:
        data_for_view = data_cont
        ch_names_for_view = list(ch_names)

    # Compute initial artifact masks (i–iv) on the data we are viewing
    _, component_masks = build_artifact_mask_i_to_iv(
        data_for_view,
        sfreq=sfreq,
        epoch_len_samples=epoch_len_samples,
        amp_z_hard=amp_z_hard,
        deriv_z_hard=deriv_z_hard,
        drift_z_thresh=drift_z_thresh,
        intercept_z_thresh=intercept_z_thresh,
        min_dur_ms_amp=min_dur_ms_amp,
        min_dur_ms_deriv=min_dur_ms_deriv,
    )

    # Build and display GUI
    _build_gui(
        data_for_view,
        sfreq,
        epoch_len_samples,
        component_masks,
        ch_names_for_view,
        amp_z_hard,
        deriv_z_hard,
        drift_z_thresh,
        intercept_z_thresh,
        min_dur_ms_amp,
        min_dur_ms_deriv,
    )
