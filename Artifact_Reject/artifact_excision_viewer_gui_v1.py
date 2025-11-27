
"""artifact_excision_viewer_gui_v1.py

Interactive Jupyter GUI to visualize artifact detections from
`artifact_excision_resegment_5s_v1`.

Features
--------
- Loads a single .pkl file (same formats supported by artifact_excision_resegment_5s_v1).
- Reconstructs continuous multichannel data.
- Computes artifact masks for criteria (i)–(iv):
    (i)   large amplitude |z| excursions
    (ii)  sharp derivative |z| excursions
    (iii) high low-frequency drift power (0.5–1 Hz)
    (iv)  broadband 1/f power surges (1/f intercept at 20 Hz)
- Provides an interactive GUI with:
    - Channel selector
    - Time window length and center sliders
    - Checkboxes to toggle each artifact type
- Overlays lightly colored highlights for each artifact class:
    amplitude  -> light magenta
    derivative -> light cyan
    drift      -> light green
    1/f        -> light yellow

Usage
-----
In a Jupyter notebook (with artifact_excision_resegment_5s_v1.py in your path):

    from artifact_excision_viewer_gui_v1 import launch_artifact_viewer

    launch_artifact_viewer(
        r"D:\Open Ephys\Isoflurane\C57 220\...\4_C57_220_Isoflurane2per_up_epoched.pkl",
        sfreq=2000.0,
        epoch_len_s=5.0,
        amp_z_hard=7.0,
        deriv_z_hard=7.0,
        drift_z_thresh=7.0,
        intercept_z_thresh=7.0,
    )

This will display the widget GUI below the cell.
"""

import os
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display, clear_output

# Import core functions from the artifact excision module
from artifact_excision_resegment_5s_v1 import (
    load_pickle_as_continuous,
    build_artifact_mask_i_to_iv,
)


def _find_segments(mask_1d: np.ndarray) -> Tuple[Tuple[int, int], ...]:
    """Return a tuple of (start, end) index pairs for contiguous True regions.

    Parameters
    ----------
    mask_1d : 1D bool array

    Returns
    -------
    segments : tuple of (start, end) pairs
        Indices are in [start, end) (end is exclusive).
    """
    mask_1d = np.asarray(mask_1d, dtype=bool)
    idx = np.where(mask_1d)[0]
    if idx.size == 0:
        return tuple()

    segments = []
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


def _build_gui(
    data_cont: np.ndarray,
    sfreq: float,
    component_masks: Dict[str, np.ndarray],
    ch_names,
) -> None:
    """Build and display the interactive artifact viewer GUI.

    Parameters
    ----------
    data_cont : ndarray, shape (n_channels, n_times)
        Continuous multichannel data.
    sfreq : float
        Sampling frequency (Hz).
    component_masks : dict
        Dictionary mapping keys 'amp', 'deriv', 'drift', 'intercept' to
        1D boolean arrays of length n_times.
    ch_names : list of str
        Channel names.
    """
    n_channels, n_times = data_cont.shape
    total_dur_s = n_times / sfreq
    times = np.arange(n_times) / sfreq

    # --- Widgets ---
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

    # Summary label with counts (seconds & fraction)
    n_total = n_times
    amp_count = int(component_masks["amp"].sum())
    deriv_count = int(component_masks["deriv"].sum())
    drift_count = int(component_masks["drift"].sum())
    intercept_count = int(component_masks["intercept"].sum())

    def _sec(n):
        return n / sfreq

    summary = widgets.HTML(
        value=(
            f"<b>Total duration:</b> {total_dur_s:.2f} s ({n_total} samples)<br>"
            f"Amplitude (i): {amp_count} samples (~{_sec(amp_count):.2f} s)<br>"
            f"Derivative (ii): {deriv_count} samples (~{_sec(deriv_count):.2f} s)<br>"
            f"Drift (iii): {drift_count} samples (~{_sec(drift_count):.2f} s)<br>"
            f"1/f intercept (iv): {intercept_count} samples (~{_sec(intercept_count):.2f} s)"
        )
    )

    # Output area for plot
    plot_output = widgets.Output()

    # Colors for each artifact type (light shades)
    color_map = {
        "amp": "#ffb3ff",        # light magenta
        "deriv": "#b3ffff",      # light cyan
        "drift": "#b3ffb3",      # light green
        "intercept": "#ffffb3",  # light yellow
    }

    def update_plot(*_):
        """Redraw the plot for current widget settings."""
        with plot_output:
            clear_output(wait=True)

            # Determine window in samples
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

            # Overlay artifacts
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
                full_mask = component_masks[key]
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
    controls = [
        ch_select,
        win_len_slider,
        center_slider,
        show_amp,
        show_deriv,
        show_drift,
        show_intercept,
    ]
    for w in controls:
        w.observe(update_plot, names="value")

    # Initial draw
    update_plot()

    # Layout
    controls_row1 = widgets.HBox([ch_select])
    controls_row2 = widgets.HBox([win_len_slider, center_slider])
    controls_row3 = widgets.HBox([show_amp, show_deriv, show_drift, show_intercept])

    gui = widgets.VBox(
        [
            widgets.HTML(value="<h3>Artifact Viewer (i–iv)</h3>"),
            summary,
            controls_row1,
            controls_row2,
            controls_row3,
            plot_output,
        ]
    )

    display(gui)


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
) -> None:
    """Launch the interactive artifact viewer for a given .pkl file.

    Parameters
    ----------
    input_pkl_path : str
        Path to the input pickle file (same formats as artifact_excision_resegment_5s_v1).
    sfreq : float
        Sampling frequency (Hz).
    epoch_len_s : float
        Epoch length used for spectral (drift / 1-f) windowing, in seconds (default 5).
    amp_z_hard : float
        Hard z-threshold for amplitude excursions (criterion i).
    deriv_z_hard : float
        Hard z-threshold for derivative excursions (criterion ii).
    drift_z_thresh : float
        Robust z-threshold for drift power (criterion iii).
    intercept_z_thresh : float
        Robust z-threshold for 1/f intercept (criterion iv).
    min_dur_ms_amp : float
        Minimum duration for amplitude excursions, in ms.
    min_dur_ms_deriv : float
        Minimum duration for derivative excursions, in ms.

    Notes
    -----
    - This function does not excise or filter the data; it only visualizes
      where each artifact criterion would flag samples.
    - Use `clean_and_export(...)` from artifact_excision_resegment_5s_v1.py
      to actually excise and re-epoch + filter, once you are happy with
      the parameter settings.
    """
    if not os.path.isfile(input_pkl_path):
        raise FileNotFoundError(f"Input pickle file not found: {input_pkl_path}")

    # Load and build continuous data
    data_cont, epoch_len_samples, ch_names = load_pickle_as_continuous(
        input_pkl_path,
        sfreq=sfreq,
        epoch_len_s=epoch_len_s,
    )

    # Compute artifact masks (i–iv)
    artifact_mask, component_masks = build_artifact_mask_i_to_iv(
        data_cont,
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
    _build_gui(data_cont, sfreq, component_masks, ch_names)
