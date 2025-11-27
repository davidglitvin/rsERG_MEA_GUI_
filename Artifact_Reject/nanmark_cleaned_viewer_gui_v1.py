"""nanmark_cleaned_viewer_gui_v1.py

Interactive Jupyter GUI to visualize the time series *after* NaN-based
artifact marking, using the same artifact criteria (i–iv) as in
`artifact_excision_resegment_5s_v1.py`.

This is intended to be paired with the NaN-marking pipeline
`artifact_excision_nanmark_5s_v1.py`, but it runs the cleaning logic
internally so you can:

    - Load a .pkl file.
    - Detect artifacts (i–iv) using the same thresholds.
    - (Optionally) band-pass filter the continuous data.
    - Replace artifact samples with NaNs (preserving temporal structure).
    - Inspect the cleaned (NaN-marked) continuous data with a sliding window.

NaN segments appear as breaks in the plotted line, so you can visually
confirm where the artifact mask removed data and whether the coverage
is sufficient.

Usage
-----
In a Jupyter notebook, with the relevant modules in your Python path:

    from nanmark_cleaned_viewer_gui_v1 import launch_nan_cleaned_viewer

    launch_nan_cleaned_viewer(
        input_pkl_path=r"...your_file.pkl",
        sfreq=2000.0,
        epoch_len_s=5.0,
        amp_z_hard=7.0,
        deriv_z_hard=7.0,
        drift_z_thresh=7.0,
        intercept_z_thresh=7.0,
        min_dur_ms_amp=10.0,
        min_dur_ms_deriv=8.0,
        l_freq=0.5,
        h_freq=100.0,
    )

This will show:
    - Original (filtered) trace in light gray.
    - Cleaned, NaN-marked trace in blue (with gaps where NaNs are).
    - Channel dropdown and sliding center/window controls.
"""

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display, clear_output

import mne

from artifact_excision_resegment_5s_v1 import (  # type: ignore
    load_pickle_as_continuous,
    build_artifact_mask_i_to_iv,
)


def _build_gui(
    data_orig: np.ndarray,
    data_clean: np.ndarray,
    artifact_mask: np.ndarray,
    sfreq: float,
    epoch_len_samples: int,
    ch_names,
) -> None:
    """Build and display the GUI for visualizing original vs NaN-cleaned data.

    Parameters
    ----------
    data_orig : ndarray, shape (n_channels, n_times)
        Original (optionally filtered) continuous data, no NaNs.
    data_clean : ndarray, shape (n_channels, n_times)
        Same as data_orig but with artifact samples replaced by NaN.
    artifact_mask : ndarray, shape (n_times,)
        Boolean mask where True indicates an artifact sample (any criterion).
    sfreq : float
        Sampling frequency.
    epoch_len_samples : int
        Epoch length in samples (e.g. 5 s * sfreq); used for reference only.
    ch_names : list of str
        Channel names.
    """
    n_channels, n_times = data_orig.shape
    total_dur_s = n_times / sfreq
    times = np.arange(n_times) / sfreq

    # Widgets: channel & window controls
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
        max=min(60.0, total_dur_s),
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

    show_orig = widgets.Checkbox(
        value=True, description="Show original (filtered)", indent=False
    )
    show_clean = widgets.Checkbox(
        value=True, description="Show cleaned (NaN)", indent=False
    )

    summary = widgets.HTML()
    plot_output = widgets.Output()

    # Summary info
    n_artifact = int(artifact_mask.sum())
    n_good = int((~artifact_mask).sum())

    def _sec(count):
        return count / sfreq

    summary.value = (
        f"<b>Total duration:</b> {total_dur_s:.2f} s ({n_times} samples)<br>"
        f"Artifact samples (any criterion): {n_artifact} "
        f"(~{_sec(n_artifact):.2f} s; {n_artifact/n_times:.2%} of time)<br>"
        f"Non-artifact samples: {n_good} "
        f"(~{_sec(n_good):.2f} s; {n_good/n_times:.2%} of time)"
    )

    def update_plot(*_):
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

            y_orig = data_orig[ch_idx, i0:i1]
            y_clean = data_clean[ch_idx, i0:i1]

            fig, ax = plt.subplots(figsize=(10, 4))

            if show_orig.value:
                ax.plot(
                    t_seg,
                    y_orig,
                    linewidth=0.8,
                    color="0.7",
                    label="Original (filtered)",
                )

            if show_clean.value:
                ax.plot(
                    t_seg,
                    y_clean,
                    linewidth=1.0,
                    color="C0",
                    label="Cleaned (NaN-marked)",
                )

            # Optionally, highlight artifact samples as light background (for context)
            seg_mask = artifact_mask[i0:i1]
            if seg_mask.any():
                idx = np.where(seg_mask)[0]
                # Merge contiguous indices into spans
                start = idx[0]
                prev = idx[0]
                spans = []
                for k in idx[1:]:
                    if k == prev + 1:
                        prev = k
                    else:
                        spans.append((start, prev + 1))
                        start = k
                        prev = k
                spans.append((start, prev + 1))
                for s_idx, e_idx in spans:
                    ax.axvspan(
                        t_seg[s_idx],
                        t_seg[e_idx - 1],
                        color="red",
                        alpha=0.1,
                        linewidth=0,
                    )

            ax.set_ylabel("Amplitude")
            ax.set_xlabel("Time (s)")
            ax.set_title(
                f"Channel {ch_names[ch_idx]} | {t0:.2f}–{t1:.2f} s "
                f"(window {win_len:.2f} s)"
            )
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)

            plt.tight_layout()
            plt.show()

    # Wire up callbacks
    for w in [ch_select, win_len_slider, center_slider, show_orig, show_clean]:
        w.observe(update_plot, names="value")

    # Initial draw
    update_plot()

    controls_row1 = widgets.HBox([ch_select])
    controls_row2 = widgets.HBox([win_len_slider, center_slider])
    controls_row3 = widgets.HBox([show_orig, show_clean])

    gui = widgets.VBox(
        [
            widgets.HTML(
                value="<h3>NaN-marked Cleaned Time Series Viewer</h3>"
            ),
            summary,
            controls_row1,
            controls_row2,
            controls_row3,
            plot_output,
        ]
    )

    display(gui)


def launch_nan_cleaned_viewer(
    input_pkl_path: str,
    sfreq: float = 2000.0,
    epoch_len_s: float = 5.0,
    amp_z_hard: float = 8.0,
    deriv_z_hard: float = 8.0,
    drift_z_thresh: float = 8.0,
    intercept_z_thresh: float = 8.0,
    min_dur_ms_amp: float = 10.0,
    min_dur_ms_deriv: float = 8.0,
    l_freq: Optional[float] = 0.5,
    h_freq: Optional[float] = 100.0,
    fir_design: str = "firwin",
    filter_length: str = "auto",
    verbose: bool = True,
) -> None:
    """Run NaN-marking cleanup and launch an interactive viewer.

    Parameters
    ----------
    input_pkl_path : str
        Path to the input pickle file.
    sfreq : float
        Sampling frequency (Hz).
    epoch_len_s : float
        Epoch length used for spectral (drift / 1-f) criteria, in seconds (default 5).
    amp_z_hard : float
        Hard z-threshold for amplitude excursions (criterion i).
    deriv_z_hard : float
        Hard z-threshold for derivative excursions (criterion ii).
    drift_z_thresh : float
        Robust z-threshold for drift power (criterion iii).
    intercept_z_thresh : float
        Robust z-threshold for 1/f intercept (criterion iv).
    min_dur_ms_amp : float
        Minimum duration in ms for amplitude excursions.
    min_dur_ms_deriv : float
        Minimum duration in ms for derivative excursions.
    l_freq : float | None
        Lower cutoff for band-pass filter (Hz). If None, no high-pass.
    h_freq : float | None
        Upper cutoff for band-pass filter (Hz). If None, no low-pass.
    fir_design : str
        FIR design method for MNE filter (e.g., "firwin").
    filter_length : str | int
        Filter length, e.g. "auto" or an integer number of taps.
    verbose : bool
        If True, prints progress information.

    Notes
    -----
    - This does *not* write any FIF file; it only runs the NaN-marking step
      in memory and opens the interactive viewer.
    """
    if not os.path.isfile(input_pkl_path):
        raise FileNotFoundError(f"Input pickle file not found: {input_pkl_path}")

    if verbose:
        print(f"[launch_nan_cleaned_viewer] Loading pickle: {input_pkl_path}")

    data_cont, epoch_len_samples, ch_names = load_pickle_as_continuous(
        input_pkl_path,
        sfreq=sfreq,
        epoch_len_s=epoch_len_s,
    )
    # Ensure floating type for filtering
    data_cont = data_cont.astype(np.float64, copy=False)

    if verbose:
        n_channels, n_times = data_cont.shape
        print(f"  Continuous data shape: (n_channels={n_channels}, n_times={n_times})")
        print(f"  Total duration: {n_times / sfreq:.2f} s")
        print(f"  Epoch length (for spectral criteria): {epoch_len_samples} samples "
              f"({epoch_len_samples / sfreq:.2f} s)")

    # Build artifact mask (i–iv)
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

    n_times = data_cont.shape[1]
    n_artifact = int(artifact_mask.sum())
    n_good = int((~artifact_mask).sum())

    if verbose:
        def _sec(count): return count / sfreq
        print("Artifact counts per criterion (samples, not mutually exclusive):")
        for key in ["amp", "deriv", "drift", "intercept"]:
            c = int(component_masks[key].sum())
            print(f"  {key:9s}: {c:7d} samples (~{_sec(c):7.2f} s)")
        print(f"Combined (union): {n_artifact} samples (~{_sec(n_artifact):.2f} s)")
        print(f"Remaining clean (non-artifact):  {n_good} samples (~{_sec(n_good):.2f} s)")
        print(f"Fraction marked as artifact: {n_artifact / n_times:.2%}")

    # Filter continuous data (no NaNs yet)
    data_filt = data_cont.copy()
    if l_freq is not None or h_freq is not None:
        if verbose:
            print(f"Applying band-pass filter before NaN marking: "
                  f"{l_freq}-{h_freq} Hz (fir_design={fir_design}, "
                  f"filter_length={filter_length})")
        data_filt = mne.filter.filter_data(
            data_filt,
            sfreq=sfreq,
            l_freq=l_freq,
            h_freq=h_freq,
            fir_design=fir_design,
            filter_length=filter_length,
            verbose=verbose,
        )

    # Apply NaNs at artifact positions
    data_clean = data_filt.copy()
    data_clean[:, artifact_mask] = np.nan

    if verbose:
        print("Launching interactive viewer for NaN-marked cleaned time series...")

    _build_gui(
        data_orig=data_filt,
        data_clean=data_clean,
        artifact_mask=artifact_mask,
        sfreq=sfreq,
        epoch_len_samples=epoch_len_samples,
        ch_names=ch_names,
    )
