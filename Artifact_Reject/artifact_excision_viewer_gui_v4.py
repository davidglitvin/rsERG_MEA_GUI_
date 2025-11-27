"""artifact_excision_viewer_gui_v3.py

Interactive Jupyter GUI to visualize and tune artifact detections from
`artifact_excision_resegment_5s_v1`, with additional panels that track
the 1/f slope (aperiodic exponent) in 5 s windows and show the PSD + 1/f
fit used to compute that slope.

Features
--------
- Timeseries view with shaded artifact regions (criteria i–iv).
- 1/f slope panel (per 5 s window, expanded per sample) for the selected channel.
- PSD + 1/f fit panel for the 5 s window that contains the current time-center:
    - Shows PSD for that window & channel.
    - Overlays the aperiodic 1/f fit line (2–40 Hz log–log fit).
- Real-time editable thresholds and durations for criteria (i)–(iv).
- Per-type padding (ms) to extend excised regions.
- Color legend for artifact types.

Usage
-----
In a Jupyter notebook (with artifact_excision_resegment_5s_v1.py in your path):

    from artifact_excision_viewer_gui_v3 import launch_artifact_viewer

    launch_artifact_viewer(
        r"D:\Open Ephys\Isoflurane\C57 220\...\4_C57_220_Isoflurane2per_up_epoched.pkl",
        sfreq=2000.0,
        epoch_len_s=5.0,
        amp_z_hard=7.0,
        deriv_z_hard=7.0,
        drift_z_thresh=7.0,
        intercept_z_thresh=7.0,
    )

This will display the widget GUI below the cell. You can then adjust thresholds
and padding and see the artifact regions, the 1/f slope trajectory, and the
PSD + 1/f fit for successive 5 s windows as you move the time-center slider.
"""

import os
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display, clear_output

from mne.time_frequency import psd_array_welch

# Import core functions from the artifact excision module
from artifact_excision_resegment_5s_v1 import (
    load_pickle_as_continuous,
    build_artifact_mask_i_to_iv,
)

# Constants for the aperiodic fit band
FMIN_FIT = 2.0
FMAX_FIT = 40.0


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


def _pad_mask(mask_1d: np.ndarray, sfreq: float, pad_ms: float) -> np.ndarray:
    """Pad a boolean mask in time by a fixed duration (ms) before/after each run.

    Parameters
    ----------
    mask_1d : 1D bool array, shape (n_times,)
        Base artifact mask for a given criterion.
    sfreq : float
        Sampling frequency (Hz).
    pad_ms : float
        Padding in milliseconds applied both before and after each contiguous run.

    Returns
    -------
    padded : 1D bool array, shape (n_times,)
        Padded mask.
    """
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


def _compute_aperiodic_slope_per_sample(
    data_cont: np.ndarray,
    sfreq: float,
    epoch_len_samples: int,
    fmin: float = FMIN_FIT,
    fmax: float = FMAX_FIT,
):
    """Compute 1/f slopes in non-overlapping windows, then expand per sample.

    For each non-overlapping window of length `epoch_len_samples` and channel,
    this function:

    1. Computes Welch PSD.
    2. Fits a straight line in log10–log10 space over [fmin, fmax]:
           log10 P(f) ≈ slope * log10 f + intercept
    3. Returns the slope & intercept for each window and channel.
    4. Expands those slopes to all samples in the corresponding window, producing
       a per-sample slope array suitable for plotting under the timeseries.

    Parameters
    ----------
    data_cont : ndarray, shape (n_channels, n_times)
        Continuous data.
    sfreq : float
        Sampling frequency (Hz).
    epoch_len_samples : int
        Window length in samples (e.g. 5 s * sfreq).
    fmin : float
        Minimum frequency for the aperiodic fit (Hz).
    fmax : float
        Maximum frequency for the aperiodic fit (Hz).

    Returns
    -------
    slope_per_sample : ndarray, shape (n_channels, n_times)
        For each sample and channel, the slope (aperiodic exponent) of the
        1/f fit in that channel's window. Samples beyond the last full window
        are set to NaN.
    slope_per_window : ndarray, shape (n_windows, n_channels)
        The slope estimates per 5 s window and channel.
    freqs : ndarray, shape (n_freqs,)
        Frequency vector for the PSD.
    psd_win : ndarray, shape (n_windows, n_channels, n_freqs)
        Welch PSD for each window and channel.
    intercept_per_window : ndarray, shape (n_windows, n_channels)
        The intercept estimates per 5 s window and channel.
    """
    n_channels, n_times = data_cont.shape

    if n_times < epoch_len_samples:
        # Not enough data for a single window
        slope_per_sample = np.full((n_channels, n_times), np.nan)
        slope_per_window = np.zeros((0, n_channels))
        freqs = np.zeros(0)
        psd_win = np.zeros((0, n_channels, 0))
        intercept_per_window = np.zeros((0, n_channels))
        return slope_per_sample, slope_per_window, freqs, psd_win, intercept_per_window

    n_win = n_times // epoch_len_samples
    trimmed_len = n_win * epoch_len_samples
    data_trim = data_cont[:, :trimmed_len]

    # (n_windows, n_channels, epoch_len_samples)
    data_win = data_trim.reshape(n_channels, n_win, epoch_len_samples).transpose(1, 0, 2)

    # Compute PSD (same general strategy as in the excision module)
    n_fft = min(epoch_len_samples, 2048)
    n_overlap = n_fft // 2

    psd_win, freqs = psd_array_welch(
        data_win,
        sfreq=sfreq,
        fmin=0.1,
        fmax=max(80.0, fmax),
        n_fft=n_fft,
        n_overlap=n_overlap,
        average="mean",
        verbose=False,
    )

    # Fit slope over [fmin, fmax] in log10-log10 space
    freqs = np.asarray(freqs)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        # No frequencies in the desired band
        slope_per_sample = np.full((n_channels, n_times), np.nan)
        slope_per_window = np.full((n_win, n_channels), np.nan)
        intercept_per_window = np.full((n_win, n_channels), np.nan)
        return slope_per_sample, slope_per_window, freqs, psd_win, intercept_per_window

    log_f = np.log10(freqs[mask])
    log_psd = np.log10(psd_win[..., mask] + 1e-20)  # (n_win, n_channels, n_fit_freqs)

    n_win, n_channels, _ = log_psd.shape
    slope_per_window = np.zeros((n_win, n_channels), dtype=float)
    intercept_per_window = np.zeros((n_win, n_channels), dtype=float)

    for w in range(n_win):
        for ch in range(n_channels):
            y = log_psd[w, ch, :]
            if not np.all(np.isfinite(y)):
                slope_per_window[w, ch] = np.nan
                intercept_per_window[w, ch] = np.nan
                continue
            try:
                slope, intercept = np.polyfit(log_f, y, 1)
                slope_per_window[w, ch] = slope
                intercept_per_window[w, ch] = intercept
            except Exception:
                slope_per_window[w, ch] = np.nan
                intercept_per_window[w, ch] = np.nan

    # Expand to per-sample slope array
    slope_per_sample = np.full((n_channels, n_times), np.nan, dtype=float)
    for w in range(n_win):
        s = w * epoch_len_samples
        e = s + epoch_len_samples
        slope_per_sample[:, s:e] = slope_per_window[w, :, None]

    # Tail samples (if any) remain NaN
    return slope_per_sample, slope_per_window, freqs, psd_win, intercept_per_window


def _build_gui(
    data_cont: np.ndarray,
    sfreq: float,
    epoch_len_samples: int,
    component_masks_init: Dict[str, np.ndarray],
    ch_names,
    slope_per_sample: np.ndarray,
    slope_per_window: np.ndarray,
    freqs: np.ndarray,
    psd_win: np.ndarray,
    intercept_per_window: np.ndarray,
    amp_z_hard_init: float,
    deriv_z_hard_init: float,
    drift_z_thresh_init: float,
    intercept_z_thresh_init: float,
    min_dur_ms_amp_init: float,
    min_dur_ms_deriv_init: float,
) -> None:
    """Build and display the interactive artifact viewer GUI with 1/f slope & PSD panel.

    Parameters
    ----------
    data_cont : ndarray, shape (n_channels, n_times)
        Continuous multichannel data.
    sfreq : float
        Sampling frequency (Hz).
    epoch_len_samples : int
        Epoch/window length for spectral criteria (e.g. 5 s * sfreq).
    component_masks_init : dict
        Initial component masks from build_artifact_mask_i_to_iv.
    ch_names : list of str
        Channel names.
    slope_per_sample : ndarray, shape (n_channels, n_times)
        1/f slope expanded per sample (NaN if not available).
    slope_per_window : ndarray, shape (n_windows, n_channels)
        1/f slope per 5 s window.
    freqs : ndarray, shape (n_freqs,)
        Frequency vector for the PSD.
    psd_win : ndarray, shape (n_windows, n_channels, n_freqs)
        Welch PSD for each window and channel.
    intercept_per_window : ndarray, shape (n_windows, n_channels)
        1/f intercept per 5 s window.
    *_init : floats
        Initial parameter values for thresholds and min durations.
    """
    n_channels, n_times = data_cont.shape
    total_dur_s = n_times / sfreq
    times = np.arange(n_times) / sfreq
    epoch_len_s = epoch_len_samples / sfreq if epoch_len_samples > 0 else 0.0

    # --- Colors for each artifact type (light shades) ---
    color_map = {
        "amp": "#ffb3ff",        # light magenta
        "deriv": "#b3ffff",      # light cyan
        "drift": "#b3ffb3",      # light green
        "intercept": "#ffffb3",  # light yellow
    }

    # --- State: current (possibly padded) masks ---
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

    # --- Summary label with counts (seconds) ---
    n_total = n_times

    def _sec(n):
        return n / sfreq

    summary = widgets.HTML()

    # --- Color legend widget ---
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

    # Output area for plot
    plot_output = widgets.Output()

    # --- Functions to recompute masks & update summary/plot ---

    def recompute_masks(*_):
        """Recompute artifact masks based on current settings."""
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
        """Redraw the plot for current widget settings (timeseries + 1/f slope + PSD)."""
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

            # Determine which 5 s window the center belongs to
            n_win = slope_per_window.shape[0]
            if epoch_len_s > 0 and n_win > 0:
                win_idx = int(center // epoch_len_s)
                win_idx = max(0, min(n_win - 1, win_idx))
                w_t0 = win_idx * epoch_len_s
                w_t1 = w_t0 + epoch_len_s
            else:
                win_idx = None
                w_t0, w_t1 = None, None

            # Figure with three rows:
            # top = signal + artifact spans
            # middle = 1/f slope
            # bottom = PSD + 1/f fit
            fig, (ax_ts, ax_slope, ax_psd) = plt.subplots(
                3, 1, sharex=True, figsize=(10, 8),
                gridspec_kw={"height_ratios": [3, 1, 2]}
            )

            # Top: timeseries
            ax_ts.plot(t_seg, y, linewidth=0.8)

            # Overlay artifacts (padded masks)
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
                for s_idx, e_idx in segments:
                    ax_ts.axvspan(
                        t_seg[s_idx],
                        t_seg[e_idx - 1],
                        color=color_map[key],
                        alpha=0.4,
                        linewidth=0,
                    )

            # Highlight the active 5 s window (if defined)
            if w_t0 is not None and w_t1 is not None:
                ax_ts.axvspan(
                    w_t0, w_t1,
                    color="k", alpha=0.05, linewidth=0,
                )

            ax_ts.set_ylabel("Amplitude (a.u.)")
            ax_ts.set_title(
                f"Channel {ch_names[ch_idx]} | "
                f"Window {t0:.2f}–{t1:.2f} s (len {win_len:.2f} s)"
            )
            ax_ts.grid(True, alpha=0.3)

            # Middle: 1/f slope
            if slope_per_sample is not None and slope_per_sample.shape == (n_channels, n_times):
                slope_seg = slope_per_sample[ch_idx, i0:i1]
                ax_slope.plot(t_seg, slope_seg, linewidth=1.0)
                # Horizontal line at median slope for that channel (ignoring NaNs)
                ch_slopes = slope_per_sample[ch_idx]
                if np.isfinite(ch_slopes).any():
                    med_slope = np.nanmedian(ch_slopes)
                    ax_slope.axhline(med_slope, color="gray", linestyle="--", linewidth=0.8)
                    ax_slope.text(
                        t_seg[0],
                        med_slope,
                        f" median={med_slope:.2f}",
                        va="bottom",
                        ha="left",
                        fontsize=8,
                        color="gray",
                    )
                # Highlight the active 5 s window
                if w_t0 is not None and w_t1 is not None:
                    ax_slope.axvspan(
                        w_t0, w_t1,
                        color="k", alpha=0.05, linewidth=0,
                    )
                ax_slope.set_ylabel("1/f slope")
                ax_slope.grid(True, alpha=0.3)
            else:
                ax_slope.text(
                    0.5, 0.5,
                    "1/f slope unavailable (not enough data)",
                    transform=ax_slope.transAxes,
                    ha="center", va="center",
                )
                ax_slope.set_ylabel("1/f slope")

            # Bottom: PSD + 1/f fit for the active 5 s window
            if (
                win_idx is not None
                and psd_win is not None
                and psd_win.ndim == 3
                and win_idx < psd_win.shape[0]
            ):
                psd_ch = psd_win[win_idx, ch_idx, :]
                if np.isfinite(psd_ch).any():
                    ax_psd.loglog(freqs, psd_ch, linewidth=1.0, label="PSD (5 s window)")

                    slope_val = slope_per_window[win_idx, ch_idx]
                    intercept_val = intercept_per_window[win_idx, ch_idx]
                    if np.isfinite(slope_val) and np.isfinite(intercept_val):
                        log_f = np.log10(freqs + 1e-20)
                        fit_log = slope_val * log_f + intercept_val
                        fit_lin = 10.0 ** fit_log
                        ax_psd.loglog(freqs, fit_lin, linestyle="--", linewidth=1.0, label="1/f fit")

                    # Mark the fit band
                    ax_psd.axvspan(
                        FMIN_FIT, FMAX_FIT,
                        color="gray", alpha=0.1, linewidth=0,
                    )

                    ax_psd.set_ylabel("PSD (a.u.)")
                    ax_psd.grid(True, which="both", alpha=0.3)
                    ax_psd.legend(fontsize=8, loc="best")
                else:
                    ax_psd.text(
                        0.5, 0.5,
                        "PSD not finite for this window/channel",
                        transform=ax_psd.transAxes,
                        ha="center", va="center",
                    )
                    ax_psd.set_ylabel("PSD (a.u.)")
            else:
                ax_psd.text(
                    0.5, 0.5,
                    "PSD unavailable (not enough data / outside window range)",
                    transform=ax_psd.transAxes,
                    ha="center", va="center",
                )
                ax_psd.set_ylabel("PSD (a.u.)")

            ax_psd.set_xlabel("Frequency (Hz)")
            ax_psd.set_xlim(max(freqs.min(), 0.1) if freqs.size else 0.1,
                            freqs.max() if freqs.size else 100.0)

            plt.tight_layout()
            plt.show()

    # --- Wire up callbacks ---

    # Controls that only affect the view
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

    # Controls that require recomputing masks
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
            widgets.HTML(value="<h3>Artifact Viewer (i–iv) with Threshold, Padding, 1/f Slope, and PSD</h3>"),
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
        Initial hard z-threshold for amplitude excursions (criterion i).
    deriv_z_hard : float
        Initial hard z-threshold for derivative excursions (criterion ii).
    drift_z_thresh : float
        Initial robust z-threshold for drift power (criterion iii).
    intercept_z_thresh : float
        Initial robust z-threshold for 1/f intercept (criterion iv).
    min_dur_ms_amp : float
        Initial minimum duration for amplitude excursions, in ms.
    min_dur_ms_deriv : float
        Initial minimum duration for derivative excursions, in ms.

    Notes
    -----
    - This function does not excise or filter the data; it only visualizes
      where each artifact criterion would flag samples (including padding),
      shows the 1/f slope trajectory, and displays the PSD + 1/f fit for the
      5 s window that contains the current time-center.
    - Use `clean_and_export(...)` from artifact_excision_resegment_5s_v1.py
      to actually excise and re-epoch + filter, once you are happy with the
      parameter settings (you can then copy the chosen thresholds into that call).
    """
    if not os.path.isfile(input_pkl_path):
        raise FileNotFoundError(f"Input pickle file not found: {input_pkl_path}")

    # Load and build continuous data
    data_cont, epoch_len_samples, ch_names = load_pickle_as_continuous(
        input_pkl_path,
        sfreq=sfreq,
        epoch_len_s=epoch_len_s,
    )

    # Compute initial artifact masks (i–iv)
    _, component_masks = build_artifact_mask_i_to_iv(
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

    # Compute 1/f slope per sample (5 s windows) for plotting,
    # along with PSD and intercept per window.
    slope_per_sample, slope_per_window, freqs, psd_win, intercept_per_window = _compute_aperiodic_slope_per_sample(
        data_cont,
        sfreq=sfreq,
        epoch_len_samples=epoch_len_samples,
    )

    # Build and display GUI
    _build_gui(
        data_cont,
        sfreq,
        epoch_len_samples,
        component_masks,
        ch_names,
        slope_per_sample,
        slope_per_window,
        freqs,
        psd_win,
        intercept_per_window,
        amp_z_hard,
        deriv_z_hard,
        drift_z_thresh,
        intercept_z_thresh,
        min_dur_ms_amp,
        min_dur_ms_deriv,
    )
