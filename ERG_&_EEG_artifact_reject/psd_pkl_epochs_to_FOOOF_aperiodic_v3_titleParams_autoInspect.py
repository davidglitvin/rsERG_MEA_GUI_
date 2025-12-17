
"""
psd_pkl_epochs_to_FOOOF_aperiodic_v1.py

GUI tool to estimate instantaneous aperiodic components (offset, exponent, optional knee)
from epoch-wise PSD data stored in a cleaned PSD .pkl file.

Input
-----
- Cleaned PSD .pkl produced by:
  psd_clean_v8_import_psdCompute_varSfreqSafe_nanWelchLongestSeg_0_100Hz_fix1

Expected .pkl structure:
{
    "key1": {"freqs": 1D array (n_freqs,), "psd": 2D array (n_epochs, n_freqs)},
    "key2": {...},
    ...
}

For each selected key (channel / block):

- Runs a FOOOFGroup fit across all epochs (one spectrum per epoch).
- Extracts aperiodic parameters (offset, exponent, and knee if using "knee" mode)
  for each epoch, applying an R^2 threshold by setting params to NaN when R^2 is too low.
- Provides:
    * Per-channel time series of exponent and offset across epochs.
    * Interactive inspection of FOOOF fit for any epoch.
    * Export of aperiodic parameters to Excel.
    * Optional export of aperiodic time-series figures as PNG.

FOOOF fitting parameters (freq min/max, amplitude threshold, R^2 threshold, max peaks,
mode, peak width limits) and plotting parameters (axis ranges, R^2 in title, peak table)
are configurable via widgets.

This module is designed to be run inside a Jupyter notebook.
"""

import os
import pickle
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ipywidgets as widgets
from ipyfilechooser import FileChooser
from IPython.display import display, clear_output

from fooof import FOOOFGroup


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def load_psd_pkl(path):
    """
    Load a cleaned PSD .pkl.

    Accepts either:
      - dict[channel_key] = {"freqs": 1D, "psd": 2D}
      - {"psd_results": {channel_key: {...}}}
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "psd_results" in obj and isinstance(obj["psd_results"], dict):
        psd_dict = obj["psd_results"]
    elif isinstance(obj, dict):
        psd_dict = obj
    else:
        raise ValueError("Unexpected PSD pickle structure (expected dict or dict['psd_results']).")

    # Basic sanity check
    cleaned = {}
    for k, v in psd_dict.items():
        if not isinstance(v, dict):
            continue
        freqs = np.asarray(v.get("freqs", None))
        psd = np.asarray(v.get("psd", None))
        if freqs is None or psd is None:
            continue
        if psd.ndim == 1:
            psd = psd[np.newaxis, :]
        if freqs.ndim != 1:
            raise ValueError(f"Channel '{k}' freqs is not 1D.")
        cleaned[k] = {"freqs": freqs, "psd": psd}
    return cleaned


def make_safe_name(key):
    """Make a filesystem-safe string from a channel key."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(key))
    return safe


def run_fooof_on_psd_dict(
    psd_dict,
    selected_keys,
    freq_range,
    amp_threshold,
    r2_threshold,
    max_peaks,
    fitting_mode,
    peak_width_limits
):
    """
    For each selected key, run FOOOFGroup across epochs.

    Returns
    -------
    fooof_groups : dict
        key -> FOOOFGroup (fit over valid epochs only).
    aperiodic_summary : dict
        key -> {
            "offsets": (n_epochs,) array (NaN where R^2 below threshold or no valid PSD),
            "exponents": (n_epochs,),
            "knees": (n_epochs,) or None,
            "r2": (n_epochs,) full R^2 vector (NaN where no fit),
            "valid_indices": 1D array of epoch indices passed to FOOOF (for mapping)
        }
    """
    fooof_groups = {}
    aperiodic_summary = {}

    for key in selected_keys:
        info = psd_dict.get(key, None)
        if info is None:
            continue
        freqs = np.asarray(info["freqs"], dtype=float)
        psd = np.asarray(info["psd"], dtype=float)

        if psd.ndim == 1:
            psd = psd[np.newaxis, :]

        n_epochs, n_freqs = psd.shape
        if freqs.shape[0] != n_freqs:
            raise ValueError(
                f"Frequency mismatch in key '{key}': freqs len={freqs.shape[0]}, psd n_freqs={n_freqs}"
            )

        # Identify epochs with all-finite PSD values
        finite_mask = np.isfinite(psd).all(axis=1)
        valid_indices = np.where(finite_mask)[0]
        psd_valid = psd[finite_mask, :]

        offsets = np.full(n_epochs, np.nan, dtype=float)
        exponents = np.full(n_epochs, np.nan, dtype=float)
        r2_full = np.full(n_epochs, np.nan, dtype=float)
        if fitting_mode == "knee":
            knees = np.full(n_epochs, np.nan, dtype=float)
        else:
            knees = None

        if psd_valid.size == 0:
            # No valid epochs to fit
            fooof_groups[key] = None
            aperiodic_summary[key] = {
                "offsets": offsets,
                "exponents": exponents,
                "knees": knees,
                "r2": r2_full,
                "valid_indices": valid_indices,
            }
            continue

        fg = FOOOFGroup(
            peak_width_limits=peak_width_limits,
            max_n_peaks=max_peaks,
            min_peak_height=amp_threshold,
            verbose=False,
            aperiodic_mode=fitting_mode,
        )

        fg.fit(freqs, psd_valid, freq_range)

        r2_vals = fg.get_params("r_squared")
        ap_params = fg.get_params("aperiodic_params")

        # Map back to full epoch indices
        for j, ep_idx in enumerate(valid_indices):
            if j >= len(r2_vals) or j >= len(ap_params):
                continue
            r2 = r2_vals[j]
            ap = ap_params[j]
            r2_full[ep_idx] = r2

            # Apply R^2 threshold by marking low-quality fits as NaN in offset/exponent (but keep R^2)
            if (r2 is None) or np.isnan(r2) or (r2 < r2_threshold):
                continue

            ap = np.asarray(ap).ravel()
            if fitting_mode == "fixed":
                if ap.size >= 2:
                    offsets[ep_idx] = float(ap[0])
                    exponents[ep_idx] = float(ap[1])
            else:  # "knee"
                if ap.size >= 3:
                    offsets[ep_idx] = float(ap[0])
                    knees[ep_idx] = float(ap[1])
                    exponents[ep_idx] = float(ap[2])

        fooof_groups[key] = fg
        aperiodic_summary[key] = {
            "offsets": offsets,
            "exponents": exponents,
            "knees": knees,
            "r2": r2_full,
            "valid_indices": valid_indices,
        }

    return fooof_groups, aperiodic_summary


def make_aperiodic_timeseries_figure(key, ap_summary, show_knee=False):
    """
    Create a figure with exponent & offset time series (and optionally knee).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    offsets = ap_summary["offsets"]
    exponents = ap_summary["exponents"]
    knees = ap_summary["knees"]
    r2 = ap_summary["r2"]

    n_epochs = len(exponents)
    epochs = np.arange(1, n_epochs + 1)

    n_rows = 2 + (1 if (show_knee and knees is not None) else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 3 * n_rows), sharex=True)

    if n_rows == 1:
        axes = [axes]

    ax0 = axes[0]
    ax0.plot(epochs, exponents, marker="o")
    ax0.set_ylabel("Exponent")
    ax0.set_title(f"Aperiodic Exponent across epochs - {key}")
    ax0.grid(True)

    ax1 = axes[1]
    ax1.plot(epochs, offsets, marker="o")
    ax1.set_ylabel("Offset")
    ax1.grid(True)

    row_idx = 2
    if show_knee and (knees is not None):
        ax2 = axes[row_idx]
        ax2.plot(epochs, knees, marker="o")
        ax2.set_ylabel("Knee")
        ax2.grid(True)
        row_idx += 1

    axes[-1].set_xlabel("Epoch index")

    # Optionally we could overlay R^2 on a secondary axis, but for now keep it simple.
    fig.tight_layout()
    return fig


def plot_single_epoch_fooof(
    fg,
    epoch_index,
    valid_indices,
    key,
    x_min,
    x_max,
    y_min,
    y_max,
    show_grid=True,
    include_r2=True,
    include_peak_table=False,
    x_tick_font_size=8,
):
    """
    Plot FOOOF fit for a single epoch.

    Parameters
    ----------
    fg : FOOOFGroup or None
        FOOOFGroup object fitted over valid epochs for this key.
    epoch_index : int
        Epoch index in the original epoch space (0-based).
    valid_indices : array-like
        Indices of epochs that were passed to FOOOF (finite PSD).
    """
    if fg is None or len(fg) == 0 or valid_indices is None or len(valid_indices) == 0:
        print(f"No FOOOF model available for key '{key}'.")
        return

    valid_indices = np.asarray(valid_indices, dtype=int)
    where = np.where(valid_indices == epoch_index)[0]
    if where.size == 0:
        print(f"Epoch {epoch_index} was not fitted (non-finite PSD).")
        return
    j = int(where[0])

    fm = fg.get_fooof(ind=j, regenerate=True)
    r2_value = fm.get_params("r_squared")

    # Extract aperiodic params for this epoch
    ap = np.asarray(fm.get_params("aperiodic_params"), dtype=float).ravel()
    offset_val = np.nan
    exponent_val = np.nan
    knee_val = None
    if ap.size >= 2:
        offset_val = float(ap[0])
        if ap.size == 2:
            exponent_val = float(ap[1])
        elif ap.size >= 3:
            knee_val = float(ap[1])
            exponent_val = float(ap[2])

    # Build informative title
    parts = [f"{key} - Epoch {epoch_index}"]
    if include_r2 and (r2_value is not None) and not np.isnan(r2_value):
        parts.append(f"R²={r2_value:.2f}")
    if not np.isnan(offset_val):
        parts.append(f"off={offset_val:.2f}")
    if not np.isnan(exponent_val):
        parts.append(f"exp={exponent_val:.2f}")
    if knee_val is not None:
        parts.append(f"knee={knee_val:.2f}")
    title = "  |  ".join(parts)

    # FOOOF plot in log-log (log10 frequency, log10 power)
    fm.plot(
        title=title,
        plot_peaks="shade",
        plt_log=True,
        freq_range=[x_min, x_max],
    )

    # Y-axis (log10 power) limits
    if y_min is not None or y_max is not None:
        plt.ylim(y_min, y_max)

    plt.grid(show_grid)

    # X ticks: label in Hz but place on log10 scale
    ax = plt.gca()
    int_ticks = np.arange(np.ceil(x_min), np.floor(x_max) + 1, 1, dtype=float)
    if len(int_ticks) > 10:
        step = int(np.ceil(len(int_ticks) / 10.0))
        int_ticks = int_ticks[::step]
    tick_positions = np.log10(int_ticks)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{f:.0f}" for f in int_ticks])
    plt.setp(ax.get_xticklabels(), fontsize=x_tick_font_size)

    plt.show()

    if include_peak_table:
        peak_params = fm.peak_params_
        if peak_params.size > 0:
            df_peaks = pd.DataFrame(
                peak_params, columns=["Center Freq (Hz)", "Amplitude", "FWHM"]
            )
            print("Detected Peaks:")
            display(df_peaks)
        else:
            print("No peaks detected.")


# ---------------------------------------------------------------------
# Main GUI builder
# ---------------------------------------------------------------------

def build_psd_pkl_fooof_aperiodic_gui():
    """
    Build and display the GUI for epoch-wise aperiodic FOOOF analysis
    of PSD .pkl files.
    """

    # ---------------------------
    # Widgets
    # ---------------------------
    psd_pkl_chooser = FileChooser(
        os.getcwd(),
        title="Select cleaned PSD .pkl (from psd_clean_v8...)",
        select_default=False,
    )
    psd_pkl_chooser.show_only_files = True
    psd_pkl_chooser.filter_pattern = ["*.pkl"]

    output_dir_chooser = FileChooser(
        os.getcwd(),
        title="Select output directory (for Excel & figures)",
        select_default=False,
    )
    output_dir_chooser.show_only_files = False

    load_button = widgets.Button(
        description="Load PSD .pkl", button_style="info", icon="folder-open"
    )

    # Channel selection
    channel_select = widgets.SelectMultiple(
        options=[],
        description="Channels/Keys:",
        layout=widgets.Layout(width="300px", height="200px"),
    )

    # FOOOF parameter widgets
    freq_min_widget = widgets.FloatText(
        value=4.0, description="Freq Min (Hz):", layout=widgets.Layout(width="220px")
    )
    freq_max_widget = widgets.FloatText(
        value=45.0, description="Freq Max (Hz):", layout=widgets.Layout(width="220px")
    )
    amp_threshold_widget = widgets.FloatText(
        value=0.2, description="Amplitude Thresh:", layout=widgets.Layout(width="220px")
    )
    r2_threshold_widget = widgets.FloatText(
        value=0.5, description="R² Thresh:", layout=widgets.Layout(width="220px")
    )
    max_peaks_widget = widgets.IntText(
        value=2, description="Max Peaks:", layout=widgets.Layout(width="220px")
    )
    fitting_mode_widget = widgets.Dropdown(
        options=["knee", "fixed"],
        value="knee",
        description="Mode:",
        layout=widgets.Layout(width="220px"),
    )
    peak_width_min_widget = widgets.FloatText(
        value=2.0, description="Peak Width Min:", layout=widgets.Layout(width="220px")
    )
    peak_width_max_widget = widgets.FloatText(
        value=10.0, description="Peak Width Max:", layout=widgets.Layout(width="220px")
    )

    # Plot settings for single-epoch FOOOF plots
    x_min_widget = widgets.FloatText(
        value=4.0, description="X Min (Hz):", layout=widgets.Layout(width="220px")
    )
    x_max_widget = widgets.FloatText(
        value=45.0, description="X Max (Hz):", layout=widgets.Layout(width="220px")
    )
    y_min_widget = widgets.FloatText(
        value=0.0, description="Y Min (log10):", layout=widgets.Layout(width="220px")
    )
    y_max_widget = widgets.FloatText(
        value=10.0, description="Y Max (log10):", layout=widgets.Layout(width="220px")
    )
    x_tick_font_size_widget = widgets.IntSlider(
        value=8,
        min=6,
        max=20,
        step=1,
        description="X Tick Font Size:",
        layout=widgets.Layout(width="350px"),
    )
    include_r2_checkbox = widgets.Checkbox(
        value=True, description="Include R² in title"
    )
    include_peak_table_checkbox = widgets.Checkbox(
        value=False, description="Show peak table"
    )

    # Fitting & plotting controls
    run_fooof_button = widgets.Button(
        description="Run FOOOF on epochs", button_style="success", icon="check"
    )
    plot_series_button = widgets.Button(
        description="Plot exponent/offset series", button_style="primary", icon="line-chart"
    )
    export_button = widgets.Button(
        description="Export aperiodic data & figures", button_style="warning", icon="save"
    )

    # Epoch inspection widgets
    epoch_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=0,
        step=1,
        description="Epoch idx:",
        continuous_update=False,
        layout=widgets.Layout(width="400px"),
    )
    inspect_button = widgets.Button(
        description="Plot FOOOF for this epoch", button_style="info", icon="search"
    )

    # Base name for export files
    basename_widget = widgets.Text(
        value="aperiodic",
        description="Base name:",
        layout=widgets.Layout(width="300px"),
    )

    # Output areas
    load_output = widgets.Output()
    fooof_output = widgets.Output()
    series_output = widgets.Output()
    inspect_output = widgets.Output()
    export_output = widgets.Output()

    # ---------------------------
    # Shared state
    # ---------------------------
    psd_dict = {}
    fooof_groups = {}
    aperiodic_summary = {}
    n_epochs_global = 0  # max n_epochs across selected keys

    # ---------------------------
    # Callbacks
    # ---------------------------
    def on_load_clicked(b):
        nonlocal psd_dict, n_epochs_global
        with load_output:
            clear_output()
            path = psd_pkl_chooser.selected
            if not path:
                print("Please select a PSD .pkl file.")
                return
            try:
                psd_dict = load_psd_pkl(path)
            except Exception as e:
                print(f"Error loading PSD .pkl: {e}")
                return

            if not psd_dict:
                print("Loaded PSD dict is empty.")
                return

            keys = sorted(psd_dict.keys())
            channel_select.options = keys

            # Determine max n_epochs across channels
            n_epochs_global = 0
            for k, v in psd_dict.items():
                psd = np.asarray(v["psd"])
                if psd.ndim == 1:
                    psd = psd[np.newaxis, :]
                n_epochs_global = max(n_epochs_global, psd.shape[0])
            epoch_slider.min = 0
            epoch_slider.max = max(0, n_epochs_global - 1)
            epoch_slider.value = 0

            print(f"Loaded PSD .pkl with {len(keys)} keys.")
            print("Keys:", keys)

    load_button.on_click(on_load_clicked)

    def on_run_fooof_clicked(b):
        nonlocal fooof_groups, aperiodic_summary
        with fooof_output:
            clear_output()
            if not psd_dict:
                print("No PSD data loaded yet.")
                return

            selected_keys = list(channel_select.value)
            if not selected_keys:
                print("No channels/keys selected.")
                return

            freq_min = freq_min_widget.value
            freq_max = freq_max_widget.value
            if freq_min >= freq_max:
                print("Freq Min must be < Freq Max.")
                return
            freq_range = [freq_min, freq_max]

            amp_th = amp_threshold_widget.value
            r2_th = r2_threshold_widget.value
            max_peaks = max_peaks_widget.value
            mode = fitting_mode_widget.value
            pw_min = peak_width_min_widget.value
            pw_max = peak_width_max_widget.value
            peak_width_limits = [pw_min, pw_max]

            print("Running FOOOF on selected keys...")
            fooof_groups, aperiodic_summary = run_fooof_on_psd_dict(
                psd_dict,
                selected_keys,
                freq_range=freq_range,
                amp_threshold=amp_th,
                r2_threshold=r2_th,
                max_peaks=max_peaks,
                fitting_mode=mode,
                peak_width_limits=peak_width_limits,
            )
            print("Done. Extracted aperiodic parameters for:")
            for k in selected_keys:
                if k in aperiodic_summary:
                    ap = aperiodic_summary[k]
                    print(
                        f"  {k}: {np.sum(~np.isnan(ap['exponents']))} epochs with valid exponent (R² >= {r2_th})."
                    )

    run_fooof_button.on_click(on_run_fooof_clicked)

    def on_plot_series_clicked(b):
        with series_output:
            clear_output()
            if not aperiodic_summary:
                print("No aperiodic summary available. Run FOOOF first.")
                return
            selected_keys = list(channel_select.value)
            if not selected_keys:
                print("No channels/keys selected.")
                return

            for key in selected_keys:
                ap = aperiodic_summary.get(key, None)
                if ap is None:
                    print(f"No aperiodic summary for key '{key}'.")
                    continue
                fig = make_aperiodic_timeseries_figure(
                    key, ap, show_knee=(fitting_mode_widget.value == "knee")
                )
                plt.show(fig)

    plot_series_button.on_click(on_plot_series_clicked)

    def _update_inspect_plot(change=None):
        """Update the single-epoch FOOOF plot whenever epoch or settings change."""
        with inspect_output:
            clear_output()
            if not fooof_groups:
                print("No FOOOF results yet. Run FOOOF first.")
                return
            key = None
            if channel_select.value:
                # Take first selected key for inspection
                key = channel_select.value[0]
            if key is None:
                print("No channel selected for inspection.")
                return
            fg = fooof_groups.get(key, None)
            ap = aperiodic_summary.get(key, None)
            if ap is None:
                print(f"No aperiodic summary for key '{key}'.")
                return

            epoch_idx = epoch_slider.value
            valid_indices = ap.get("valid_indices", None)
            print(f"Inspecting key '{key}', epoch {epoch_idx}")
            x_min = x_min_widget.value
            x_max = x_max_widget.value
            y_min = y_min_widget.value
            y_max = y_max_widget.value

            plot_single_epoch_fooof(
                fg,
                epoch_idx,
                valid_indices,
                key,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                show_grid=True,
                include_r2=include_r2_checkbox.value,
                include_peak_table=include_peak_table_checkbox.value,
                x_tick_font_size=x_tick_font_size_widget.value,
            )

    def on_inspect_clicked(b):
        """Button handler: manually trigger an update of the inspect plot."""
        _update_inspect_plot()

    inspect_button.on_click(on_inspect_clicked)

    # Automatically refresh the inspect plot when epoch index or plotting
    # settings change, so you don't have to repeatedly click the button.
    epoch_slider.observe(_update_inspect_plot, "value")
    channel_select.observe(_update_inspect_plot, "value")
    x_min_widget.observe(_update_inspect_plot, "value")
    x_max_widget.observe(_update_inspect_plot, "value")
    y_min_widget.observe(_update_inspect_plot, "value")
    y_max_widget.observe(_update_inspect_plot, "value")
    x_tick_font_size_widget.observe(_update_inspect_plot, "value")
    include_r2_checkbox.observe(_update_inspect_plot, "value")
    include_peak_table_checkbox.observe(_update_inspect_plot, "value")

    def on_export_clicked(b):
        with export_output:
            clear_output()
            if not aperiodic_summary:
                print("No aperiodic summary available to export.")
                return

            out_dir = output_dir_chooser.selected
            if not out_dir:
                # Try to fall back to directory of PSD file
                if psd_pkl_chooser.selected:
                    out_dir = os.path.dirname(psd_pkl_chooser.selected)
                else:
                    print("Please select an output directory.")
                    return

            if not os.path.isdir(out_dir):
                print(f"Output directory does not exist: {out_dir}")
                return

            base = basename_widget.value.strip() or "aperiodic"

            # 1) Export Excel with per-epoch parameters
            rows = []
            for key, ap in aperiodic_summary.items():
                offsets = ap["offsets"]
                exponents = ap["exponents"]
                knees = ap["knees"]
                r2_vals = ap["r2"]
                n_epochs = len(exponents)
                for ep in range(n_epochs):
                    row = {
                        "key": key,
                        "epoch_index": ep,
                        "offset": float(offsets[ep]) if not np.isnan(offsets[ep]) else np.nan,
                        "exponent": float(exponents[ep]) if not np.isnan(exponents[ep]) else np.nan,
                        "r_squared": float(r2_vals[ep]) if not np.isnan(r2_vals[ep]) else np.nan,
                    }
                    if knees is not None:
                        val = knees[ep]
                        row["knee"] = float(val) if not np.isnan(val) else np.nan
                    rows.append(row)

            if rows:
                df = pd.DataFrame(rows)
                excel_path = os.path.join(out_dir, f"{base}_aperiodic_params.xlsx")
                df.to_excel(excel_path, index=False)
                print(f"Exported aperiodic parameters to: {excel_path}")
            else:
                print("No rows to export to Excel (all NaN?).")

            # 2) Export exponent/offset series figures as PNG
            for key, ap in aperiodic_summary.items():
                fig = make_aperiodic_timeseries_figure(
                    key, ap, show_knee=(fitting_mode_widget.value == "knee")
                )
                safe_key = make_safe_name(key)
                fig_path = os.path.join(out_dir, f"{base}_aperiodic_timeseries_{safe_key}.png")
                fig.savefig(fig_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved figure for key '{key}' to: {fig_path}")

    export_button.on_click(on_export_clicked)

    # ---------------------------
    # Layout
    # ---------------------------
    file_box = widgets.VBox(
        [
            widgets.HTML("<h3>1) Load PSD .pkl</h3>"),
            psd_pkl_chooser,
            load_button,
            load_output,
        ]
    )

    channel_box = widgets.VBox(
        [
            widgets.HTML("<h3>2) Select channels / keys</h3>"),
            channel_select,
        ]
    )

    fooof_params_box = widgets.VBox(
        [
            widgets.HTML("<h3>3) FOOOF parameters</h3>"),
            widgets.HBox([freq_min_widget, freq_max_widget]),
            widgets.HBox([amp_threshold_widget, r2_threshold_widget]),
            widgets.HBox([max_peaks_widget, fitting_mode_widget]),
            widgets.HBox([peak_width_min_widget, peak_width_max_widget]),
            run_fooof_button,
            fooof_output,
        ]
    )

    inspect_box = widgets.VBox(
        [
            widgets.HTML("<h3>4) Inspect single-epoch FOOOF fit</h3>"),
            widgets.Label("Select one key in the channel list; then choose epoch index:"),
            epoch_slider,
            widgets.HBox(
                [x_min_widget, x_max_widget, y_min_widget, y_max_widget]
            ),
            x_tick_font_size_widget,
            widgets.HBox([include_r2_checkbox, include_peak_table_checkbox]),
            inspect_button,
            inspect_output,
        ]
    )

    series_box = widgets.VBox(
        [
            widgets.HTML("<h3>5) Aperiodic exponent/offset series</h3>"),
            widgets.Label("Plots are generated for all selected keys."),
            plot_series_button,
            series_output,
        ]
    )

    export_box = widgets.VBox(
        [
            widgets.HTML("<h3>6) Export results</h3>"),
            output_dir_chooser,
            basename_widget,
            export_button,
            export_output,
        ]
    )

    ui = widgets.VBox(
        [
            widgets.HTML("<h2>Epoch-wise aperiodic FOOOF analysis from PSD .pkl</h2>"),
            widgets.HBox([file_box, channel_box]),
            fooof_params_box,
            inspect_box,
            series_box,
            export_box,
        ]
    )

    display(ui)


# If this module is run directly in a notebook, build the GUI
if __name__ == "__main__":
    build_psd_pkl_fooof_aperiodic_gui()
