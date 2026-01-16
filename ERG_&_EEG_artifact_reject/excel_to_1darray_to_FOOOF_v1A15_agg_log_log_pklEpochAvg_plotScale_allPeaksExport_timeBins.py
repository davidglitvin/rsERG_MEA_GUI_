import os
import re
import pickle
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from fooof import FOOOFGroup
from scipy.spatial.distance import cdist
from tqdm.notebook import tqdm

# Tkinter for file/directory dialogs
from tkinter import Tk, filedialog

# Optional: For PPT export
try:
    from pptx import Presentation
    from pptx.util import Inches
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

###############################################################################
#                           CORE ANALYSIS FUNCTIONS
###############################################################################

def process_single_sheet_excel(file_paths):
    """
    For *grouped* data:
    Reads each .xlsx file (assuming each file has exactly ONE relevant sheet).
      - The first column is assumed to be Frequency.
      - Each subsequent column is a PSD for a particular group or mouse.

    Returns a dict of the form:
      {
         file_path: { 
            "freq": freq_array,
            "Mouse1_Eye1": PSD_array,
            "Mouse1_Eye2": PSD_array,
            ...
         }
      }
    """
    psd_data_dict_all = {}

    for file_path in file_paths:
        try:
            # Assume the Excel has only ONE relevant sheet; read the first sheet by default
            df = pd.read_excel(file_path, sheet_name=0)

            # The first column is frequency
            freq_column = df.iloc[:, 0].values

            # Prepare a dictionary for all PSD columns
            psd_data_dict = {"freq": freq_column}

            # For each remaining column, store it as a separate PSD vector
            for col_name in df.columns[1:]:
                psd_data_dict[col_name] = df[col_name].values

            psd_data_dict_all[file_path] = psd_data_dict

        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    return psd_data_dict_all

def process_eye_averages_excel(file_paths):
    """
    For *individual* mice (old approach):
    Reads the sheet named "Eye Averages" (the 17th sheet, but we reference by name).
    Expects columns:
      - Column A: Frequencies
      - Column B: "Eye1 Average PSD"
      - Column C: "Eye2 Average PSD"

    Returns a dict of the form:
      {
         file_path: {
            "freq": freq_array,
            "Eye1 Average PSD": psd_array_for_eye1,
            "Eye2 Average PSD": psd_array_for_eye2
         }
      }
    """
    psd_data_dict_all = {}

    for file_path in file_paths:
        try:
            df = pd.read_excel(file_path, sheet_name="Eye Averages")

            freq_column = df.iloc[:, 0].values
            eye1_col = df["Eye1 Average PSD"].values
            eye2_col = df["Eye2 Average PSD"].values

            psd_data_dict = {
                "freq": freq_column,
                "Eye1 Average PSD": eye1_col,
                "Eye2 Average PSD": eye2_col
            }

            psd_data_dict_all[file_path] = psd_data_dict

        except Exception as e:
            print(f"An error occurred while processing 'Eye Averages' in {file_path}: {e}")

    return psd_data_dict_all

def process_individual_mice_means_excel(file_paths):
    """
    For *individual mice means* (NEW approach):
    Reads exactly one sheet, expecting four columns:
      - Column A: PSD for Eye1
      - Column B: Frequency (for Eye1)
      - Column C: PSD for Eye2
      - Column D: Frequency (redundant, for Eye2)

    We use Column A & C for PSD, and Column B for Frequency.
    Column D is ignored since it's redundant.

    Returns a dict of the form:
      {
         file_path: {
            "freq": freq_array,
            "Eye1": psd_array_for_eye1,
            "Eye2": psd_array_for_eye2
         }
      }
    """
    psd_data_dict_all = {}

    for file_path in file_paths:
        try:
            df = pd.read_excel(file_path, sheet_name=0)
            
            # Column A: PSD for Eye1
            eye1_col = df.iloc[:, 0].values
            # Column B: frequency
            freq_col = df.iloc[:, 1].values
            # Column C: PSD for Eye2
            eye2_col = df.iloc[:, 2].values
            # Column D: freq again (redundant), so ignore

            psd_data_dict = {
                "freq": freq_col,
                "Eye1": eye1_col,
                "Eye2": eye2_col
            }

            psd_data_dict_all[file_path] = psd_data_dict

        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    return psd_data_dict_all


def _average_consecutive_epochs(psd_2d: np.ndarray, n_avg: int):
    """Average consecutive epochs in blocks of n_avg.

    Notes
    -----
    - If n_avg <= 1, returns the input unchanged.
    - If the number of epochs is not divisible by n_avg, the remainder epochs at
      the end are dropped.

    Returns
    -------
    psd_avg : np.ndarray
        Shape (n_blocks, n_freqs)
    meta : dict
        Information about the reduction.
    """
    psd_2d = np.asarray(psd_2d)
    if psd_2d.ndim != 2:
        raise ValueError(f"Expected a 2D PSD array (n_epochs, n_freqs), got shape {psd_2d.shape}.")

    n_avg = int(n_avg) if n_avg is not None else 1
    if n_avg <= 1:
        meta = {
            "epoch_avg_n": 1,
            "n_epochs_original": int(psd_2d.shape[0]),
            "n_epochs_used": int(psd_2d.shape[0]),
            "n_epochs_dropped_remainder": 0,
            "n_blocks": int(psd_2d.shape[0]),
        }
        return psd_2d, meta

    n_epochs, n_freqs = psd_2d.shape
    n_blocks = n_epochs // n_avg
    n_used = n_blocks * n_avg
    n_dropped = n_epochs - n_used

    if n_blocks == 0:
        raise ValueError(
            f"Averaging factor n_avg={n_avg} is larger than the number of epochs ({n_epochs})."
        )

    psd_used = psd_2d[:n_used, :]
    psd_avg = psd_used.reshape(n_blocks, n_avg, n_freqs).mean(axis=1)

    meta = {
        "epoch_avg_n": int(n_avg),
        "n_epochs_original": int(n_epochs),
        "n_epochs_used": int(n_used),
        "n_epochs_dropped_remainder": int(n_dropped),
        "n_blocks": int(n_blocks),
    }
    return psd_avg, meta


def _format_time_window_label(start_s: float, end_s: float, avg_n: int, epoch_len_s: float) -> str:
    """Create a compact human-readable label for an averaged time window.

    Examples
    --------
    - "0–10 min (avg 10 min; N=120)"
    - "30–40 min (avg 10 min; N=120)"
    """
    # Guard against weird inputs
    try:
        start_s = float(start_s)
        end_s = float(end_s)
        epoch_len_s = float(epoch_len_s)
        avg_n = int(avg_n)
    except Exception:
        return ""

    if not np.isfinite(start_s) or not np.isfinite(end_s) or not np.isfinite(epoch_len_s) or avg_n <= 0:
        return ""

    window_s = max(end_s - start_s, 0.0)
    window_min = window_s / 60.0
    start_min = start_s / 60.0
    end_min = end_s / 60.0

    def _fmt_num(x: float) -> str:
        # show integers cleanly, otherwise one decimal
        if abs(x - round(x)) < 1e-9:
            return f"{int(round(x))}"
        return f"{x:.1f}"

    if window_s >= 60:
        return f"{_fmt_num(start_min)}–{_fmt_num(end_min)} min (avg {_fmt_num(window_min)} min; N={avg_n})"
    else:
        # Rare, but keep sane for small windows
        return f"{_fmt_num(start_s)}–{_fmt_num(end_s)} s (avg {_fmt_num(window_s)} s; N={avg_n})"


def _build_fit_time_labels(fg_meta: dict, n_fits: int):
    """Return per-fit time window labels using epoch-averaging metadata (if present)."""
    if not isinstance(fg_meta, dict) or n_fits <= 0:
        return [""] * max(int(n_fits), 0)

    epoch_meta = fg_meta.get("epoch_averaging") or {}
    avg_n = epoch_meta.get("epoch_avg_n", None)
    epoch_len_s = fg_meta.get("epoch_length_s", None)
    fit_indices_original = fg_meta.get("fit_indices_original", None)

    try:
        avg_n = int(avg_n) if avg_n is not None else None
    except Exception:
        avg_n = None
    try:
        epoch_len_s = float(epoch_len_s) if epoch_len_s is not None else None
    except Exception:
        epoch_len_s = None

    if avg_n is None or epoch_len_s is None or avg_n <= 1 or epoch_len_s <= 0:
        return [""] * int(n_fits)

    window_s = avg_n * epoch_len_s
    labels = []
    for i in range(int(n_fits)):
        # Use original index if low-R2 dropping was enabled
        i_orig = (
            fit_indices_original[i]
            if isinstance(fit_indices_original, list) and i < len(fit_indices_original)
            else i
        )
        start_s = float(i_orig) * window_s
        end_s = start_s + window_s
        labels.append(_format_time_window_label(start_s, end_s, avg_n=avg_n, epoch_len_s=epoch_len_s))
    return labels


def process_cleaned_psd_pickle(file_paths, epoch_avg_n: int = 1):
    """Load cleaned per-epoch PSDs from a .pkl and optionally average epochs.

    Expected .pkl structure (as in rd1_56_psd_results_cleaned.pkl):
      {
        'Ch1': {'psd': (n_epochs, n_freqs), 'freqs': (n_freqs,)},
        ...
        'Ch16': {'psd': (n_epochs, n_freqs), 'freqs': (n_freqs,)}
      }

    Returns
    -------
    psd_data_dict_all : dict
        {
          file_path: {
            'freq': freq_array,
            'Ch1': psd_2d_or_avg_2d,
            ...,
            '__meta__': { 'Ch1': {...}, ... }
          }
        }
    """
    psd_data_dict_all = {}

    for file_path in file_paths:
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            if not isinstance(data, dict) or not data:
                raise ValueError("Pickle did not contain a non-empty dict.")

            # Find first channel-like entry that has freqs
            first_key = None
            for k, v in data.items():
                if isinstance(v, dict) and ("freqs" in v) and ("psd" in v):
                    first_key = k
                    break
            if first_key is None:
                raise ValueError("Pickle dict did not have the expected structure: {ChX: {'psd','freqs'}}")

            freq_ref = np.asarray(data[first_key]["freqs"], dtype=float)
            if freq_ref.ndim != 1:
                raise ValueError(f"Expected 1D freqs array, got shape {freq_ref.shape}")

            psd_data_dict = {"freq": freq_ref}
            meta_dict = {}

            # Load each channel
            for ch_name, ch_blob in data.items():
                if not isinstance(ch_blob, dict) or "psd" not in ch_blob or "freqs" not in ch_blob:
                    continue

                freqs = np.asarray(ch_blob["freqs"], dtype=float)
                if freqs.shape != freq_ref.shape or not np.allclose(freqs, freq_ref, rtol=0, atol=1e-12):
                    raise ValueError(
                        f"Frequency axis mismatch for {ch_name} in {os.path.basename(file_path)}."
                    )

                psd = np.asarray(ch_blob["psd"], dtype=float)
                if psd.ndim != 2:
                    raise ValueError(f"Expected 2D PSD array for {ch_name}, got shape {psd.shape}")

                psd_avg, meta = _average_consecutive_epochs(psd, epoch_avg_n)
                psd_data_dict[ch_name] = psd_avg
                meta_dict[ch_name] = meta

            if len(psd_data_dict) <= 1:
                raise ValueError("No valid channel PSDs found in pickle.")

            psd_data_dict["__meta__"] = meta_dict
            psd_data_dict_all[file_path] = psd_data_dict

        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    return psd_data_dict_all

def run_fooof_analysis(psd_data_dict_all, freq_range, amp_threshold, r2_threshold, max_peaks,
                       fitting_mode, peak_width_limits, drop_low_r2=True, epoch_length_s: float = 5.0):
    """
    Runs FOOOF analysis on each PSD column (except for 'freq') in each file.
    Returns a dictionary (fg_dict) mapping a unique key -> FOOOFGroup object.
    The unique key is constructed as: "columnName_fileName".
    """
    fg_dict = {}
    fg_meta_dict = {}

    for file_path, psd_data_dict in tqdm(psd_data_dict_all.items(), desc="Fitting PSDs with FOOOF"):
        freq = psd_data_dict["freq"]
        file_suffix = os.path.splitext(os.path.basename(file_path))[0]

        # For each PSD column in the dictionary
        meta_by_signal = psd_data_dict.get("__meta__", {}) if isinstance(psd_data_dict, dict) else {}

        for col_name, psd in psd_data_dict.items():
            if col_name == "freq" or (isinstance(col_name, str) and col_name.startswith("__")):
                continue  # skip the frequency array

            fg = FOOOFGroup(
                peak_width_limits=peak_width_limits,
                max_n_peaks=max_peaks,
                min_peak_height=amp_threshold,
                verbose=False,
                aperiodic_mode=fitting_mode,
            )

            psd_arr = np.asarray(psd)
            if psd_arr.ndim == 1:
                spectra = psd_arr[np.newaxis, :]
            elif psd_arr.ndim == 2:
                spectra = psd_arr
            else:
                raise ValueError(f"Unexpected PSD array shape for {col_name}: {psd_arr.shape}")

            # Fit all spectra
            fg.fit(freq, spectra, freq_range)

            # Optional: drop low-R2 fits (retain mapping to original indices via metadata)
            r2_all = np.asarray(fg.get_params('r_squared'))
            pass_mask = r2_all >= r2_threshold
            keep_indices = np.where(pass_mask)[0]
            if drop_low_r2:
                fg.drop(~pass_mask)

            # Mapping from post-drop index -> original index (before any dropping)
            fit_indices_original = keep_indices.astype(int).tolist() if drop_low_r2 else list(range(int(len(r2_all))))

            dict_key = f"{col_name}_{file_suffix}"
            fg_dict[dict_key] = fg

            # Attach metadata for downstream export/plotting
            fg_meta_dict[dict_key] = {
                "source_file": file_path,
                "signal_name": col_name,
                "file_suffix": file_suffix,
                "freq_range": list(freq_range),
                "peak_width_limits": list(peak_width_limits),
                "max_n_peaks": int(max_peaks),
                "min_peak_height": float(amp_threshold),
                "aperiodic_mode": str(fitting_mode),
                "r2_threshold": float(r2_threshold),
                "drop_low_r2": bool(drop_low_r2),
                "n_fits_attempted": int(len(r2_all)),
                "fit_indices_original": fit_indices_original,
                "epoch_averaging": meta_by_signal.get(col_name, None),
                "epoch_length_s": float(epoch_length_s),
            }

    return fg_dict, fg_meta_dict

def plot_closest_to_mean(
    fg,
    sheet_name,
    export_dir,
    base_filename,
    formats,
    x_min,
    x_max,
    y_min,
    y_max,
    show_grid=True,
    x_tick_font_size=8,
    include_r2=False,
    include_peak_table=False,
    freq_axis_mode="log",
    fit_time_labels=None,
):
    """
    For FOOOFGroup objects with multiple PSD fits, selects 10 spectra closest
    to the mean (in Euclidean sense) and plots them.
    """
    models = [fg.get_fooof(ind=i) for i in range(len(fg))]
    spectra = [model.power_spectrum for model in models]
    spectra = np.array(spectra)

    if spectra.ndim != 2:
        raise ValueError(f"Unexpected spectra shape: {spectra.shape}. Expected 2D array.")

    mean_spectrum = np.mean(spectra, axis=0)
    distances = cdist(spectra, mean_spectrum[None, :], metric='euclidean').flatten()
    closest_indices = np.argsort(distances)[:10]

    # If exporting to PPT:
    if "ppt" in formats and PPTX_AVAILABLE:
        prs = Presentation()
        slide_layout = prs.slide_layouts[6]  # blank

    for i, index in enumerate(closest_indices):
        fm = fg.get_fooof(ind=index, regenerate=True)
        r2_value = fm.get_params('r_squared')

        time_lbl = ""
        if isinstance(fit_time_labels, (list, tuple)) and index < len(fit_time_labels):
            time_lbl = str(fit_time_labels[index] or "").strip()

        if include_r2:
            title = (
                f"{sheet_name} - {time_lbl} - Closest Example {index} (R²: {r2_value:.2f})"
                if time_lbl else f"{sheet_name} - Closest Example {index} (R²: {r2_value:.2f})"
            )
        else:
            title = (
                f"{sheet_name} - {time_lbl} - Closest Example {index}"
                if time_lbl else f"{sheet_name} - Closest Example {index}"
            )

        # Plot scaling
        # - freq_axis_mode='log': x-axis shown in log10(Hz) space (log-frequency view)
        # - freq_axis_mode='linear': x-axis shown in linear Hz
        use_log_freq = (str(freq_axis_mode).lower() == "log")

        fm.plot(
            title=title,
            plot_peaks='shade',
            plt_log=use_log_freq,
            freq_range=[x_min, x_max]
        )

        # y-axis remains under your control (values are already log10(power) in FOOOF)
        plt.ylim(y_min, y_max)
        plt.grid(show_grid)
        plt.margins(0)

        # X ticks in Hz (labels), with positions depending on frequency axis mode
        ax = plt.gca()
        int_ticks = np.arange(np.ceil(x_min), np.floor(x_max) + 1, 1, dtype=float)
        if len(int_ticks) > 10:
            step = int(np.ceil(len(int_ticks) / 10))
            int_ticks = int_ticks[::step]

        if use_log_freq:
            # FOOOF's plt_log=True uses log10(frequency) on the x-data
            tick_positions = np.log10(int_ticks)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f"{f:.0f}" for f in int_ticks])
        else:
            # Linear frequency axis in Hz
            ax.set_xlim([x_min, x_max])
            ax.set_xticks(int_ticks)
            ax.set_xticklabels([f"{f:.0f}" for f in int_ticks])

        plt.setp(ax.get_xticklabels(), fontsize=x_tick_font_size)

        fig = plt.gcf()
        file_tag = f"{base_filename}_{sheet_name}_ex{i}"

        # Save or export
        for fmt in formats:
            if fmt in ["png", "svg", "jpeg"]:
                filename = os.path.join(export_dir, f"{file_tag}.{fmt}")
                fig.savefig(filename, dpi=300, bbox_inches='tight')
            elif fmt == "ppt" and PPTX_AVAILABLE:
                tmp_png = os.path.join(export_dir, f"{file_tag}_ppt_temp.png")
                fig.savefig(tmp_png, dpi=300, bbox_inches='tight')
                slide = prs.slides.add_slide(slide_layout)
                left = top = Inches(1)
                slide.shapes.add_picture(tmp_png, left, top, height=Inches(5))

        plt.show()

        # Optionally display peak parameters as a table
        if include_peak_table:
            peak_params = fm.peak_params_
            if peak_params.size > 0:
                df_peaks = pd.DataFrame(
                    peak_params,
                    columns=['Center Freq (Hz)', 'Amplitude', 'FWHM']
                )
                print("Detected Peaks:")
                display(df_peaks)
            else:
                print("No peaks detected.")

        plt.clf()

    # Save PPT if requested
    if "ppt" in formats and PPTX_AVAILABLE:
        pptx_filename = os.path.join(export_dir, f"{base_filename}_{sheet_name}.pptx")
        prs.save(pptx_filename)

def plot_all_psds(
    fg,
    sheet_name,
    export_dir,
    base_filename,
    formats,
    x_min,
    x_max,
    y_min,
    y_max,
    show_grid=True,
    x_tick_font_size=8,
    include_r2=False,
    include_peak_table=False,
    freq_axis_mode="log",
    fit_time_labels=None,
):
    """
    Plots each fitted PSD in the FOOOFGroup. Typically each FOOOFGroup here has 1 PSD,
    but if there's more, it plots each.
    """
    n_fits = len(fg)
    if "ppt" in formats and PPTX_AVAILABLE:
        prs = Presentation()
        slide_layout = prs.slide_layouts[6]

    for i in range(n_fits):
        fm = fg.get_fooof(ind=i, regenerate=True)
        r2_value = fm.get_params('r_squared')
        time_lbl = ""
        if isinstance(fit_time_labels, (list, tuple)) and i < len(fit_time_labels):
            time_lbl = str(fit_time_labels[i] or "").strip()

        if include_r2:
            title = (
                f"{sheet_name} - {time_lbl} - Example {i} (R²: {r2_value:.2f})"
                if time_lbl else f"{sheet_name} - Example {i} (R²: {r2_value:.2f})"
            )
        else:
            title = (
                f"{sheet_name} - {time_lbl} - Example {i}"
                if time_lbl else f"{sheet_name} - Example {i}"
            )

        use_log_freq = (str(freq_axis_mode).lower() == "log")

        fm.plot(
            title=title,
            plot_peaks='shade',
            plt_log=use_log_freq,
            freq_range=[x_min, x_max]
        )

        # y-axis control
        plt.ylim(y_min, y_max)
        plt.grid(show_grid)
        plt.margins(0)

        # X ticks in Hz (labels), with positions depending on frequency axis mode
        ax = plt.gca()
        int_ticks = np.arange(np.ceil(x_min), np.floor(x_max) + 1, 1, dtype=float)
        if len(int_ticks) > 10:
            step = int(np.ceil(len(int_ticks) / 10))
            int_ticks = int_ticks[::step]

        if use_log_freq:
            tick_positions = np.log10(int_ticks)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f"{f:.0f}" for f in int_ticks])
        else:
            ax.set_xlim([x_min, x_max])
            ax.set_xticks(int_ticks)
            ax.set_xticklabels([f"{f:.0f}" for f in int_ticks])

        plt.setp(ax.get_xticklabels(), fontsize=x_tick_font_size)

        fig = plt.gcf()
        file_tag = f"{base_filename}_{sheet_name}_ex{i}"
        for fmt in formats:
            if fmt in ["png", "svg", "jpeg"]:
                filename = os.path.join(export_dir, f"{file_tag}.{fmt}")
                fig.savefig(filename, dpi=300, bbox_inches='tight')
            elif fmt == "ppt" and PPTX_AVAILABLE:
                tmp_png = os.path.join(export_dir, f"{file_tag}_ppt_temp.png")
                fig.savefig(tmp_png, dpi=300, bbox_inches='tight')
                slide = prs.slides.add_slide(slide_layout)
                left = top = Inches(1)
                slide.shapes.add_picture(tmp_png, left, top, height=Inches(5))

        plt.show()

        if include_peak_table:
            peak_params = fm.peak_params_
            if peak_params.size > 0:
                df_peaks = pd.DataFrame(
                    peak_params,
                    columns=['Center Freq (Hz)', 'Amplitude', 'FWHM']
                )
                print(f"Detected Peaks for Example {i}:")
                display(df_peaks)
            else:
                print(f"No peaks detected for Example {i}.")

        plt.clf()

    if "ppt" in formats and PPTX_AVAILABLE:
        pptx_filename = os.path.join(export_dir, f"{base_filename}_{sheet_name}.pptx")
        prs.save(pptx_filename)
        
def save_fooof_group(fg_dict, fg_meta_dict=None, filename="fooof_groups.pkl"):
    """Serialize and save FOOOF results to a pickle.

    Backwards-compatible behavior:
    - If fg_meta_dict is None, saves fg_dict only (as in older versions).
    - Otherwise saves a dict with both fg_dict and fg_meta_dict.
    """
    payload = fg_dict if fg_meta_dict is None else {"fg_dict": fg_dict, "fg_meta_dict": fg_meta_dict}
    with open(filename, "wb") as f:
        pickle.dump(payload, f)
    print(f"FOOOFGroup results saved to {filename}")


def export_fg_dict_excel(fg_dict, excel_filename, fg_meta_dict=None):
    """Export FOOOFGroup parameters to an Excel file.

    Notes
    -----
    In FOOOF/FOOOFGroup, `get_params('peak_params')` returns *all peaks across all fits*
    (often as a long array with an extra index column). Indexing it by fit can therefore
    silently drop peaks.

    To guarantee exporting *all peaks for each fit*, this exporter pulls peak parameters
    from each underlying FOOOFModel via `fg.get_fooof(ind=i).peak_params_`.

    Output
    ------
    - Sheet 'Sheet1' : one row per fit, plus "wide" peak columns up to max_n_peaks.
    - Sheet 'peaks' : long-format table, one row per detected peak.

    If fg_meta_dict is provided, also exports source + epoch-averaging metadata and
    preserves pre-drop indices (fit_index_original) when low-R² fits are dropped.
    """

    # Determine a global max_n_peaks for wide export columns (keeps sheet consistent)
    max_n_peaks_global = 0
    for key in fg_dict.keys():
        meta = (fg_meta_dict or {}).get(key, {})
        try:
            max_n_peaks_global = max(max_n_peaks_global, int(meta.get('max_n_peaks', 0) or 0))
        except Exception:
            pass

    fit_rows = []
    peak_rows = []

    for key, fg in fg_dict.items():
        meta = (fg_meta_dict or {}).get(key, {})
        fit_indices_original = meta.get("fit_indices_original", None)

        n_fits = len(fg)

        # These are reliably per-fit arrays
        r2_params = list(fg.get_params('r_squared'))
        ap_params = list(fg.get_params('aperiodic_params'))

        epoch_meta = meta.get("epoch_averaging") or {}
        epoch_len_s = meta.get("epoch_length_s", None)

        for i in range(n_fits):
            # Fit-level params
            r2 = r2_params[i] if i < len(r2_params) else None
            ap = ap_params[i] if i < len(ap_params) else None

            fit_i_orig = (
                fit_indices_original[i]
                if isinstance(fit_indices_original, list) and i < len(fit_indices_original)
                else i
            )

            # Time-bin labeling (only meaningful when epoch averaging is used)
            avg_n = epoch_meta.get("epoch_avg_n", None)
            try:
                avg_n = int(avg_n) if avg_n is not None else None
            except Exception:
                avg_n = None
            try:
                epoch_len_s_f = float(epoch_len_s) if epoch_len_s is not None else None
            except Exception:
                epoch_len_s_f = None

            time_start_s = np.nan
            time_end_s = np.nan
            time_start_min = np.nan
            time_end_min = np.nan
            avg_window_s = np.nan
            avg_window_min = np.nan
            time_bin_label = ""

            if avg_n is not None and epoch_len_s_f is not None and avg_n > 1 and epoch_len_s_f > 0:
                avg_window_s = float(avg_n) * float(epoch_len_s_f)
                avg_window_min = avg_window_s / 60.0
                time_start_s = float(fit_i_orig) * avg_window_s
                time_end_s = time_start_s + avg_window_s
                time_start_min = time_start_s / 60.0
                time_end_min = time_end_s / 60.0
                time_bin_label = _format_time_window_label(time_start_s, time_end_s, avg_n=avg_n, epoch_len_s=epoch_len_s_f)

            # Pull ALL peaks from the individual model (robust)
            fm = fg.get_fooof(ind=i, regenerate=False)
            peaks_arr = getattr(fm, 'peak_params_', None)
            if peaks_arr is None:
                peaks_arr = np.empty((0, 3), dtype=float)
            peaks_arr = np.asarray(peaks_arr)
            if peaks_arr.size == 0:
                peaks_arr = np.empty((0, 3), dtype=float)
            if peaks_arr.ndim == 1 and peaks_arr.shape[0] == 3:
                peaks_arr = peaks_arr.reshape(1, 3)

            n_peaks = int(peaks_arr.shape[0])

            # Fit summary row (backward-compatible columns kept)
            row = {
                "dict_key": key,
                "signal_name": meta.get("signal_name", None),
                "source_file": meta.get("source_file", None),
                "fit_index": i,
                "fit_index_original": fit_i_orig,
                "r_squared": r2,
                "aperiodic_params": str(ap),
                # Store *all* peaks for this fit as a stringified array
                "peak_params": str(peaks_arr),
                "n_peaks": n_peaks,
                "epoch_avg_n": epoch_meta.get("epoch_avg_n", None),
                "epoch_length_s": epoch_len_s,
                "avg_window_s": avg_window_s,
                "avg_window_min": avg_window_min,
                "time_start_s": time_start_s,
                "time_end_s": time_end_s,
                "time_start_min": time_start_min,
                "time_end_min": time_end_min,
                "time_bin_label": time_bin_label,
                "n_epochs_original": epoch_meta.get("n_epochs_original", None),
                "n_epochs_used": epoch_meta.get("n_epochs_used", None),
                "n_epochs_dropped_remainder": epoch_meta.get("n_epochs_dropped_remainder", None),
                "n_blocks": epoch_meta.get("n_blocks", None),
                "drop_low_r2": meta.get("drop_low_r2", None),
                "r2_threshold": meta.get("r2_threshold", None),
                "n_fits_attempted": meta.get("n_fits_attempted", None),
            }

            # Add wide peak columns up to global max peaks
            # Columns follow FOOOF convention: [CF, PW, BW]
            for p in range(max_n_peaks_global):
                cf_col = f"peak{p+1}_cf_hz"
                pw_col = f"peak{p+1}_amp"
                bw_col = f"peak{p+1}_bw_hz"
                if p < n_peaks:
                    row[cf_col] = float(peaks_arr[p, 0])
                    row[pw_col] = float(peaks_arr[p, 1])
                    row[bw_col] = float(peaks_arr[p, 2])
                else:
                    row[cf_col] = np.nan
                    row[pw_col] = np.nan
                    row[bw_col] = np.nan

            fit_rows.append(row)

            # Long-format peak rows (one row per peak)
            for p in range(n_peaks):
                peak_rows.append({
                    "dict_key": key,
                    "signal_name": meta.get("signal_name", None),
                    "source_file": meta.get("source_file", None),
                    "fit_index": i,
                    "fit_index_original": fit_i_orig,
                    "peak_index": p,
                    "peak_cf_hz": float(peaks_arr[p, 0]),
                    "peak_amp": float(peaks_arr[p, 1]),
                    "peak_bw_hz": float(peaks_arr[p, 2]),
                    "r_squared": r2,
                    "aperiodic_params": str(ap),
                    "epoch_avg_n": epoch_meta.get("epoch_avg_n", None),
                    "epoch_length_s": epoch_len_s,
                    "avg_window_s": avg_window_s,
                    "avg_window_min": avg_window_min,
                    "time_start_s": time_start_s,
                    "time_end_s": time_end_s,
                    "time_start_min": time_start_min,
                    "time_end_min": time_end_min,
                    "time_bin_label": time_bin_label,
                    "n_epochs_original": epoch_meta.get("n_epochs_original", None),
                    "n_epochs_used": epoch_meta.get("n_epochs_used", None),
                    "n_epochs_dropped_remainder": epoch_meta.get("n_epochs_dropped_remainder", None),
                    "n_blocks": epoch_meta.get("n_blocks", None),
                    "drop_low_r2": meta.get("drop_low_r2", None),
                    "r2_threshold": meta.get("r2_threshold", None),
                    "n_fits_attempted": meta.get("n_fits_attempted", None),
                })

    df_fits = pd.DataFrame(fit_rows)
    df_peaks = pd.DataFrame(peak_rows)

    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        df_fits.to_excel(writer, index=False, sheet_name='Sheet1')
        # Only write peaks sheet if any peaks exist
        if len(df_peaks) > 0:
            df_peaks.to_excel(writer, index=False, sheet_name='peaks')

    print(f"FOOOF results exported to Excel: {excel_filename}")


def display_fg_dict_table(fg_dict, fg_meta_dict=None):
    """Displays FOOOFGroup parameters in a table within the notebook.

    This shows one row per fit. The 'peak_params' cell contains *all* peaks for that fit
    (stringified array). For full peak-by-peak output, use the Excel export (sheet 'peaks').
    """
    rows = []
    for key, fg in fg_dict.items():
        meta = (fg_meta_dict or {}).get(key, {})
        fit_indices_original = meta.get("fit_indices_original", None)

        epoch_meta = meta.get("epoch_averaging") or {}
        epoch_len_s = meta.get("epoch_length_s", None)

        n_fits = len(fg)
        r2_params = list(fg.get_params('r_squared'))
        ap_params = list(fg.get_params('aperiodic_params'))

        for i in range(n_fits):
            r2 = r2_params[i] if i < len(r2_params) else None
            ap = ap_params[i] if i < len(ap_params) else None
            fit_i_orig = (
                fit_indices_original[i]
                if isinstance(fit_indices_original, list) and i < len(fit_indices_original)
                else i
            )
            fm = fg.get_fooof(ind=i, regenerate=False)
            peaks_arr = np.asarray(getattr(fm, 'peak_params_', np.empty((0, 3))))
            if peaks_arr.size == 0:
                peaks_arr = np.empty((0, 3))
            if peaks_arr.ndim == 1 and peaks_arr.shape[0] == 3:
                peaks_arr = peaks_arr.reshape(1, 3)

            # Optional time-bin label for epoch-averaged pickle mode
            avg_n = epoch_meta.get("epoch_avg_n", None)
            try:
                avg_n = int(avg_n) if avg_n is not None else None
            except Exception:
                avg_n = None
            try:
                epoch_len_s_f = float(epoch_len_s) if epoch_len_s is not None else None
            except Exception:
                epoch_len_s_f = None

            time_bin_label = ""
            if avg_n is not None and epoch_len_s_f is not None and avg_n > 1 and epoch_len_s_f > 0:
                window_s = avg_n * epoch_len_s_f
                start_s = float(fit_i_orig) * window_s
                end_s = start_s + window_s
                time_bin_label = _format_time_window_label(start_s, end_s, avg_n=avg_n, epoch_len_s=epoch_len_s_f)

            row = {
                "dict_key": key,
                "fit_index": i,
                "fit_index_original": fit_i_orig,
                "time_bin_label": time_bin_label,
                "r_squared": r2,
                "aperiodic_params": str(ap),
                "peak_params": str(peaks_arr),
                "n_peaks": int(peaks_arr.shape[0]),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    print("FOOOF Group Export Table:")
    display(df)

###############################################################################
#                         TKINTER-BASED GUI CODE & WIDGETS
###############################################################################


selected_files = []
export_directory = ""
fg_dict = {}       # Will store final FOOOFGroup results
fg_meta_dict = {}  # Will store metadata about each fit (useful for epoch-averaged pickle input)

###############################################################################
# Widgets
###############################################################################

# 1) Analysis mode: Now has a third option 'Individual Mice Means'
analysis_mode = widgets.Dropdown(
    options=["Grouped Mice", "Individual Mice", "Individual Mice Means", "Cleaned PSD Pickle (per-epoch)"],
    value="Grouped Mice",
    description="Analysis Mode:",
    layout=widgets.Layout(width="50%")
)

# Epoch averaging (only used for "Cleaned PSD Pickle (per-epoch)" mode)
epoch_avg_n_widget = widgets.IntText(
    description="Epoch Avg N:",
    value=1,
    layout=widgets.Layout(width="30%")
)

# Epoch length (seconds). Needed to label time bins when epoch averaging is used.
# Default assumes 5s epochs (e.g., 720 epochs = 60 min).
epoch_length_s_widget = widgets.FloatText(
    description="Epoch Len (s):",
    value=5.0,
    layout=widgets.Layout(width="30%")
)

# Whether to drop low-R² fits after fitting
drop_low_r2_checkbox = widgets.Checkbox(value=True, description="Drop fits below R²")

# Parameter widgets
freq_range_min = widgets.FloatText(description="Freq Min:", value=4.0, layout=widgets.Layout(width="30%"))
freq_range_max = widgets.FloatText(description="Freq Max:", value=45.0, layout=widgets.Layout(width="30%"))
amp_threshold = widgets.FloatText(description="Amplitude Threshold:", value=0.2, layout=widgets.Layout(width="50%"))
r2_threshold = widgets.FloatText(description="R² Threshold:", value=0.5, layout=widgets.Layout(width="50%"))
max_peaks = widgets.IntText(description="Max Peaks:", value=2, layout=widgets.Layout(width="50%"))
fitting_mode = widgets.Dropdown(description="Fitting Mode:", options=["knee", "fixed"], value="knee", layout=widgets.Layout(width="50%"))
peak_width_min = widgets.FloatText(description="Peak Width Min:", value=2.0, layout=widgets.Layout(width="50%"))
peak_width_max = widgets.FloatText(description="Peak Width Max:", value=10.0, layout=widgets.Layout(width="50%"))

# Axis-range widgets
x_axis_min = widgets.FloatText(description="X Axis Min:", value=4.0, layout=widgets.Layout(width="33%"))
x_axis_max = widgets.FloatText(description="X Axis Max:", value=45.0, layout=widgets.Layout(width="33%"))
y_axis_min = widgets.FloatText(description="Y Axis Min:", value=0.0, layout=widgets.Layout(width="33%"))
y_axis_max = widgets.FloatText(description="Y Axis Max:", value=10.0, layout=widgets.Layout(width="33%"))

# Plot axis scaling (frequency axis only; power is already in log10 units inside FOOOF)
freq_axis_mode_widget = widgets.Dropdown(
    options=[
        ("Log–log view (log10 frequency axis)", "log"),
        ("Log-power view (linear frequency axis)", "linear"),
    ],
    value="log",
    description="Plot Scale:",
    layout=widgets.Layout(width="50%")
)

# Show Grid
grid_checkbox = widgets.Checkbox(value=True, description="Show Grid")

# X-Tick Font Size
x_tick_font_size_widget = widgets.IntSlider(
    value=8,
    min=6,
    max=20,
    step=1,
    description="X Tick Font Size:",
    readout=True,
    layout=widgets.Layout(width="50%")
)

# Include R² in plot title
include_r2_checkbox = widgets.Checkbox(value=True, description="Include R² in Plot Titles")

# Include Peak Table
include_peak_table_checkbox = widgets.Checkbox(value=True, description="Include Peak Params Table")

# Export figures?
export_figures_checkbox = widgets.Checkbox(value=True, description="Export Figures")

# Plot closest to mean
plot_closest_to_mean_checkbox = widgets.Checkbox(value=True, description="Use Closest-to-Mean Plotting")

# Export FOOOF results as pickle/Excel
export_fg_pickle_checkbox = widgets.Checkbox(value=True, description="Export as Pickle")
export_fg_excel_checkbox = widgets.Checkbox(value=False, description="Export as Excel")

# Display FOOOF results table
display_fg_export_table_checkbox = widgets.Checkbox(value=True, description="Display Export Table")

# Figure format checkboxes
fmt_options = ["png", "svg", "jpeg", "ppt"]
format_checkboxes = [widgets.Checkbox(value=False, description=fmt.upper()) for fmt in fmt_options]

# Buttons
file_picker_button = widgets.Button(description="Select Excel or PSD Pickle Files", button_style="info", icon="folder")
directory_picker_button = widgets.Button(description="Select Output Directory", button_style="info", icon="folder")
run_button = widgets.Button(description="Process and Analyze", button_style="success", icon="check")

output = widgets.Output()

###################################
# 1. FILE SELECTION CALLBACK
###################################
def on_file_picker_button_click(b):
    global selected_files
    root = Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(filetypes=[("Excel files", "*.xlsx"), ("Pickle files", "*.pkl")])
    selected_files = list(file_paths)
    if selected_files:
        print(f"Selected files: {selected_files}")
    else:
        print("No files selected.")

file_picker_button.on_click(on_file_picker_button_click)

###################################
# 2. DIRECTORY SELECTION CALLBACK
###################################
def on_directory_picker_button_click(b):
    global export_directory
    root = Tk()
    root.withdraw()  # Hide the root window
    directory_path = filedialog.askdirectory()
    if directory_path:
        export_directory = directory_path
        print(f"Selected output directory: {export_directory}")
    else:
        print("No directory selected.")

directory_picker_button.on_click(on_directory_picker_button_click)

######################################################
# 3. MAIN BUTTON: PROCESS, ANALYZE, & EXPORT
######################################################
def on_run_button_click(b):
    global fg_dict, fg_meta_dict
    with output:
        output.clear_output()

        # 1) Check if files are selected.
        if not selected_files:
            print("Error: Please select at least one input file (.xlsx or .pkl).")
            return

        # 2) If exporting figures or results, ensure an export directory is selected.
        if (export_figures_checkbox.value or export_fg_pickle_checkbox.value or export_fg_excel_checkbox.value) and not export_directory:
            print("Error: Please select an output directory for exporting.")
            return

        # 3) Figure formats
        if export_figures_checkbox.value:
            chosen_formats = [cb.description.lower() for cb in format_checkboxes if cb.value]
            if not chosen_formats:
                print("Warning: No figure format selected. Proceeding without figure export.")
        else:
            chosen_formats = []  # Show plots but do not export

        # 4) Process the input files according to analysis mode
        print(f"Analysis mode: {analysis_mode.value}")
        
        if analysis_mode.value == "Grouped Mice":
            psd_data_dict_all = process_single_sheet_excel(selected_files)
        elif analysis_mode.value == "Individual Mice":
            psd_data_dict_all = process_eye_averages_excel(selected_files)
        elif analysis_mode.value == "Individual Mice Means":
            psd_data_dict_all = process_individual_mice_means_excel(selected_files)
        elif analysis_mode.value == "Cleaned PSD Pickle (per-epoch)":
            # Basic sanity: ensure the selected files are pickles
            bad_ext = [p for p in selected_files if os.path.splitext(p)[1].lower() != ".pkl"]
            if bad_ext:
                print("Error: In 'Cleaned PSD Pickle (per-epoch)' mode, please select .pkl files only.")
                print("Offending files:")
                for p in bad_ext:
                    print("  -", p)
                return
            psd_data_dict_all = process_cleaned_psd_pickle(selected_files, epoch_avg_n=epoch_avg_n_widget.value)
        else:
            psd_data_dict_all = {}

        # If using pickle mode, give a quick epoch-averaging summary
        if analysis_mode.value == "Cleaned PSD Pickle (per-epoch)" and psd_data_dict_all:
            for fp, dd in psd_data_dict_all.items():
                meta = dd.get("__meta__", {})
                n_blocks = [m.get("n_blocks") for m in meta.values() if isinstance(m, dict) and m.get("n_blocks") is not None]
                n_orig = [m.get("n_epochs_original") for m in meta.values() if isinstance(m, dict) and m.get("n_epochs_original") is not None]
                n_drop = [m.get("n_epochs_dropped_remainder") for m in meta.values() if isinstance(m, dict) and m.get("n_epochs_dropped_remainder") is not None]
                if n_blocks and n_orig:
                    try:
                        epoch_len_s = float(epoch_length_s_widget.value)
                    except Exception:
                        epoch_len_s = 5.0
                    avg_n = int(epoch_avg_n_widget.value)
                    win_s = avg_n * epoch_len_s
                    win_min = win_s / 60.0
                    print(
                        f"{os.path.basename(fp)}: epoch_avg_n={avg_n} | epoch_len={epoch_len_s:g}s | "
                        f"avg window={win_min:.3g} min | "
                        f"original epochs (min-max across channels)={min(n_orig)}-{max(n_orig)} | "
                        f"blocks to fit={min(n_blocks)}-{max(n_blocks)} | "
                        f"remainder dropped={min(n_drop) if n_drop else 0}-{max(n_drop) if n_drop else 0}"
                    )

        if not psd_data_dict_all:
            print("No data was processed. Please check your files and try again.")
            return

        # 5) Run FOOOF analysis
        freq_range = [freq_range_min.value, freq_range_max.value]
        peak_widths = [peak_width_min.value, peak_width_max.value]

        print("Running FOOOF analysis...")
        fg_dict, fg_meta_dict = run_fooof_analysis(
            psd_data_dict_all,
            freq_range=freq_range,
            amp_threshold=amp_threshold.value,
            r2_threshold=r2_threshold.value,
            max_peaks=max_peaks.value,
            fitting_mode=fitting_mode.value,
            peak_width_limits=peak_widths,
            drop_low_r2=drop_low_r2_checkbox.value,
            epoch_length_s=epoch_length_s_widget.value
        )

        # 6) Plot the results
        print("\nPlotting FOOOF results...")
        for dict_key, fg_item in fg_dict.items():
            try:
                print(f"\nPlotting results for: {dict_key}")
                meta = fg_meta_dict.get(dict_key, {})
                fit_time_labels = _build_fit_time_labels(meta, len(fg_item))
                if plot_closest_to_mean_checkbox.value and len(fg_item) > 1:
                    plot_closest_to_mean(
                        fg_item, dict_key, export_directory, dict_key, chosen_formats,
                        x_min=x_axis_min.value,
                        x_max=x_axis_max.value,
                        y_min=y_axis_min.value,
                        y_max=y_axis_max.value,
                        show_grid=grid_checkbox.value,
                        x_tick_font_size=x_tick_font_size_widget.value,
                        include_r2=include_r2_checkbox.value,
                        include_peak_table=include_peak_table_checkbox.value,
                        freq_axis_mode=freq_axis_mode_widget.value,
                        fit_time_labels=fit_time_labels
                    )
                else:
                    plot_all_psds(
                        fg_item, dict_key, export_directory, dict_key, chosen_formats,
                        x_min=x_axis_min.value,
                        x_max=x_axis_max.value,
                        y_min=y_axis_min.value,
                        y_max=y_axis_max.value,
                        show_grid=grid_checkbox.value,
                        x_tick_font_size=x_tick_font_size_widget.value,
                        include_r2=include_r2_checkbox.value,
                        include_peak_table=include_peak_table_checkbox.value,
                        freq_axis_mode=freq_axis_mode_widget.value,
                        fit_time_labels=fit_time_labels
                    )
            except Exception as e:
                print(f"Error plotting {dict_key}: {e}")

        # 7) Export FOOOF results
        if export_fg_pickle_checkbox.value:
            pickle_filename = os.path.join(export_directory, "fooof_groups.pkl")
            save_fooof_group(fg_dict, fg_meta_dict=fg_meta_dict, filename=pickle_filename)

        if export_fg_excel_checkbox.value:
            excel_filename = os.path.join(export_directory, "fooof_results.xlsx")
            export_fg_dict_excel(fg_dict, excel_filename, fg_meta_dict=fg_meta_dict)

        # 8) Optionally display a summary table
        if display_fg_export_table_checkbox.value:
            display_fg_dict_table(fg_dict, fg_meta_dict=fg_meta_dict)

run_button.on_click(on_run_button_click)

##############################
# 4. DISPLAY THE GUI
##############################
display(
    widgets.VBox(
        [
            widgets.Label(value="1) Select input files (.xlsx or cleaned PSD .pkl):"),
            file_picker_button,
            widgets.Label(value="2) Select Output Directory (for exporting):"),
            directory_picker_button,
            widgets.Label(value="3) Choose Analysis Mode:"),
            analysis_mode,
            widgets.Label(value="(Pickle mode only) Epoch averaging + optional low-R² dropping:"),
            widgets.HBox([epoch_avg_n_widget, epoch_length_s_widget, drop_low_r2_checkbox]),
            widgets.Label(value="4) Configure FOOOF Parameters:"),
            widgets.HBox([freq_range_min, freq_range_max]),
            amp_threshold,
            r2_threshold,
            max_peaks,
            fitting_mode,
            widgets.HBox([peak_width_min, peak_width_max]),
            widgets.Label(value="(Optional) Specify Plot Axis Ranges:"),
            widgets.HBox([x_axis_min, x_axis_max]),
            widgets.HBox([y_axis_min, y_axis_max]),
            widgets.Label(value="(Optional) Plot scaling (log frequency vs linear frequency):"),
            freq_axis_mode_widget,
            widgets.Label(value="(Optional) Grid, X-Tick Font, R² in Titles, and Peak Tables:"),
            widgets.HBox([grid_checkbox, x_tick_font_size_widget, include_r2_checkbox, include_peak_table_checkbox]),
            widgets.Label(value="5) Figure Export Options:"),
            export_figures_checkbox,
            plot_closest_to_mean_checkbox,
            widgets.Label(value="Choose Figure Export Formats:"),
            widgets.HBox(format_checkboxes),
            widgets.Label(value="6) Export FOOOF Results:"),
            widgets.HBox([export_fg_pickle_checkbox, export_fg_excel_checkbox]),
            widgets.Label(value="7) Display FOOOF Group Export in the Notebook:"),
            display_fg_export_table_checkbox,
            run_button,
            output,
        ]
    )
)
