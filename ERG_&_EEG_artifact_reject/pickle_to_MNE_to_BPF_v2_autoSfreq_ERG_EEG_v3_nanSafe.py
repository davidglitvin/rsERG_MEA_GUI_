import os
import pickle
import numpy as np
import mne
from tqdm.notebook import tqdm

import ipywidgets as widgets
from ipywidgets import VBox, HBox, Label, Output
from IPython.display import display, clear_output

from ipyfilechooser import FileChooser

##############################################################################
# HELPER FUNCTIONS
##############################################################################

def strip_datetime_prefix(basename):
    """
    Optionally strips the date-time prefix from the basename.
    For example:
      "2025-01-14_11-08-39_rd1_59_ChA-P4-LE_ChB-P2-RE_ind_1" becomes
      "rd1_59_ChA-P4-LE_ChB-P2-RE_ind_1"
    (Not used in the output naming below so that the source file name is preserved.)
    """
    parts = basename.split('_')
    if len(parts) > 2:
        return '_'.join(parts[2:])
    else:
        return basename


def load_pickle(file_path):
    """Loads a pickle file and returns the data."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def _extract_numeric(d, keys):
    for k in keys:
        if k in d:
            try:
                val = float(d[k])
                if np.isfinite(val) and val > 0:
                    return val
            except Exception:
                pass
    return None


def _extract_ch_names(d):
    for k in ["ch_names", "channel_names", "channels", "chans"]:
        if k in d:
            names = d[k]
            if isinstance(names, (list, tuple)) and all(isinstance(x, str) for x in names):
                return list(names)
    return None


def infer_sfreq_from_obj(obj):
    """
    Attempt to infer the sampling rate from a pickled object.
    This supports dictionaries produced by custom loaders that store metadata
    alongside the data array(s).
    """
    if isinstance(obj, dict):
        return _extract_numeric(obj, ["sfreq", "sampling_rate", "sample_rate", "fs", "samplingRate"])
    return None


def infer_ch_names(n_channels, mode="auto_eeg"):
    """
    Create channel names for an epoched array.

    mode:
      - "generic": Ch1..ChN
      - "auto_eeg": Ch1..Ch16 plus EEG/EEG1.. for any channels beyond 16
    """
    if mode == "generic":
        return [f"Ch{i+1}" for i in range(n_channels)]

    # auto_eeg
    if n_channels <= 16:
        return [f"Ch{i+1}" for i in range(n_channels)]

    base = [f"Ch{i+1}" for i in range(16)]
    extra = n_channels - 16
    if extra == 1:
        base.append("EEG")
    else:
        base.extend([f"EEG{i+1}" for i in range(extra)])
    return base


def create_epochs_array(data, sfreq, ch_names):
    """
    Converts a numpy array to MNE EpochsArray.
    Parameters:
      - data: numpy array of shape (n_epochs, n_channels, n_times)
      - sfreq: Sampling frequency
      - ch_names: List of channel names
    Returns:
      - epochs: MNE EpochsArray object
    """
    info = mne.create_info(ch_names=ch_names, sfreq=float(sfreq), ch_types='eeg')
    epochs = mne.EpochsArray(data, info, verbose=False)
    return epochs


##############################################################################
# NaN / Inf HANDLING
##############################################################################

def _count_nonfinite(data):
    """Return count of non-finite values (NaN/Inf) in the array."""
    try:
        return int(np.size(data) - np.isfinite(data).sum())
    except Exception:
        return 0


def _drop_nonfinite_epochs(data):
    """Drop epochs that contain any non-finite values. Returns (data_clean, kept_mask)."""
    # data: (n_epochs, n_channels, n_times)
    finite_epoch = np.isfinite(data).all(axis=(1, 2))
    return data[finite_epoch], finite_epoch


def _interp_1d_nonfinite(x, fill_value=0.0):
    """Linear interpolate non-finite values in a 1D vector; fallback to fill_value if no finite samples."""
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if finite.all():
        return x
    if finite.sum() == 0:
        x[:] = float(fill_value)
        return x
    idx = np.arange(x.size)
    # np.interp extrapolates ends using first/last finite values
    x[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    return x


def handle_nonfinite_epochs(data, policy="drop", fill_value=0.0, verbose_prefix=""):
    """
    Handle NaNs/Infs in epoched data.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_epochs, n_channels, n_times)
    policy : str
        One of:
          - 'drop'  : drop epochs that contain any non-finite values (recommended default)
          - 'interp': linear interpolation along time within each epoch/channel
          - 'fill'  : replace non-finite values with a constant fill_value
          - 'error' : raise ValueError if any non-finite values are present
    fill_value : float
        Used when policy == 'fill' OR when interpolation cannot be performed (all-NaN vectors).
    verbose_prefix : str
        Prefix string for log messages.

    Returns
    -------
    data_out : np.ndarray
        Cleaned array.
    report : dict
        Summary: {'n_nonfinite': int, 'policy': str, 'n_epochs_in': int,
                  'n_epochs_out': int, 'n_epochs_dropped': int}
    """
    data = np.asarray(data)
    report = {
        "policy": str(policy),
        "n_epochs_in": int(data.shape[0]) if data.ndim == 3 else 0,
        "n_epochs_out": int(data.shape[0]) if data.ndim == 3 else 0,
        "n_epochs_dropped": 0,
        "n_nonfinite": 0,
    }

    if data.ndim != 3:
        return data, report

    n_nonfinite = _count_nonfinite(data)
    report["n_nonfinite"] = n_nonfinite

    if n_nonfinite == 0:
        return data, report

    if policy == "error":
        raise ValueError(f"{verbose_prefix}Found {n_nonfinite} non-finite values (NaN/Inf).")

    if policy == "drop":
        data2, kept_mask = _drop_nonfinite_epochs(data)
        report["n_epochs_out"] = int(data2.shape[0])
        report["n_epochs_dropped"] = int((~kept_mask).sum())
        return data2, report

    # For interpolation / fill, we operate in float space
    data2 = data.astype(float, copy=True)

    if policy == "fill":
        data2[~np.isfinite(data2)] = float(fill_value)
        return data2, report

    if policy == "interp":
        # Interpolate per epoch/channel only if needed
        nonfinite_mask = ~np.isfinite(data2)
        if not nonfinite_mask.any():
            return data2, report
        n_epochs, n_ch, _ = data2.shape
        for ei in range(n_epochs):
            # quick check for epoch
            if not nonfinite_mask[ei].any():
                continue
            for ci in range(n_ch):
                if nonfinite_mask[ei, ci].any():
                    data2[ei, ci, :] = _interp_1d_nonfinite(
                        data2[ei, ci, :],
                        fill_value=fill_value
                    )
        return data2, report

    # unknown policy -> default to drop
    data2, kept_mask = _drop_nonfinite_epochs(data)
    report["n_epochs_out"] = int(data2.shape[0])
    report["n_epochs_dropped"] = int((~kept_mask).sum())
    report["policy"] = "drop"
    return data2, report



def _select_block_arrays_from_dict(d):
    """
    From a dictionary, return a new dict containing only values that look like
    3D epoched arrays: (n_epochs, n_channels, n_times).
    """
    out = {}
    for k, v in d.items():
        try:
            arr = np.asarray(v)
        except Exception:
            continue
        if arr.ndim == 3:
            out[k] = arr
    return out


def _extract_data_and_metadata(obj):
    """
    Normalize a pickled object into either:
      - a single 3D array
      - a dict of 3D block arrays

    Also attempts to extract:
      - sfreq metadata
      - channel names metadata

    Returns
    -------
    kind : str
        "array" or "blocks"
    payload : np.ndarray or dict
    sfreq_meta : float or None
    ch_names_meta : list[str] or None
    """
    sfreq_meta = infer_sfreq_from_obj(obj)
    ch_names_meta = _extract_ch_names(obj) if isinstance(obj, dict) else None

    # Common custom-export pattern: {"data": <array>, "sfreq": ..., "ch_names": ...}
    if isinstance(obj, dict) and "data" in obj:
        try:
            arr = np.asarray(obj["data"])
            if arr.ndim == 3:
                # allow metadata keys in same dict
                if sfreq_meta is None:
                    sfreq_meta = infer_sfreq_from_obj(obj)
                if ch_names_meta is None:
                    ch_names_meta = _extract_ch_names(obj)
                return "array", arr, sfreq_meta, ch_names_meta
        except Exception:
            pass

    # Block dictionary pattern: {"part_A": <3D>, "part_B": <3D>, ...}
    if isinstance(obj, dict):
        block_arrays = _select_block_arrays_from_dict(obj)
        if block_arrays:
            if sfreq_meta is None:
                sfreq_meta = infer_sfreq_from_obj(obj)
            if ch_names_meta is None:
                ch_names_meta = _extract_ch_names(obj)
            return "blocks", block_arrays, sfreq_meta, ch_names_meta

    # Plain numpy array
    if isinstance(obj, np.ndarray):
        arr = np.asarray(obj)
        if arr.ndim == 3:
            return "array", arr, sfreq_meta, ch_names_meta

    # MNE Epochs objects (rare but safe)
    if hasattr(obj, "get_data") and hasattr(obj, "info"):
        try:
            arr = obj.get_data()
            sf = float(obj.info.get("sfreq", 0)) or None
            names = list(getattr(obj, "ch_names", [])) or None
            return "array", np.asarray(arr), sf, names
        except Exception:
            pass

    return "unknown", None, sfreq_meta, ch_names_meta


def process_pickle(
    file_path,
    sfreq_fallback=2000.0,
    ch_name_mode="auto_eeg",
    prefer_pickle_ch_names=True,
    nan_policy="drop",
    nan_fill_value=0.0,
):
    """
    Processes a single pickle file and returns an MNE Epochs object.

    Supports:
      1) Single 3D array: (n_epochs, n_channels, n_times)
      2) Dict of 3D arrays (10-min blocks, etc.)
      3) Dict with {"data": <3D array>, "sfreq": ..., "ch_names": ...}

    Parameters
    ----------
    sfreq_fallback:
        Used when no sampling-rate metadata is found in the pickle.
    ch_name_mode:
        "auto_eeg" or "generic".
    prefer_pickle_ch_names:
        If True and the pickle provides channel names matching the channel count,
        those names will be used.

    nan_policy:
        How to handle NaN/Inf values before filtering. One of: "drop", "interp", "fill", "error".
    nan_fill_value:
        Fill value used when nan_policy == "fill", and as fallback when interpolation is impossible.

    Returns
    -------
    epochs : mne.Epochs or None
    """
    obj = load_pickle(file_path)

    kind, payload, sfreq_meta, ch_names_meta = _extract_data_and_metadata(obj)

    if kind == "unknown" or payload is None:
        print(f"Unrecognized data format in {file_path}")
        return None

    def _resolve_names(n_channels):
        if prefer_pickle_ch_names and ch_names_meta and len(ch_names_meta) == n_channels:
            return ch_names_meta
        return infer_ch_names(n_channels, mode=ch_name_mode)

    sfreq_use = sfreq_meta if (sfreq_meta is not None) else float(sfreq_fallback)

    if kind == "blocks":
        print(f"Detected split blocks in {os.path.basename(file_path)}")
        all_epochs = []
        for block_name in sorted(payload.keys()):
            block_data = np.asarray(payload[block_name])
            # Handle NaNs/Infs before creating MNE objects
            block_data, rep = handle_nonfinite_epochs(
                block_data,
                policy=nan_policy,
                fill_value=nan_fill_value,
                verbose_prefix=f"{os.path.basename(file_path)}:{block_name}: ",
            )
            if rep.get('n_nonfinite', 0) > 0:
                print(
                    f"  Non-finite values detected in block '{block_name}': {rep['n_nonfinite']} | "
                    f"policy={rep['policy']} | dropped_epochs={rep.get('n_epochs_dropped', 0)}"
                )
            if block_data.ndim == 3 and block_data.shape[0] == 0:
                print(f"  SKIP block '{block_name}': all epochs removed by NaN policy.")
                continue
            if block_data.ndim != 3:
                print(f"  SKIP block '{block_name}': expected 3D array, got shape {block_data.shape}")
                continue
            print(f"  Processing block: {block_name} with shape {block_data.shape}")
            ch_names = _resolve_names(block_data.shape[1])
            epochs = create_epochs_array(block_data, sfreq=sfreq_use, ch_names=ch_names)
            all_epochs.append(epochs)
        if all_epochs:
            combined_epochs = mne.concatenate_epochs(all_epochs)
            return combined_epochs
        else:
            print(f"No valid blocks found in {file_path}")
            return None

    # kind == "array"
    data = np.asarray(payload)
    # Handle NaNs/Infs before creating MNE objects
    data, rep = handle_nonfinite_epochs(
        data,
        policy=nan_policy,
        fill_value=nan_fill_value,
        verbose_prefix=f"{os.path.basename(file_path)}: ",
    )
    if rep.get('n_nonfinite', 0) > 0:
        print(
            f"Non-finite values detected: {rep['n_nonfinite']} | policy={rep['policy']} | "
            f"dropped_epochs={rep.get('n_epochs_dropped', 0)}"
        )
    if data.ndim == 3 and data.shape[0] == 0:
        print(f"All epochs removed by NaN policy in {file_path}.")
        return None
    if data.ndim != 3:
        print(f"Unrecognized array shape in {file_path}: {data.shape}")
        return None

    print(f"Detected single concatenated array in {os.path.basename(file_path)} with shape {data.shape}")
    ch_names = _resolve_names(data.shape[1])
    epochs = create_epochs_array(data, sfreq=sfreq_use, ch_names=ch_names)
    return epochs


def list_pkl_files_in_directory(directory):
    """Return a sorted list of .pkl files in the given directory."""
    if not os.path.isdir(directory):
        return []
    all_files = os.listdir(directory)
    pkl_files = [f for f in all_files if f.endswith('.pkl')]
    return sorted(pkl_files)


def get_unique_output_path(output_directory, source_filename):
    """
    Creates a unique output file path in the given directory using the source filename.
    The new file name is generated as:
        <source_name>_filtered.fif
    and if a file with that name already exists, an incremental counter is added.
    """
    base_name = os.path.splitext(source_filename)[0]
    candidate = os.path.join(output_directory, f"{base_name}_filtered.fif")
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(output_directory, f"{base_name}_filtered_{counter}.fif")
        counter += 1
    return candidate


##############################################################################
# BUILDING THE UI
##############################################################################

# --- Input Directory UI (for pickle files) ---
input_dir_chooser = FileChooser(
    os.path.expanduser("~"),
    title='Select Directory with Pickle Files',
    show_only_dirs=True,
    select_multiple=False
)

load_input_dir_button = widgets.Button(
    description='Load Input Directory',
    button_style='info'
)

# --- List of Pickle Files ---
pkl_select = widgets.SelectMultiple(
    options=[],
    description='Pickle(s):',
    layout=widgets.Layout(width='600px', height='300px')
)

remove_selected_button = widgets.Button(
    description='Remove Selected',
    button_style='danger',
    tooltip='Remove highlighted items from the list above'
)

# --- Output Directory UI (for saving FIF files) ---
output_dir_chooser = FileChooser(
    os.getcwd(),
    title='Select Output Directory for FIF Files',
    show_only_dirs=True,
    select_multiple=False
)

# --- Sampling-rate & channel-name controls ---
sfreq_fallback_widget = widgets.FloatText(
    value=2000.0,
    description='Fallback sfreq:',
    layout=widgets.Layout(width='220px'),
    tooltip='Used when the pickle does not include sampling-rate metadata.'
)

ch_name_mode_widget = widgets.Dropdown(
    options=[
        ('Auto EEG labels', 'auto_eeg'),
        ('Generic Ch#', 'generic'),
    ],
    value='auto_eeg',
    description='Ch names:',
    layout=widgets.Layout(width='250px'),
    tooltip='How to name channels when channel names are not provided in the pickle.'
)

prefer_pickle_names_widget = widgets.Checkbox(
    value=True,
    description='Use pickle ch_names if available',
    tooltip='If the pickle provides channel names matching the data shape, use them.'
)

# --- NaN/Inf handling controls ---
nan_policy_widget = widgets.Dropdown(
    options=[
        ('Drop epochs with NaNs/Infs (recommended)', 'drop'),
        ('Interpolate NaNs/Infs (linear)', 'interp'),
        ('Fill NaNs/Infs with constant', 'fill'),
        ('Error if NaNs/Infs present', 'error'),
    ],
    value='drop',
    description='NaN policy:',
    layout=widgets.Layout(width='380px'),
    tooltip='How to handle NaN/Inf values in the epoched data before filtering.'
)

nan_fill_value_widget = widgets.FloatText(
    value=0.0,
    description='Fill value:',
    layout=widgets.Layout(width='220px'),
    tooltip='Used when NaN policy is "fill", or as fallback when interpolation is impossible (all-NaN traces).'
)
nan_fill_value_widget.layout.display = 'none'

def _on_nan_policy_change(change):
    if change.get('name') != 'value':
        return
    nan_fill_value_widget.layout.display = 'none' if change['new'] in ('drop', 'error') else 'block'

nan_policy_widget.observe(_on_nan_policy_change, names='value')



# --- Bandpass Filter Parameter Widgets ---
bandpass_l_freq_widget = widgets.Text(
    value='0',
    description='Low freq (Hz):',
    layout=widgets.Layout(width='200px'),
    tooltip='Enter low cutoff frequency in Hz or "None" to disable'
)

bandpass_h_freq_widget = widgets.Text(
    value='45',
    description='High freq (Hz):',
    layout=widgets.Layout(width='200px'),
    tooltip='Enter high cutoff frequency in Hz or "None" to disable'
)

fir_design_widget = widgets.Dropdown(
    options=['firwin', 'firwin2'],
    value='firwin',
    description='FIR design:',
    layout=widgets.Layout(width='200px'),
    tooltip='Select FIR filter design method'
)

filter_length_widget = widgets.Text(
    value='auto',
    description='Filter Length:',
    layout=widgets.Layout(width='250px'),
    tooltip='Enter number of taps (odd integer) or "auto" for automatic selection'
)

# --- Processing Buttons ---
process_selected_button = widgets.Button(
    description='Process & Filter Selected',
    button_style='success',
    tooltip='Process and filter selected pickle files'
)

process_all_button = widgets.Button(
    description='Process & Filter ALL',
    button_style='warning',
    tooltip='Process and filter all pickle files'
)

# --- Output Area for Logging ---
output_area = widgets.Output()

##############################################################################
# CALLBACK FUNCTIONS
##############################################################################

def on_load_input_dir_clicked(b):
    """Load the list of .pkl files from the chosen input directory."""
    with output_area:
        clear_output()
        directory = input_dir_chooser.selected_path
        if not directory or not os.path.isdir(directory):
            print("Invalid input directory selected.")
            return

        pkl_files = list_pkl_files_in_directory(directory)
        if not pkl_files:
            print(f"No .pkl files found in {directory}.")
            pkl_select.options = []
            return

        pkl_select.options = pkl_files
        print(f"Found {len(pkl_files)} pickle files in {directory}:")
        for f in pkl_files:
            print(f"  - {f}")
        print("\nSelect the pickle files you want to process, or use 'Process ALL'.")


def on_remove_selected_clicked(b):
    """Remove highlighted pickle files from the list."""
    with output_area:
        current_options = list(pkl_select.options)
        selected_files = list(pkl_select.value)
        if not selected_files:
            print("No files highlighted to remove.")
            return
        new_options = [f for f in current_options if f not in selected_files]
        pkl_select.options = new_options
        pkl_select.value = ()
        print(f"Removed: {selected_files}")


def process_and_filter_files(file_list):
    """
    For each file in file_list, load the pickle data, create an MNE Epochs object,
    apply the bandpass filter, and save the filtered epochs as a FIF file.
    """
    input_directory = input_dir_chooser.selected_path
    output_directory = output_dir_chooser.selected_path

    if not input_directory or not os.path.isdir(input_directory):
        with output_area:
            print("Invalid input directory. Please select a valid directory containing pickle files.")
        return

    if not output_directory or not os.path.isdir(output_directory):
        with output_area:
            print("Invalid output directory. Please select a valid directory to save FIF files.")
        return

    # --- Parse sampling-rate & naming preferences ---
    sfreq_fallback = float(sfreq_fallback_widget.value)
    ch_name_mode = ch_name_mode_widget.value
    prefer_pickle_names = bool(prefer_pickle_names_widget.value)

    # --- Parse NaN/Inf handling ---
    nan_policy = str(nan_policy_widget.value)
    nan_fill_value = float(nan_fill_value_widget.value)

    # --- Parse Filter Parameters ---
    l_freq_text = bandpass_l_freq_widget.value.strip()
    if l_freq_text.lower() == "none":
        l_freq = None
    else:
        try:
            l_freq = float(l_freq_text)
            if l_freq < 0:
                with output_area:
                    print(f"ERROR: Low frequency must be >= 0. Got: {l_freq}")
                return
        except ValueError:
            with output_area:
                print(f"ERROR: Could not parse low freq '{l_freq_text}' as a float or 'None'.")
            return

    h_freq_text = bandpass_h_freq_widget.value.strip()
    if h_freq_text.lower() == "none":
        h_freq = None
    else:
        try:
            h_freq = float(h_freq_text)
            if h_freq < 0:
                with output_area:
                    print(f"ERROR: High frequency must be >= 0. Got: {h_freq}")
                return
        except ValueError:
            with output_area:
                print(f"ERROR: Could not parse high freq '{h_freq_text}' as float or 'None'.")
            return

    fir_design = fir_design_widget.value

    filter_length_text = filter_length_widget.value.strip()
    if filter_length_text.lower() == "auto":
        filter_length = 'auto'
    else:
        try:
            filter_length = int(filter_length_text)
            if filter_length % 2 == 0:
                with output_area:
                    print(f"ERROR: Filter length must be an odd integer. Got: {filter_length}")
                return
            if filter_length < 1:
                with output_area:
                    print(f"ERROR: Filter length must be positive. Got: {filter_length}")
                return
        except ValueError:
            with output_area:
                print(f"ERROR: Could not parse filter length '{filter_length_text}' as integer or 'auto'.")
            return

    with output_area:
        print(f"Processing {len(file_list)} file(s).")
        print(f"Fallback sfreq: {sfreq_fallback}")
        print(f"Channel naming mode: {ch_name_mode}")
        print(f"Prefer pickle names: {prefer_pickle_names}\n")

    for pkl_file in tqdm(file_list, desc="Processing files"):
        pkl_path = os.path.join(input_directory, pkl_file)
        with output_area:
            print(f"\nProcessing file: {pkl_file}")
        if not os.path.isfile(pkl_path):
            with output_area:
                print(f"  SKIP: File not found - {pkl_file}")
            continue

        try:
            epochs = process_pickle(
                pkl_path,
                sfreq_fallback=sfreq_fallback,
                ch_name_mode=ch_name_mode,
                prefer_pickle_ch_names=prefer_pickle_names,
                nan_policy=nan_policy,
                nan_fill_value=nan_fill_value,
            )
            if epochs is None:
                with output_area:
                    print(f"  SKIP: Could not process {pkl_file}")
                continue
        except Exception as e:
            with output_area:
                print(f"  ERROR processing {pkl_file}: {e}")
            continue

        # --- Apply Bandpass Filter ---
        try:
            epochs_filtered = epochs.copy().filter(
                l_freq=l_freq,
                h_freq=h_freq,
                fir_design=fir_design,
                filter_length=filter_length,
                verbose=False,
            )
        except Exception as e:
            with output_area:
                print(f"  ERROR applying filter to {pkl_file}: {e}")
            continue

        # --- Save to FIF ---
        try:
            save_path = get_unique_output_path(output_directory, pkl_file)
            epochs_filtered.save(save_path, overwrite=False, verbose=False)
            with output_area:
                print(f"  Saved filtered FIF => {save_path}")
        except Exception as e:
            with output_area:
                print(f"  ERROR saving FIF for {pkl_file}: {e}")
            continue

    with output_area:
        print("\nDone.")


def on_process_selected_clicked(b):
    files = list(pkl_select.value)
    if not files:
        with output_area:
            clear_output()
            print("No pickle files selected.")
        return
    process_and_filter_files(files)


def on_process_all_clicked(b):
    files = list(pkl_select.options)
    if not files:
        with output_area:
            clear_output()
            print("No pickle files available.")
        return
    process_and_filter_files(files)


##############################################################################
# WIRING CALLBACKS
##############################################################################
load_input_dir_button.on_click(on_load_input_dir_clicked)
remove_selected_button.on_click(on_remove_selected_clicked)
process_selected_button.on_click(on_process_selected_clicked)
process_all_button.on_click(on_process_all_clicked)

##############################################################################
# DISPLAY THE UI
##############################################################################

controls_row1 = HBox([
    load_input_dir_button,
    remove_selected_button,
])

controls_row2 = HBox([
    Label("Input directory:"),
    input_dir_chooser,
])

controls_row3 = HBox([
    Label("Output directory:"),
    output_dir_chooser,
])

controls_row4 = HBox([
    sfreq_fallback_widget,
    ch_name_mode_widget,
    prefer_pickle_names_widget,
])

controls_row4b = HBox([
    nan_policy_widget,
    nan_fill_value_widget,
])

controls_row5 = HBox([
    bandpass_l_freq_widget,
    bandpass_h_freq_widget,
    fir_design_widget,
    filter_length_widget,
])

controls_row6 = HBox([
    process_selected_button,
    process_all_button,
])

ui = VBox([
    controls_row1,
    controls_row2,
    pkl_select,
    controls_row3,
    controls_row4,
    controls_row4b,
    controls_row5,
    controls_row6,
    output_area,
])

display(ui)
