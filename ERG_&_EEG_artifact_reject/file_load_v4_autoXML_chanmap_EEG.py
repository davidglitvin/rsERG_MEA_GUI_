# Step 1: Import Necessary Libraries
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import xml.etree.ElementTree as ET

import ipywidgets as widgets
from ipywidgets import (
    IntText,
    SelectMultiple,
    Button,
    ToggleButtons,
    VBox,
    HBox,
    Label,
    Accordion,
    Output,
    Textarea
)
from IPython.display import display, clear_output
from tqdm.notebook import tqdm

# For the in-notebook directory chooser:
from ipyfilechooser import FileChooser

# Check for python-pptx
try:
    from pptx import Presentation
    from pptx.util import Inches
    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False
    print("Warning: python-pptx not installed, PPT export will be disabled.")

##############################################################################
# 1) HELPER FUNCTIONS
##############################################################################
def strip_datetime_prefix(basename):
    """
    Strips the date-time prefix from the basename.

    Example:
    Input: "2025-01-14_11-08-39_rd1_59_ChA-P4-LE_ChB-P2-RE_ind_1"
    Output: "rd1_59_ChA-P4-LE_ChB-P2-RE_ind_1"
    """
    parts = basename.split('_')
    if len(parts) > 2:
        # Join from the third part onwards
        return '_'.join(parts[2:])
    else:
        return basename


##############################################################################
# 1B) OPEN EPHYS XML METADATA HELPERS (AUTO-DETECT SFREQ/CHANNELS/CHANMAP)
##############################################################################
def find_nearest_settings_xml(start_path, max_levels=14):
    """
    Locate an Open Ephys 'settings.xml' by walking upward from a given path.
    Works with typical folder structures such as:
      Record Node XXX/experimentY/recordingZ/continuous/.../continuous.dat
    """
    if not start_path:
        return None

    cur = start_path
    if os.path.isfile(cur):
        cur = os.path.dirname(cur)

    for _ in range(max_levels):
        candidate = os.path.join(cur, "settings.xml")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    return None


def parse_settings_xml(xml_path):
    """
    Parse Open Ephys settings.xml to extract:
      - sampling rate (Hz)
      - channel_count for the acquisition stream
      - enabled indices from Channel Map (if present)
    """
    if not xml_path or not os.path.isfile(xml_path):
        raise FileNotFoundError(f"settings.xml not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    metadata = {
        "sampling_rate": None,
        "n_channels": None,
        "enabled_indices": []
    }

    # Prefer Record Node stream
    record_node = None
    for proc in root.findall(".//PROCESSOR"):
        if proc.get("name") == "Record Node":
            record_node = proc
            break

    def _extract_stream_info(proc_node):
        if proc_node is None:
            return None, None
        streams = proc_node.findall("./STREAM")
        for st in streams:
            if st.get("name") == "acquisition_board":
                return st.get("sample_rate"), st.get("channel_count")
        for st in streams:
            cc = st.get("channel_count")
            sr = st.get("sample_rate")
            if cc is not None or sr is not None:
                return sr, cc
        return None, None

    sr_str, ch_count_str = _extract_stream_info(record_node)

    # Fallback to Acquisition Board
    if sr_str is None or ch_count_str is None:
        acq_node = None
        for proc in root.findall(".//PROCESSOR"):
            if proc.get("name") == "Acquisition Board":
                acq_node = proc
                break
        sr2, cc2 = _extract_stream_info(acq_node)
        sr_str = sr_str or sr2
        ch_count_str = ch_count_str or cc2

    if sr_str is not None:
        try:
            metadata["sampling_rate"] = float(sr_str)
        except Exception:
            metadata["sampling_rate"] = None

    if ch_count_str is not None:
        try:
            metadata["n_channels"] = int(ch_count_str)
        except Exception:
            metadata["n_channels"] = None

    # Channel Map enabled channels
    channel_map_node = None
    for proc in root.findall(".//PROCESSOR"):
        if proc.get("name") == "Channel Map":
            channel_map_node = proc
            break

    if channel_map_node is not None:
        custom_stream = channel_map_node.find("./CUSTOM_PARAMETERS/STREAM")
        if custom_stream is not None:
            enabled = []
            for ch_tag in custom_stream.findall("CH"):
                idx_str = ch_tag.get("index")
                enabled_str = ch_tag.get("enabled")
                if idx_str is not None and enabled_str == "1":
                    try:
                        enabled.append(int(idx_str))
                    except Exception:
                        pass
            metadata["enabled_indices"] = sorted(set(enabled))

    return metadata


def get_recording_metadata(recording_folder, cache=None,
                           default_sfreq=10000.0, default_n_channels=64):
    """
    Get per-recording metadata using the nearest settings.xml.
    """
    if cache is not None and recording_folder in cache:
        return cache[recording_folder]

    xml_path = find_nearest_settings_xml(recording_folder)

    sfreq = default_sfreq
    n_ch = default_n_channels
    enabled = []

    if xml_path:
        try:
            md = parse_settings_xml(xml_path)
            if md.get("sampling_rate") is not None:
                sfreq = float(md["sampling_rate"])
            if md.get("n_channels") is not None:
                n_ch = int(md["n_channels"])
            enabled = md.get("enabled_indices", []) or []
        except Exception:
            pass

    meta = {
        "sfreq": sfreq,
        "n_channels_total": n_ch,
        "xml_path": xml_path,
        "enabled_indices": enabled
    }

    if cache is not None:
        cache[recording_folder] = meta

    return meta


def compute_channel_options(recording_paths):
    """
    Compute channel label options using Channel Map enabled indices if available.
    Uses union across supplied recording paths.

    Also includes the presumed ADC tail (last 8 channels) when channel_count is known,
    so optional EEG selection is always possible.
    """
    enabled_union = set()
    max_ch = 0

    for p in recording_paths:
        try:
            meta = get_recording_metadata(p, cache=recording_metadata)
            max_ch = max(max_ch, int(meta.get("n_channels_total") or 0))
            enabled = meta.get("enabled_indices") or []
            if enabled:
                enabled_union.update(enabled)
        except Exception:
            continue

    if enabled_union:
        if max_ch and max_ch > 0:
            adc_start = max(0, int(max_ch) - 8)
            enabled_union.update(range(adc_start, int(max_ch)))
        options = [f"Ch{i+1}" for i in sorted(enabled_union)]
    else:
        if max_ch <= 0:
            max_ch = 64
        options = [f"Ch{ch}" for ch in range(1, max_ch + 1)]

    return options, max_ch, enabled_union


def get_eeg_candidate_labels(total_channels):
    """
    EEG ADC channels are assumed to be the last 8 channels in the stream.
    """
    if total_channels is None:
        total_channels = 64
    total_channels = int(total_channels)
    start = max(1, total_channels - 8 + 1)
    return [f"Ch{ch}" for ch in range(start, total_channels + 1)]


def get_eeg_auto_labels(total_channels, n_eeg):
    """
    Returns last n_eeg labels from the presumed ADC tail of the stream.
    """
    candidates = get_eeg_candidate_labels(total_channels)
    n_eeg = int(max(0, min(n_eeg, len(candidates))))
    if n_eeg == 0:
        return []
    return candidates[-n_eeg:]


def required_channel_count():
    """
    Number of channels required given the EEG toggle state.
    """
    try:
        if include_eeg_widget.value == 'Yes':
            return 16 + int(eeg_count_widget.value)
    except Exception:
        pass
    return 16


def segment_data_into_5s_epochs(data, sfreq=10000, n_channels=16):
    """
    Segments raw data into 5-second epochs.

    Parameters:
    - data: numpy array of shape (n_samples, n_channels)
    - sfreq: Sampling frequency (samples per second)
    - n_channels: Number of channels

    Returns:
    - epoched_data: numpy array of shape (n_epochs, n_channels, epoch_length)
    """
    epoch_duration = 5  # seconds

    # Ensure integer epoch length even when sfreq is float (e.g., 2000.0)
    epoch_length = int(round(float(sfreq) * epoch_duration))
    epoch_length = max(epoch_length, 1)

    total_samples = int(data.shape[0])
    n_epochs = int(total_samples // epoch_length)

    if n_epochs == 0:
        return np.empty((0, n_channels, epoch_length), dtype=data.dtype)

    end = int(n_epochs * epoch_length)
    truncated_data = data[:end, :]
    epoched_data = truncated_data.reshape(n_epochs, epoch_length, n_channels)
    # Move channels to second dimension => (n_epochs, n_channels, epoch_length)
    epoched_data = np.transpose(epoched_data, (0, 2, 1))
    return epoched_data

def split_into_10min_blocks(epoched_data):
    """
    Splits epoched data into 10-minute blocks.

    Parameters:
    - epoched_data: numpy array of shape (N, 16, 50000)

    Returns:
    - parts_dict: dictionary with keys like 'part_A', 'part_B', etc., each containing a numpy array
    """
    n_epochs = epoched_data.shape[0]
    block_size = 120  # 10 minutes = 120 epochs (5s each)
    parts_dict = {}
    
    start_idx = 0
    part_counter = 0
    while (start_idx + block_size) <= n_epochs:
        end_idx = start_idx + block_size
        block = epoched_data[start_idx:end_idx, :, :]
        label = f"part_{chr(ord('A') + part_counter)}"  # "part_A", "part_B", ...
        parts_dict[label] = block
        part_counter += 1
        start_idx = end_idx

    return parts_dict

##############################################################################
# 2) HELPER FUNCTIONS: UPDATED TO FIND 'continuous.dat' FILES FLEXIBLY
##############################################################################
def find_open_ephys_recordings_in_dir(base_dir):
    """
    Finds subfolders within base_dir that contain 'continuous.dat'.

    Parameters:
    - base_dir: The directory to search within.

    Returns:
    - matches: List of paths to directories containing 'continuous.dat'.
    """
    matches = []
    for root, dirs, files in os.walk(base_dir):
        if "continuous.dat" in files:
            # Add the full path to the continuous folder
            matches.append(root)
    return matches

##############################################################################
# 3) PLOTTING FUNCTIONS
##############################################################################
def exclude_traces(
    psd_array,
    freqs,
    low_band=(1, 3),
    low_band_threshold=3.0,  # multiple of mean
    test_bands=[(7, 9), (9, 11), (11, 13), (13, 15), (15, 17), (17, 19), (19, 21)],
    test_band_threshold=10.0,  # multiple of mean
    test_band_count_threshold=None
):
    if test_band_count_threshold is None:
        test_band_count_threshold = len(test_bands) // 2

    mean_psd = np.mean(psd_array, axis=0)
    excluded_traces = []
    kept_traces = []

    low_band_indices = np.where((freqs >= low_band[0]) & (freqs <= low_band[1]))[0]
    band_indices = [
        np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
        for band in test_bands
    ]

    for i, trace in enumerate(psd_array):
        # 1) Check low-frequency extreme outlier
        if np.any(trace[low_band_indices] > low_band_threshold * mean_psd[low_band_indices]):
            excluded_traces.append(i)
            continue

        # 2) Check repeated suprathreshold events in test bands
        suprathreshold_count = 0
        for indices in band_indices:
            if np.any(trace[indices] > test_band_threshold * mean_psd[indices]):
                suprathreshold_count += 1

        if suprathreshold_count >= test_band_count_threshold:
            excluded_traces.append(i)
        else:
            kept_traces.append(i)

    return kept_traces, excluded_traces

def plot_psds_with_exclusion(
    ax,
    psd_array,
    freqs,
    kept_traces,
    excluded_traces,
    original_mean_psd,
    title,
    # Booleans to show/hide
    show_kept=True,
    show_excluded=True,
    show_original_mean=True,
    show_new_mean=True,
    show_vertical_lines=True,
    # Colors, alpha, etc.
    color_kept="lightgray",
    color_excluded="red",
    color_old_mean="blue",
    color_new_mean="green",
    alpha_excluded=0.05,
    alpha_kept=0.7,
    # Font settings
    title_fontsize=8,
    axis_label_fontsize=8,
    legend_fontsize=8,
    tick_label_fontsize=8,
    max_title_length=40,
    # Vertical lines
    vertical_lines=None,
    vertical_line_color="black",
    vertical_line_style="--",
    vertical_line_alpha=0.6
):
    # Truncate extremely long titles if desired
    if len(title) > max_title_length:
        title = title[:max_title_length] + "..."

    # Plot kept traces
    if show_kept:
        for idx_i, idx in enumerate(kept_traces):
            label = "Kept Trace" if idx_i == 0 else None
            ax.plot(freqs, psd_array[idx], color=color_kept, alpha=alpha_kept, label=label)

    # Plot original (old) mean
    if show_original_mean:
        ax.plot(freqs, original_mean_psd, color=color_old_mean, linewidth=2, label="Original Mean")

    # Plot new mean (only among kept)
    if show_new_mean and len(kept_traces) > 0:
        new_mean_psd = np.mean(psd_array[kept_traces], axis=0)
        ax.plot(freqs, new_mean_psd, color=color_new_mean, linewidth=2, label="New Mean")

    # Plot excluded traces
    if show_excluded:
        for idx_j, idx in enumerate(excluded_traces):
            label = "Excluded Trace" if idx_j == 0 else None
            ax.plot(freqs, psd_array[idx], color=color_excluded, alpha=alpha_excluded, 
                    zorder=10, label=label)

    # Optionally plot vertical lines at certain frequencies
    if show_vertical_lines and vertical_lines is not None:
        for vfreq in vertical_lines:
            ax.axvline(
                vfreq, 
                color=vertical_line_color, 
                linestyle=vertical_line_style, 
                alpha=vertical_line_alpha
            )

    # Axis labels, title, and legend with smaller font
    ax.set_xlabel("Frequency (Hz)", fontsize=axis_label_fontsize)
    ax.set_ylabel("PSD (V²/Hz)", fontsize=axis_label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.legend(loc="upper right", fontsize=legend_fontsize)
    ax.tick_params(axis='both', labelsize=tick_label_fontsize)

def plot_psds_with_dropped_traces(
    psds_dict,
    rows_of_psds,
    # Exclusion parameters
    low_band=(1,3),
    low_band_threshold=3.0,
    test_bands=[(7,9),(9,11),(11,13),(13,15),(15,17),(17,19),(19,21)],
    test_band_threshold=10.0,
    test_band_count_threshold=None,
    # Axis ranges
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    # Booleans for toggles
    show_kept=True,
    show_excluded=True,
    show_original_mean=True,
    show_new_mean=True,
    show_vertical_lines=True,
    # Colors
    color_kept="lightgray",
    color_excluded="red",
    color_old_mean="blue",
    color_new_mean="green",
    # Layout
    num_cols=4,
    # Font sizes
    title_fontsize=8,
    axis_label_fontsize=8,
    legend_fontsize=8,
    tick_label_fontsize=8,
    max_title_length=40,
    # Vertical lines
    vertical_lines=None,
    vertical_line_color="black",
    vertical_line_style="--",
    vertical_line_alpha=0.6
):
    figures = []  # We'll store each figure so we can export them later

    for row_idx, row in enumerate(rows_of_psds, start=1):
        valid_keys = [k for k in row if k in psds_dict]
        num_plots = len(valid_keys)
        if num_plots == 0:
            continue

        num_rows = math.ceil(num_plots / num_cols)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
        plt.subplots_adjust(hspace=0.5)

        if num_rows * num_cols == 1:
            axes = [axes]  # single plot for 1 key
        else:
            axes = axes.flatten()

        for idx, key in enumerate(valid_keys):
            ax = axes[idx]
            psd_output = psds_dict[key]
            psd_data, freq_data = psd_output.get('psd'), psd_output.get('freqs')
            original_mean_psd = np.mean(psd_data, axis=0)

            kept_traces, excluded_traces = exclude_traces(
                psd_data,
                freq_data,
                low_band=low_band,
                low_band_threshold=low_band_threshold,
                test_bands=test_bands,
                test_band_threshold=test_band_threshold,
                test_band_count_threshold=test_band_count_threshold
            )

            plot_psds_with_exclusion(
                ax=ax,
                psd_array=psd_data,
                freqs=freq_data,
                kept_traces=kept_traces,
                excluded_traces=excluded_traces,
                original_mean_psd=original_mean_psd,
                title=key.replace("_filtered_PSD", ""),
                show_kept=show_kept,
                show_excluded=show_excluded,
                show_original_mean=show_original_mean,
                show_new_mean=show_new_mean,
                show_vertical_lines=show_vertical_lines,
                color_kept=color_kept,
                color_excluded=color_excluded,
                color_old_mean=color_old_mean,
                color_new_mean=color_new_mean,
                title_fontsize=title_fontsize,
                axis_label_fontsize=axis_label_fontsize,
                legend_fontsize=legend_fontsize,
                tick_label_fontsize=tick_label_fontsize,
                max_title_length=max_title_length,
                vertical_lines=vertical_lines,
                vertical_line_color=vertical_line_color,
                vertical_line_style=vertical_line_style,
                vertical_line_alpha=vertical_line_alpha
            )

            # x/y range
            if x_min is not None or x_max is not None:
                ax.set_xlim(x_min, x_max)
            if y_min is not None or y_max is not None:
                ax.set_ylim(y_min, y_max)

        # Turn off any unused subplots
        for ax in axes[num_plots:]:
            ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(f"Row {row_idx}", fontsize=title_fontsize + 2)
        figures.append(fig)
        plt.show()

    return figures  # Return a list of all figure objects

def group_keys_by_rows(psd_keys, row_size=4):
    rows = []
    for i in range(0, len(psd_keys), row_size):
        rows.append(psd_keys[i:i+row_size])
    return rows

##############################################################################
# 4) GLOBAL VARIABLES & WIDGETS
##############################################################################
# Cache for per-recording metadata parsed from settings.xml
recording_metadata = {}
# Max detected channel count across loaded recordings (used for EEG auto-select)
max_detected_channels = 64

directories_list = []  # Store user-chosen directories

# A) FileChooser to pick directories
fc = FileChooser(os.getcwd())
fc.title = 'Select a Directory'
fc.show_only_dirs = True  # Only show directories in the chooser

add_directory_button = Button(
    description='Add Selected Directory',
    button_style='info'
)

# B) SelectMultiple widget to list currently added directories
dir_remove_select = SelectMultiple(
    options=[],
    description='Added Directories:',
    layout=widgets.Layout(width='600px', height='120px')
)

# C) Button to remove selected directories
remove_directory_button = Button(
    description='Remove Selected Directories',
    button_style='danger'
)

# D) Accordion to display subfolders found in each directory
directory_accordion = VBox()

# E) SelectMultiple to choose recordings to process
recording_select = SelectMultiple(
    options=[],
    description='All Recordings:',
    layout=widgets.Layout(width='900px', height='350px'),
    style={'description_width': 'initial'}
)

# F) Button to scan added directories and find recordings
load_dirs_button = Button(
    description='Load from Added Directories',
    button_style='info'
)

# G) Text widget to specify output directory
output_dir_text = widgets.Text(
    value='',
    placeholder='Type or paste output folder path here',
    description='Output Directory:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='700px')
)

# **New**: Filename Prefix and Suffix
output_filename_prefix_widget = widgets.Text(
    value='',
    placeholder='Optional prefix for output filenames',
    description='Filename Prefix:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='300px')
)

output_filename_suffix_widget = widgets.Text(
    value='',
    placeholder='Optional suffix for output filenames',
    description='Filename Suffix:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='300px')
)

# H) Container for per-recording crop options
crop_options_box = VBox()

# I) Toggle to optionally split concatenated data into 10-minute blocks
split_10min_toggle = ToggleButtons(
    options=['No', 'Yes'],
    value='No',
    description='Split 10min?',
    button_style='',
    tooltips=[
        'Do not split into 10-minute blocks',
        'Split concatenated data into 10-minute blocks'
    ],
    layout=widgets.Layout(width='150px')
)

# J) Toggle to decide whether to concatenate selected recordings
concatenate_toggle = ToggleButtons(
    options=['No', 'Yes'],
    value='No',
    description='Concatenate Recordings?',
    button_style='',
    tooltips=[
        'Process recordings individually',
        'Concatenate recordings before processing'
    ],
    layout=widgets.Layout(width='200px')
)

# K) Button to start processing
process_button = Button(
    description='Process Selected Recordings',
    button_style='success'
)

# L) Output area for logs
log_output = Output()

# M) **New**: SelectMultiple widget for channel selection (Channels 1-64)
channel_select_widget = SelectMultiple(
    options=[f"Ch{ch}" for ch in range(1, 129)],
    value=[f"Ch{ch}" for ch in range(1, 17)],  # Default selection: first 16 channels
    description='Select Channels:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='600px', height='200px'),
    tooltip='Select the required number of channels (16 for eyes, optionally + EEG).'
)

# N) **New**: Label to display channel selection instructions
channel_selection_label = Label(
    value="STEP J) Select eye channels (and optional EEG channels) to include in processing:"
)

# O) **New**: ToggleButtons to select Load Mode (Auto Load vs Manual Load)
load_mode_widget = ToggleButtons(
    options=['Auto Load', 'Manual Load'],
    value='Auto Load',
    description='Load Mode:',
    button_style='',
    tooltips=[
        'Automatically load exactly 16 predefined channels',
        'Manually select eye channels (and optional EEG channels) from the available channels'
    ],
    layout=widgets.Layout(width='200px')
)


# Q) Toggle to decide whether EEG (ADC tail channels) should be included
include_eeg_widget = ToggleButtons(
    options=['No', 'Yes'],
    value='No',
    description='Include EEG (ADC)?',
    button_style='',
    tooltips=[
        'Process only the 16 eye channels',
        'Process 16 eye channels plus a selectable number of EEG channels'
    ],
    layout=widgets.Layout(width='220px')
)

# R) Number of EEG channels to include (from the last 8 channels of the stream)
eeg_count_widget = widgets.IntSlider(
    value=1,
    min=1,
    max=8,
    step=1,
    description='EEG ch count:',
    disabled=True,
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='320px')
)

def on_include_eeg_change(change):
    if change.get('name') == 'value':
        eeg_count_widget.disabled = (change['new'] != 'Yes')
        # Refresh Auto Load selection if applicable
        try:
            on_load_mode_change({'name': 'value', 'new': load_mode_widget.value})
        except Exception:
            pass
        try:
            update_verification_window()
        except Exception:
            pass

include_eeg_widget.observe(on_include_eeg_change, names='value')

def on_eeg_count_change(change):
    if change.get('name') == 'value':
        try:
            if include_eeg_widget.value == 'Yes' and load_mode_widget.value == 'Auto Load':
                on_load_mode_change({'name': 'value', 'new': 'Auto Load'})
        except Exception:
            pass
        try:
            update_verification_window()
        except Exception:
            pass

eeg_count_widget.observe(on_eeg_count_change, names='value')

# P) **New**: Verification Window (VBox)
verification_output = Output()

##############################################################################
# 5) DIRECTORY MANIPULATION CALLBACKS
##############################################################################
def on_add_directory_clicked(b):
    """
    Adds the selected directory from FileChooser to directories_list and updates the UI.
    """
    with log_output:
        clear_output()
        selected_dir = fc.selected_path
        if not selected_dir:
            print("Please select a directory using the FileChooser.")
            return
        if not os.path.isdir(selected_dir):
            print(f"Invalid directory: {selected_dir}")
            return
        if selected_dir not in directories_list:
            directories_list.append(selected_dir)
            print(f"Added directory: {selected_dir}")
        else:
            print(f"Directory already in list: {selected_dir}")
    
    # Update the remove-directory widget
    dir_remove_select.options = directories_list

add_directory_button.on_click(on_add_directory_clicked)

def on_remove_directory_clicked(b):
    """
    Removes selected directories from directories_list and updates the UI.
    """
    with log_output:
        clear_output()
        selected_to_remove = list(dir_remove_select.value)
        if not selected_to_remove:
            print("No directories selected for removal.")
            return
        
        removed_count = 0
        for d in selected_to_remove:
            if d in directories_list:
                directories_list.remove(d)
                removed_count += 1
        
        if removed_count > 0:
            print(f"Removed {removed_count} directories:")
            for d in selected_to_remove:
                print(f"  - {d}")
        else:
            print("Nothing removed.")
        
    # Update the remove-directory widget
    dir_remove_select.options = directories_list

remove_directory_button.on_click(on_remove_directory_clicked)

##############################################################################
# 6) LOGIC TO FIND OPEN EPHYS RECORDINGS
##############################################################################
def on_load_dirs_clicked(b):
    """
    Scans added directories to find recordings and updates the UI.
    Additionally, reports the number of 5-second epochs for each loaded recording.
    """
    with log_output:
        clear_output()
        if not directories_list:
            print("No directories to load.")
            return
        
        all_recordings = []
        children_for_accordion = []
        
        for d in directories_list:
            subfolders = find_open_ephys_recordings_in_dir(d)
            all_recordings.extend(subfolders)
            
            out = Output(layout=widgets.Layout(width='400px'))
            with out:
                if subfolders:
                    for path in subfolders:
                        basename = os.path.basename(path)
                        dat_file = os.path.join(path, "continuous.dat")
                        if not os.path.exists(dat_file):
                            print(f"{basename}: 'continuous.dat' not found.")
                            continue
                        # Calculate number of epochs
                        file_size_bytes = os.path.getsize(dat_file)
                        bytes_per_sample = 2  # int16
                        # Attempt to pull sfreq/n_channels from the nearest settings.xml
                        meta = get_recording_metadata(path, cache=recording_metadata)
                        n_channels = meta['n_channels_total']
                        sfreq = meta['sfreq']
                        epoch_duration = 5  # seconds
                        epoch_length = sfreq * epoch_duration  # samples per epoch

                        total_samples = file_size_bytes / bytes_per_sample
                        number_of_channels = n_channels
                        number_of_frames = total_samples / number_of_channels
                        number_of_epochs = int(number_of_frames // epoch_length)

                        print(f"{strip_datetime_prefix(basename)}: {number_of_epochs} epochs")
                else:
                    print("No Open Ephys recordings found here.")
            
            acc = Accordion(children=[out])
            acc.set_title(0, d)
            children_for_accordion.append(acc)

        directory_accordion.children = children_for_accordion
        
        # Deduplicate and sort
        all_recordings = list(set(all_recordings))
        all_recordings.sort()

        recording_select.options = all_recordings

        # Update channel selection options using Channel Map where available
        try:
            options, max_ch, _enabled_union = compute_channel_options(all_recordings)
            global max_detected_channels
            max_detected_channels = int(max_ch) if max_ch else max_detected_channels
            channel_select_widget.options = options
        except Exception:
            pass

        print(f"Loaded {len(all_recordings)} recordings from {len(directories_list)} directories.")

load_dirs_button.on_click(on_load_dirs_clicked)

##############################################################################
# 7) PER-RECORDING CROP OPTIONS CALLBACKS
##############################################################################
# We will dynamically create crop options for each selected recording
# and store them in a dictionary for easy access during processing
crop_settings_widgets = {}  # Key: recording path, Value: widget VBox

def create_crop_widgets_for_recording(recording_path):
    """
    Creates crop option widgets for a single recording.

    Parameters:
    - recording_path: The path of the recording.

    Returns:
    - widget_box: VBox containing widgets for crop options.
    """
    basename = os.path.basename(recording_path)
    
    # Toggle to apply crop
    apply_crop_toggle = ToggleButtons(
        options=['No', 'Yes'],
        value='No',
        description='Apply Crop?',
        layout=widgets.Layout(width='150px')
    )
    
    # Start and End time inputs
    start_time = IntText(
        value=0,
        description='Start time (s):',
        disabled=True,
        layout=widgets.Layout(width='200px')
    )
    end_time = IntText(
        value=1800,
        description='End time (s):',
        disabled=True,
        layout=widgets.Layout(width='200px')
    )
    
    # Callback to enable/disable time inputs based on toggle
    def on_apply_crop_toggle_change(change):
        if change['name'] == 'value':
            if change['new'] == 'Yes':
                start_time.disabled = False
                end_time.disabled = False
            else:
                start_time.disabled = True
                end_time.disabled = True
    
    apply_crop_toggle.observe(on_apply_crop_toggle_change, names='value')
    
    # Assemble the widgets
    widget_box = VBox([
        Label(f"Recording: {strip_datetime_prefix(basename)}"),
        HBox([apply_crop_toggle]),
        HBox([start_time, end_time])
    ])
    
    return widget_box

def update_crop_options_box(*args):
    """
    Updates the crop_options_box based on the current selection in recording_select.
    """
    selected_recordings = list(recording_select.value)
    
    # Clear existing widgets
    crop_options_box.children = []
    
    # Remove widgets for recordings that are no longer selected
    to_remove = [path for path in crop_settings_widgets if path not in selected_recordings]
    for path in to_remove:
        del crop_settings_widgets[path]
    
    # Add widgets for newly selected recordings
    for path in selected_recordings:
        if path not in crop_settings_widgets:
            crop_settings_widgets[path] = create_crop_widgets_for_recording(path)
    
    # Display all crop widgets
    crop_widgets = [crop_settings_widgets[path] for path in selected_recordings]
    crop_options_box.children = crop_widgets

# Observe changes in recording_select
recording_select.observe(update_crop_options_box, names='value')

##############################################################################
# 8) CHANNEL SELECTION CALLBACKS
##############################################################################
def on_load_mode_change(change):
    """
    Adjusts the channel selection widget based on the selected load mode.
    """
    if change['name'] == 'value':
        load_mode = change['new']
        if load_mode == 'Auto Load':
            # Automatically select predefined eye channels (16)
            auto_eye_channels = [f"Ch{ch}" for ch in range(17, 25)] + [f"Ch{ch}" for ch in range(41, 49)]
            auto_selected_channels = list(auto_eye_channels)

            # Optionally append EEG ADC channels (assumed to be the last 8 channels)
            if include_eeg_widget.value == 'Yes':
                n_eeg = int(eeg_count_widget.value)
                auto_selected_channels += get_eeg_auto_labels(max_detected_channels, n_eeg)

            channel_select_widget.value = auto_selected_channels
            channel_select_widget.disabled = True  # Disable manual selection
            print("Auto Load: Predefined 16 channels have been selected and manual selection is disabled.")
        elif load_mode == 'Manual Load':
            # Allow user to manually select channels
            channel_select_widget.disabled = False
            # Reset selection to default or keep current selection
            req = required_channel_count()
            if len(channel_select_widget.value) != req:
                channel_select_widget.value = [f"Ch{ch}" for ch in range(1, req + 1)]  # Default to first required channels
            print(f"Manual Load: Please select exactly {required_channel_count()} channels from the list.")

# Attach the callback to load_mode_widget
load_mode_widget.observe(on_load_mode_change, names='value')

# Update available channel options when the recording selection changes
def update_channel_options_on_recording_change(change):
    if change.get('name') == 'value':
        try:
            selected = list(change.get('new') or [])
            if not selected:
                selected = list(recording_select.options or [])
            if selected:
                options, max_ch, _enabled = compute_channel_options(selected)
                global max_detected_channels
                max_detected_channels = int(max_ch) if max_ch else max_detected_channels
                channel_select_widget.options = options
                if load_mode_widget.value == 'Auto Load':
                    on_load_mode_change({'name': 'value', 'new': 'Auto Load'})
        except Exception:
            pass

recording_select.observe(update_channel_options_on_recording_change, names='value')


##############################################################################
# 9) VERIFICATION WINDOW CALLBACKS
##############################################################################
def update_verification_window():
    """
    Updates the verification window to display currently selected channels.
    Provides options to remove channels if needed.
    """
    with verification_output:
        clear_output()
        selected_channels = list(channel_select_widget.value)
        req = required_channel_count()
        if len(selected_channels) != req:
            print(f"ERROR: Exactly {req} channels must be selected. Currently selected: {len(selected_channels)}")
            return
        
        # Display the list of selected channels with Remove buttons
        display_label = Label("Verify your channel selection below. Remove any incorrectly selected channels:")
        display(display_label)
        
        channel_buttons = []
        for ch in selected_channels:
            btn = Button(
                description='Remove',
                button_style='danger',
                tooltip=f'Remove {ch}',
                layout=widgets.Layout(width='80px')
            )
            # Attach callback with closure to capture channel name
            btn.on_click(lambda b, ch=ch: on_remove_channel_clicked(ch))
            channel_buttons.append(HBox([Label(ch), btn]))
        
        display(VBox(channel_buttons))
        
        # Instruction to re-select channels after removal
        print(f"\nAfter removing channels, please re-select channels to maintain exactly {required_channel_count()} channels.")

def on_remove_channel_clicked(channel):
    """
    Removes a channel from the selection.
    """
    current_selection = list(channel_select_widget.value)
    if channel in current_selection:
        current_selection.remove(channel)
        channel_select_widget.value = current_selection
        print(f"Removed {channel} from selection.")
        # Update the verification window
        update_verification_window()
    else:
        print(f"Channel {channel} not found in selection.")

def on_confirm_verification_clicked(b):
    """
    Confirms the verification of channel selection.
    """
    with verification_output:
        clear_output()
        selected_channels = list(channel_select_widget.value)
        req = required_channel_count()
        if len(selected_channels) != req:
            print(f"ERROR: Exactly {req} channels must be selected. Currently selected: {len(selected_channels)}")
            return
        print("Channel selection verified successfully.")

# Create a Confirm button for verification
confirm_verification_button = Button(
    description='Confirm Selection',
    button_style='success',
    tooltip='Confirm that the selected channels are correct'
)

confirm_verification_button.on_click(on_confirm_verification_clicked)

def on_channel_selection_change(change):
    """
    Updates the verification window whenever the channel selection changes.
    """
    if change['name'] == 'value':
        update_verification_window()

# Attach the callback to channel_select_widget
channel_select_widget.observe(on_channel_selection_change, names='value')

##############################################################################
# 10) PROCESSING LOGIC
##############################################################################
def on_process_button_clicked(b):
    """
    Processes the selected recordings:
    - Loads data
    - Optionally applies per-recording time crops
    - Selects specific channels based on user selection
    - Epochs data into 5-second epochs
    - Optionally concatenates recordings
    - Optionally splits concatenated data into 10-minute blocks
    - Pickles the final data
    """
    with log_output:
        clear_output()
        
        selected_paths = list(recording_select.value)
        out_dir = output_dir_text.value.strip()
        
        if not out_dir:
            print("ERROR: No output directory specified.")
            return
        if not os.path.isdir(out_dir):
            print(f"ERROR: '{out_dir}' is not a valid directory.")
            return
        if not selected_paths:
            print("No recordings selected.")
            return
        
        # **New**: Retrieve selected channels
        selected_channels = list(channel_select_widget.value)
        req = required_channel_count()
        if len(selected_channels) != req:
            print(f"ERROR: Exactly {req} channels must be selected. Currently selected: {len(selected_channels)}")
            return
        
        # Map selected channels to zero-based indices
        # Assuming channels are labeled as 'Ch1' to 'ChN'
        try:
            selected_channel_indices = [int(ch.replace('Ch', '')) - 1 for ch in selected_channels]
        except ValueError:
            print("ERROR: Channel labels must be in the format 'Ch1' to 'ChN'.")
            return
        
        # Determine if we need to apply crop for each recording
        # Gather crop settings for each recording
        apply_crop_dict = {}
        crop_times_dict = {}
        
        for path in selected_paths:
            widgets_box = crop_settings_widgets.get(path, None)
            if widgets_box:
                # Extract the widgets
                apply_crop_toggle = widgets_box.children[1].children[0]  # HBox[0]: ToggleButtons
                start_time = widgets_box.children[2].children[0]          # HBox[1]: Start time
                end_time = widgets_box.children[2].children[1]            # HBox[1]: End time
                
                apply_crop = (apply_crop_toggle.value == 'Yes')
                if apply_crop:
                    start_s = start_time.value
                    end_s = end_time.value
                    if start_s < 0:
                        start_s = 0
                    if end_s <= start_s:
                        print(f"WARNING: For '{os.path.basename(path)}', end time <= start time. Ignoring crop.")
                        apply_crop = False
                else:
                    start_s = None
                    end_s = None
                
                apply_crop_dict[path] = apply_crop
                crop_times_dict[path] = (start_s, end_s) if apply_crop else (None, None)
        
        # Toggle to decide whether to concatenate
        concatenate_selected = (concatenate_toggle.value == 'Yes')
        
        # Optionally split into 10-min blocks
        do_split_10min = (split_10min_toggle.value == 'Yes')

        # **New**: Retrieve Filename Prefix and Suffix
        prefix = output_filename_prefix_widget.value.strip()
        suffix = output_filename_suffix_widget.value.strip()

        # Function to get the stripped basename
        def get_stripped_basename(path):
            basename = os.path.basename(path)
            return strip_datetime_prefix(basename)

        # **New**: Ensure channel verification is completed
        with verification_output:
            current_selection = list(channel_select_widget.value)
            req = required_channel_count()
            if len(current_selection) != req:
                print(f"ERROR: Channel selection verification incomplete. Please ensure exactly {req} channels are selected and verified.")
                return

        # If concatenation is chosen and multiple recordings are selected, prepare to combine
        if concatenate_selected and len(selected_paths) >= 2:
            print("Concatenation selected. Processing recordings to concatenate...")
            
            epoched_list = []
            final_labels = []

            for path in selected_paths:
                # Directly use the path to 'continuous.dat' as found by find_open_ephys_recordings_in_dir
                dat_file = os.path.join(path, "continuous.dat")
                if not os.path.exists(dat_file):
                    print(f"SKIP: 'continuous.dat' not found in '{path}'")
                    continue

                # Load and reshape
                try:
                    raw_data = np.fromfile(dat_file, dtype=np.int16)
                except Exception as e:
                    print(f"  ERROR: Failed to load '{dat_file}': {e}")
                    continue

                # Attempt to pull sfreq/n_channels from the nearest settings.xml
                meta = get_recording_metadata(path, cache=recording_metadata)
                n_channels_total = meta['n_channels_total']
                sfreq = meta['sfreq']
                try:
                    reshaped_data = raw_data.reshape(-1, n_channels_total)
                except Exception as e:
                    print(f"  ERROR: Failed to reshape data from '{dat_file}': {e}")
                    continue

                # **New**: Select only the desired channels
                reshaped_data = reshaped_data[:, selected_channel_indices]  # Shape: (n_samples, 16)
                n_channels_selected = len(selected_channel_indices)  # Should be 16

                # Apply crop if needed
                apply_crop = apply_crop_dict.get(path, False)
                if apply_crop:
                    start_s, end_s = crop_times_dict.get(path, (None, None))
                    if start_s is not None and end_s is not None:
                        sidx = int(start_s * sfreq)
                        eidx = int(end_s * sfreq)
                        sidx = max(0, sidx)
                        eidx = min(eidx, reshaped_data.shape[0])
                        if sidx >= eidx:
                            print(f"  WARNING: For '{os.path.basename(path)}', start_idx >= end_idx. Skipping crop.")
                            cropped_data = reshaped_data
                        else:
                            cropped_data = reshaped_data[sidx:eidx, :]
                            actual_duration = eidx - sidx
                            actual_duration_s = actual_duration / sfreq
                            if actual_duration_s < (end_s - start_s):
                                print(f"  NOTICE: For '{os.path.basename(path)}', end time exceeds recording duration. "
                                      f"Adjusted end time to {actual_duration_s:.2f} seconds.")
                    else:
                        cropped_data = reshaped_data
                else:
                    cropped_data = reshaped_data

                # Epoch into 5-second chunks
                epoched_data = segment_data_into_5s_epochs(
                    cropped_data, sfreq=sfreq, n_channels=n_channels_selected
                )

                if epoched_data.size == 0:
                    print(f"  WARNING: No epochs created for '{os.path.basename(path)}'. Skipping.")
                    continue

                epoched_list.append(epoched_data)
                final_labels.append(get_stripped_basename(path))

                print(f"Loaded & epoched => '{get_stripped_basename(path)}'  Shape={epoched_data.shape}, Apply Crop={apply_crop}")

            if not epoched_list:
                print("No valid recordings to concatenate.")
            else:
                # Concatenate all epoched data along the epoch dimension
                try:
                    concatenated_epoched = np.concatenate(epoched_list, axis=0)
                    concatenated_label = "&".join(final_labels)
                    print(f"Concatenated recordings into '{concatenated_label}' with shape {concatenated_epoched.shape}")
                except Exception as e:
                    print(f"ERROR: Failed to concatenate recordings: {e}")
                    concatenated_epoched = None

                # Optionally split into 10-min blocks
                if concatenated_epoched is not None:
                    if do_split_10min:
                        parts_dict = split_into_10min_blocks(concatenated_epoched)
                        if not parts_dict:
                            print(f"  * No full 10min blocks found in concatenated data.")
                        else:
                            # Construct output filename with prefix and suffix
                            filename_base = f"{prefix}{concatenated_label}{suffix}" if (prefix or suffix) else concatenated_label
                            pkl_filename = f"{filename_base}_concat_with_10min_split.pkl"
                            save_path = os.path.join(out_dir, pkl_filename)
                            try:
                                with open(save_path, 'wb') as f:
                                    pickle.dump(parts_dict, f)
                                print(f"  -> Saved 10min blocks: {list(parts_dict.keys())} => '{pkl_filename}'")
                            except Exception as e:
                                print(f"  ERROR: Failed to save 10min blocks: {e}")
                    else:
                        # Save the concatenated epoched data
                        filename_base = f"{prefix}{concatenated_label}{suffix}" if (prefix or suffix) else concatenated_label
                        pkl_filename = f"{filename_base}_concat.pkl"
                        save_path = os.path.join(out_dir, pkl_filename)
                        try:
                            with open(save_path, 'wb') as f:
                                pickle.dump(concatenated_epoched, f)
                            print(f"  -> Saved concatenated epoched array => '{pkl_filename}'")
                        except Exception as e:
                            print(f"  ERROR: Failed to save concatenated data: {e}")

        # If concatenation is not chosen or only one recording is selected, process individually
        if not concatenate_selected or len(selected_paths) < 2:
            print("Processing recordings individually...")
            
            for path in selected_paths:
                # Directly use the path to 'continuous.dat' as found by find_open_ephys_recordings_in_dir
                dat_file = os.path.join(path, "continuous.dat")
                if not os.path.exists(dat_file):
                    print(f"SKIP: 'continuous.dat' not found in '{path}'")
                    continue

                # Load and reshape
                try:
                    raw_data = np.fromfile(dat_file, dtype=np.int16)
                except Exception as e:
                    print(f"  ERROR: Failed to load '{dat_file}': {e}")
                    continue

                # Attempt to pull sfreq/n_channels from the nearest settings.xml
                meta = get_recording_metadata(path, cache=recording_metadata)
                n_channels_total = meta['n_channels_total']
                sfreq = meta['sfreq']
                try:
                    reshaped_data = raw_data.reshape(-1, n_channels_total)
                except Exception as e:
                    print(f"  ERROR: Failed to reshape data from '{dat_file}': {e}")
                    continue

                # **New**: Select only the desired channels
                reshaped_data = reshaped_data[:, selected_channel_indices]  # Shape: (n_samples, 16)
                n_channels_selected = len(selected_channel_indices)  # Should be 16

                # Apply crop if needed
                apply_crop = apply_crop_dict.get(path, False)
                if apply_crop:
                    start_s, end_s = crop_times_dict.get(path, (None, None))
                    if start_s is not None and end_s is not None:
                        sidx = int(start_s * sfreq)
                        eidx = int(end_s * sfreq)
                        sidx = max(0, sidx)
                        eidx = min(eidx, reshaped_data.shape[0])
                        if sidx >= eidx:
                            print(f"  WARNING: For '{os.path.basename(path)}', start_idx >= end_idx. Skipping crop.")
                            cropped_data = reshaped_data
                        else:
                            cropped_data = reshaped_data[sidx:eidx, :]
                            actual_duration = eidx - sidx
                            actual_duration_s = actual_duration / sfreq
                            if actual_duration_s < (end_s - start_s):
                                print(f"  NOTICE: For '{os.path.basename(path)}', end time exceeds recording duration. "
                                      f"Adjusted end time to {actual_duration_s:.2f} seconds.")
                    else:
                        cropped_data = reshaped_data
                else:
                    cropped_data = reshaped_data

                # Epoch into 5-second chunks
                epoched_data = segment_data_into_5s_epochs(
                    cropped_data, sfreq=sfreq, n_channels=n_channels_selected
                )

                if epoched_data.size == 0:
                    print(f"  WARNING: No epochs created for '{os.path.basename(path)}'. Skipping.")
                    continue

                # Optionally split into 10-min blocks
                if do_split_10min:
                    parts_dict = split_into_10min_blocks(epoched_data)
                    if not parts_dict:
                        print(f"  * No full 10min blocks found for '{os.path.basename(path)}'")
                    else:
                        stripped_basename = get_stripped_basename(path)
                        # Construct output filename with prefix and suffix
                        filename_base = f"{prefix}{stripped_basename}{suffix}" if (prefix or suffix) else stripped_basename
                        pkl_filename = f"{filename_base}_concat_with_10min_split.pkl"
                        save_path = os.path.join(out_dir, pkl_filename)
                        try:
                            with open(save_path, 'wb') as f:
                                pickle.dump(parts_dict, f)
                            print(f"  -> Saved 10min blocks: {list(parts_dict.keys())} => '{pkl_filename}'")
                        except Exception as e:
                            print(f"  ERROR: Failed to save 10min blocks for '{os.path.basename(path)}': {e}")
                else:
                    # Save the epoched data
                    stripped_basename = get_stripped_basename(path)
                    # Construct output filename with prefix and suffix
                    filename_base = f"{prefix}{stripped_basename}{suffix}" if (prefix or suffix) else stripped_basename
                    pkl_filename = f"{filename_base}_epoched.pkl"
                    save_path = os.path.join(out_dir, pkl_filename)
                    try:
                        with open(save_path, 'wb') as f:
                            pickle.dump(epoched_data, f)
                        print(f"  -> Saved epoched array => '{pkl_filename}'")
                    except Exception as e:
                        print(f"  ERROR: Failed to save epoched data for '{os.path.basename(path)}': {e}")
    
        print("\nDONE processing selected recordings.")

process_button.on_click(on_process_button_clicked)

##############################################################################
# 11) BUILD & DISPLAY THE UI
##############################################################################
# Create a verification section
verification_section = VBox([
    Label("STEP K) Verify Selected Channels:"),
    verification_output,
    confirm_verification_button
])

# Create a confirmation button
confirm_verification_button = Button(
    description='Confirm Selection',
    button_style='success',
    tooltip='Confirm that the selected channels are correct'
)

def on_confirm_verification_clicked(b):
    """
    Confirms the verification of channel selection.
    """
    with verification_output:
        clear_output()
        selected_channels = list(channel_select_widget.value)
        req = required_channel_count()
        if len(selected_channels) != req:
            print(f"ERROR: Exactly {req} channels must be selected. Currently selected: {len(selected_channels)}")
            return
        print("Channel selection verified successfully.")

confirm_verification_button.on_click(on_confirm_verification_clicked)

# Arrange PSD parameters horizontally
# Note: In this context, it's not applicable. Remove or adjust accordingly.

# Combine all widgets into a vertical box with modifications
ui = VBox([
    # STEP A: Add Directory
    Label("STEP A) Use the file chooser below to select a directory, then click 'Add Selected Directory':"),
    fc,
    add_directory_button,

    # STEP B: Remove Directories
    Label("STEP B) If you want to remove previously added directories, select them below:"),
    dir_remove_select,
    remove_directory_button,
    
    # STEP C: Load Recordings from Directories
    Label("STEP C) Load subfolders from the directories:"),
    load_dirs_button,
    directory_accordion,
    
    # STEP D: Select Recordings to Process
    Label("STEP D) Choose which recordings to process:"),
    recording_select,

    # STEP E: Set Per-Recording Crop Options
    Label("STEP E) Set per-recording time crop options:"),
    crop_options_box,

    # STEP F: Concatenation Option
    Label("STEP F) Decide whether to concatenate the selected recordings:"),
    concatenate_toggle,

    # STEP G: Output Directory and Filename Prefix/Suffix
    Label("STEP G) Output folder and Filename Customization:"),
    HBox([
        output_dir_text,
        VBox([
            output_filename_prefix_widget,
            output_filename_suffix_widget
        ])
    ]),
    
    # STEP H: Optional Splitting
    Label("STEP H) Split final data into 10-minute blocks (optional)?"),
    split_10min_toggle,
    
    # STEP J: Load Mode Selection
    Label("STEP J) Select Load Mode:"),
    load_mode_widget,

    # Optional EEG (ADC tail) channels
    include_eeg_widget,
    eeg_count_widget,
    
    # STEP K: Channel Selection (New Section)
    channel_selection_label,
    channel_select_widget,
    
    # STEP L: Verification Window (New Section)
    verification_section,
    
    # STEP I: Process Button
    Label("STEP I) Process:"),
    process_button,

    # Log Output
    log_output
])

# Display the UI
display(ui)

##############################################################################
# 12) ADDITIONAL INITIALIZATION
##############################################################################
# Initialize the verification window based on the default load mode and channel selection
update_verification_window()