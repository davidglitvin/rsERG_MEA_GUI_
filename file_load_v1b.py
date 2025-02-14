##############################################################################
#  BIG_OPEN_EPHYS_GUI_WITH_CORRECT_CHANNEL_MAPPING.PY
#
#  Features:
#    1) Parse settings.xml to extract sampling rate, number of channels, and enabled channel IDs.
#    2) Automatically populate channel selection based on enabled channels.
#    3) Correctly map selected channels to data indices to prevent indexing errors.
#    4) Maintain existing GUI functionality for processing recordings.
##############################################################################

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math

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
import xml.etree.ElementTree as ET

# For the in-notebook directory chooser:
from ipyfilechooser import FileChooser

# Check for python-pptx (optional)
try:
    from pptx import Presentation
    from pptx.util import Inches
    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False
    print("Warning: python-pptx not installed, PPT export will be disabled.")

##############################################################################
# 0) PARSE SETTINGS.XML FUNCTION + WIDGETS
##############################################################################

parsed_xml_metadata = None  # Will store {"sampling_rate": ..., "n_channels": ..., "enabled_indices": [...]}

def parse_settings_xml(xml_path):
    """
    Parse an Open Ephys settings.xml file to extract:
      - The sample rate from the 'Record Node' processor's <STREAM> block
      - The number of channels from the 'Record Node'
      - Which channel indices were enabled by the 'Channel Map' processor.

    Returns a dict, e.g.:
      {
        "sampling_rate": 10000.0,
        "n_channels": 16,
        "enabled_indices": [16, 17, 18, ..., 63]
      }
    """
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"File not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    metadata = {
        "sampling_rate": None,
        "n_channels": None,
        "enabled_indices": []
    }
    
    # 1) Find the <PROCESSOR> for the Record Node
    record_node = None
    for proc in root.findall(".//PROCESSOR"):
        if proc.get("name") == "Record Node":
            record_node = proc
            break
    
    if record_node is not None:
        # Find <STREAM> under the 'Record Node'
        stream_tag = record_node.find("./STREAM")
        if stream_tag is not None:
            sr_str = stream_tag.get("sample_rate")
            ch_count_str = stream_tag.get("channel_count")
            if sr_str is not None:
                metadata["sampling_rate"] = float(sr_str)
            if ch_count_str is not None:
                metadata["n_channels"] = int(ch_count_str)
    
    # 2) Find the <PROCESSOR> named "Channel Map" to see which channels were enabled
    channel_map_node = None
    for proc in root.findall(".//PROCESSOR"):
        if proc.get("name") == "Channel Map":
            channel_map_node = proc
            break
    
    if channel_map_node is not None:
        # Inside <PROCESSOR>, go to <CUSTOM_PARAMETERS>/<STREAM>
        custom_params = channel_map_node.find("./CUSTOM_PARAMETERS/STREAM")
        if custom_params is not None:
            # Each <CH index="X" enabled="Y"/>
            for ch_tag in custom_params.findall("CH"):
                idx_str = ch_tag.get("index")
                enabled_str = ch_tag.get("enabled")
                if idx_str is not None and enabled_str == "1":
                    idx = int(idx_str)
                    metadata["enabled_indices"].append(idx)
    
    return metadata

# Widgets to parse the XML
xml_path_text = widgets.Text(
    value='',
    placeholder='Paste path to settings.xml here',
    description='XML Path:',
    layout=widgets.Layout(width='500px')
)

parse_xml_button = widgets.Button(
    description='Parse settings.xml',
    button_style='info'
)

xml_parse_output = Output()

def on_parse_xml_button_clicked(b):
    global parsed_xml_metadata
    with xml_parse_output:
        clear_output()
        xml_path = xml_path_text.value.strip()
        if not xml_path:
            print("Please enter the path to settings.xml first.")
            return
        try:
            parsed_xml_metadata = parse_settings_xml(xml_path)
            print("=== Parsed Open Ephys Metadata ===")
            print(f"Sampling rate (Hz):  {parsed_xml_metadata['sampling_rate']}")
            print(f"Number of channels:  {parsed_xml_metadata['n_channels']}")
            print(f"Enabled channel IDs: {parsed_xml_metadata['enabled_indices']}")
            print("Now you can use 'Auto Load' to auto-select these channels in the GUI.")
            
            # Update channel_select_widget options based on enabled_indices
            enabled_channels_labels = [f"Ch{ch+1}" for ch in parsed_xml_metadata['enabled_indices']]
            channel_select_widget.options = enabled_channels_labels
            channel_select_widget.value = enabled_channels_labels  # Auto-select all in Auto Load
            if load_mode_widget.value == 'Auto Load':
                channel_select_widget.disabled = True
                print("Auto Load: Channels have been automatically selected based on settings.xml.")
            else:
                channel_select_widget.disabled = False
                print("Manual Load: You can manually select channels from the enabled channels.")
                
        except Exception as e:
            print(f"Error parsing XML: {e}")

parse_xml_button.on_click(on_parse_xml_button_clicked)

xml_parse_box = VBox([
    Label("STEP 0) (Optional) Parse your Open Ephys settings.xml:"),
    HBox([xml_path_text, parse_xml_button]),
    xml_parse_output
])

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

def segment_data_into_5s_epochs(data, sfreq=10000, n_channels=16):
    """
    Segments raw data into 5-second epochs.

    Parameters:
    - data: numpy array of shape (n_samples, n_channels)
    - sfreq: Sampling frequency
    - n_channels: Number of channels

    Returns:
    - epoched_data: shape (n_epochs, n_channels, epoch_length)
    """
    epoch_duration = 5  # seconds
    epoch_length = int(sfreq * epoch_duration)
    total_samples = data.shape[0]
    n_epochs = total_samples // epoch_length

    if n_epochs == 0:
        return np.empty((0, n_channels, epoch_length))

    truncated_data = data[: n_epochs * epoch_length, :]
    epoched_data = truncated_data.reshape(n_epochs, epoch_length, n_channels)
    # Move channels to (n_epochs, n_channels, epoch_length)
    epoched_data = np.transpose(epoched_data, (0, 2, 1))
    return epoched_data

def split_into_10min_blocks(epoched_data):
    """
    Splits epoched data (5s epochs) into 10-minute blocks.

    - epoched_data: shape (N, 16, 50000) if 10kHz / 5s epochs
    Returns dict of { 'part_A': block, 'part_B': block, ... }
    Each block is 120 epochs => 10 min
    """
    n_epochs = epoched_data.shape[0]
    block_size = 120  # 10 minutes = 120 epochs
    parts_dict = {}
    
    start_idx = 0
    part_counter = 0
    while (start_idx + block_size) <= n_epochs:
        end_idx = start_idx + block_size
        block = epoched_data[start_idx:end_idx, :, :]
        label = f"part_{chr(ord('A') + part_counter)}"
        parts_dict[label] = block
        part_counter += 1
        start_idx = end_idx

    return parts_dict

##############################################################################
# 2) HELPER FUNCTIONS: FIND 'continuous.dat'
##############################################################################
def find_open_ephys_recordings_in_dir(base_dir):
    """
    Finds subfolders within base_dir that contain 'continuous.dat'.
    """
    matches = []
    for root, dirs, files in os.walk(base_dir):
        if "continuous.dat" in files:
            matches.append(root)
    return matches

##############################################################################
# 3) PLOTTING FUNCTIONS (unchanged from your code)
##############################################################################
def exclude_traces(
    psd_array,
    freqs,
    low_band=(1, 3),
    low_band_threshold=3.0,
    test_bands=[(7,9), (9,11), (11,13), (13,15), (15,17), (17,19), (19,21)],
    test_band_threshold=10.0,
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
        # 1) Check low-frequency outlier
        if np.any(trace[low_band_indices] > low_band_threshold * mean_psd[low_band_indices]):
            excluded_traces.append(i)
            continue

        # 2) Check repeated suprathreshold events
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
    show_kept=True,
    show_excluded=True,
    show_original_mean=True,
    show_new_mean=True,
    show_vertical_lines=True,
    color_kept="lightgray",
    color_excluded="red",
    color_old_mean="blue",
    color_new_mean="green",
    alpha_excluded=0.05,
    alpha_kept=0.7,
    title_fontsize=8,
    axis_label_fontsize=8,
    legend_fontsize=8,
    tick_label_fontsize=8,
    max_title_length=40,
    vertical_lines=None,
    vertical_line_color="black",
    vertical_line_style="--",
    vertical_line_alpha=0.6
):
    if len(title) > max_title_length:
        title = title[:max_title_length] + "..."

    if show_kept:
        for idx_i, idx in enumerate(kept_traces):
            label = "Kept Trace" if idx_i == 0 else None
            ax.plot(freqs, psd_array[idx], color=color_kept, alpha=alpha_kept, label=label)

    if show_original_mean:
        ax.plot(freqs, original_mean_psd, color=color_old_mean, linewidth=2, label="Original Mean")

    if show_new_mean and len(kept_traces) > 0:
        new_mean_psd = np.mean(psd_array[kept_traces], axis=0)
        ax.plot(freqs, new_mean_psd, color=color_new_mean, linewidth=2, label="New Mean")

    if show_excluded:
        for idx_j, idx in enumerate(excluded_traces):
            label = "Excluded Trace" if idx_j == 0 else None
            ax.plot(freqs, psd_array[idx], color=color_excluded, alpha=alpha_excluded, 
                    zorder=10, label=label)

    if show_vertical_lines and vertical_lines is not None:
        for vfreq in vertical_lines:
            ax.axvline(
                vfreq, 
                color=vertical_line_color, 
                linestyle=vertical_line_style, 
                alpha=vertical_line_alpha
            )

    ax.set_xlabel("Frequency (Hz)", fontsize=axis_label_fontsize)
    ax.set_ylabel("PSD (V²/Hz)", fontsize=axis_label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.legend(loc="upper right", fontsize=legend_fontsize)
    ax.tick_params(axis='both', labelsize=tick_label_fontsize)

def plot_psds_with_dropped_traces(
    psds_dict,
    rows_of_psds,
    low_band=(1,3),
    low_band_threshold=3.0,
    test_bands=[(7,9),(9,11),(11,13),(13,15),(15,17),(17,19),(19,21)],
    test_band_threshold=10.0,
    test_band_count_threshold=None,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    show_kept=True,
    show_excluded=True,
    show_original_mean=True,
    show_new_mean=True,
    show_vertical_lines=True,
    color_kept="lightgray",
    color_excluded="red",
    color_old_mean="blue",
    color_new_mean="green",
    num_cols=4,
    title_fontsize=8,
    axis_label_fontsize=8,
    legend_fontsize=8,
    tick_label_fontsize=8,
    max_title_length=40,
    vertical_lines=None,
    vertical_line_color="black",
    vertical_line_style="--",
    vertical_line_alpha=0.6
):
    figures = []
    for row_idx, row in enumerate(rows_of_psds, start=1):
        valid_keys = [k for k in row if k in psds_dict]
        num_plots = len(valid_keys)
        if num_plots == 0:
            continue

        num_rows = math.ceil(num_plots / num_cols)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
        plt.subplots_adjust(hspace=0.5)

        if num_rows * num_cols == 1:
            axes = [axes]
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

        for ax in axes[num_plots:]:
            ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(f"Row {row_idx}", fontsize=title_fontsize + 2)
        figures.append(fig)
        plt.show()

    return figures

def group_keys_by_rows(psd_keys, row_size=4):
    rows = []
    for i in range(0, len(psd_keys), row_size):
        rows.append(psd_keys[i:i+row_size])
    return rows

##############################################################################
# 4) GLOBAL VARIABLES & WIDGETS
##############################################################################
directories_list = []

fc = FileChooser(os.getcwd())
fc.title = 'Select a Directory'
fc.show_only_dirs = True

add_directory_button = Button(
    description='Add Selected Directory',
    button_style='info'
)

dir_remove_select = SelectMultiple(
    options=[],
    description='Added Directories:',
    layout=widgets.Layout(width='600px', height='120px')
)

remove_directory_button = Button(
    description='Remove Selected Directories',
    button_style='danger'
)

directory_accordion = VBox()

recording_select = SelectMultiple(
    options=[],
    description='All Recordings:',
    layout=widgets.Layout(width='600px', height='250px'),
    style={'description_width': 'initial'}
)

load_dirs_button = Button(
    description='Load from Added Directories',
    button_style='info'
)

output_dir_text = widgets.Text(
    value='',
    placeholder='Output folder path',
    description='Output Directory:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='600px')
)

output_filename_prefix_widget = widgets.Text(
    value='',
    placeholder='Optional prefix',
    description='Filename Prefix:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='300px')
)

output_filename_suffix_widget = widgets.Text(
    value='',
    placeholder='Optional suffix',
    description='Filename Suffix:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='300px')
)

crop_options_box = VBox()

split_10min_toggle = ToggleButtons(
    options=['No', 'Yes'],
    value='No',
    description='Split 10min?',
    layout=widgets.Layout(width='150px')
)

concatenate_toggle = ToggleButtons(
    options=['No', 'Yes'],
    value='No',
    description='Concatenate Recordings?',
    layout=widgets.Layout(width='200px')
)

process_button = Button(
    description='Process Selected Recordings',
    button_style='success'
)

log_output = Output()

# **Updated**: Initialize channel_select_widget with no options; will be set after parsing XML
channel_select_widget = SelectMultiple(
    options=[],  # Will be populated after parsing XML
    value=[],    # Will be set based on load mode
    description='Select Channels:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='600px', height='200px')
)

channel_selection_label = Label("STEP J) Select exactly 16 channels to include in processing:")

load_mode_widget = ToggleButtons(
    options=['Auto Load', 'Manual Load'],
    value='Auto Load',
    description='Load Mode:',
    layout=widgets.Layout(width='200px')
)

verification_output = Output()

##############################################################################
# 5) DIRECTORY MANIPULATION CALLBACKS
##############################################################################
def on_add_directory_clicked(b):
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
    dir_remove_select.options = directories_list

add_directory_button.on_click(on_add_directory_clicked)

def on_remove_directory_clicked(b):
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
    dir_remove_select.options = directories_list

remove_directory_button.on_click(on_remove_directory_clicked)

def on_load_dirs_clicked(b):
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
                        # Calculate number of epochs based on actual channel count from parsed metadata
                        if parsed_xml_metadata is not None and parsed_xml_metadata["n_channels"] == 16:
                            n_channels_guess = 16
                        else:
                            n_channels_guess = 64  # Fallback
                        sfreq_guess = 10000
                        epoch_duration = 5
                        epoch_length = sfreq_guess * epoch_duration

                        file_size_bytes = os.path.getsize(dat_file)
                        bytes_per_sample = 2  # int16
                        total_samples = file_size_bytes / bytes_per_sample
                        number_of_channels = n_channels_guess
                        number_of_frames = total_samples / number_of_channels
                        number_of_epochs = int(number_of_frames // epoch_length)

                        print(f"{strip_datetime_prefix(basename)}: ~{number_of_epochs} epochs (estimated)")
                else:
                    print("No Open Ephys recordings found here.")
            
            acc = Accordion(children=[out])
            acc.set_title(0, d)
            children_for_accordion.append(acc)

        directory_accordion.children = children_for_accordion
        
        all_recordings = list(set(all_recordings))
        all_recordings.sort()

        recording_select.options = all_recordings
        print(f"Loaded {len(all_recordings)} recordings from {len(directories_list)} directories.")

load_dirs_button.on_click(on_load_dirs_clicked)

##############################################################################
# 7) PER-RECORDING CROP OPTIONS
##############################################################################
crop_settings_widgets = {}

def create_crop_widgets_for_recording(recording_path):
    basename = os.path.basename(recording_path)
    
    apply_crop_toggle = ToggleButtons(
        options=['No', 'Yes'],
        value='No',
        description='Apply Crop?',
        layout=widgets.Layout(width='150px')
    )
    
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
    
    def on_apply_crop_toggle_change(change):
        if change['name'] == 'value':
            if change['new'] == 'Yes':
                start_time.disabled = False
                end_time.disabled = False
            else:
                start_time.disabled = True
                end_time.disabled = True
    
    apply_crop_toggle.observe(on_apply_crop_toggle_change, names='value')
    
    widget_box = VBox([
        Label(f"Recording: {strip_datetime_prefix(basename)}"),
        HBox([apply_crop_toggle]),
        HBox([start_time, end_time])
    ])
    
    return widget_box

def update_crop_options_box(*args):
    selected_recordings = list(recording_select.value)
    crop_options_box.children = []
    
    to_remove = [path for path in crop_settings_widgets if path not in selected_recordings]
    for path in to_remove:
        del crop_settings_widgets[path]
    
    for path in selected_recordings:
        if path not in crop_settings_widgets:
            crop_settings_widgets[path] = create_crop_widgets_for_recording(path)
    
    crop_widgets = [crop_settings_widgets[path] for path in selected_recordings]
    crop_options_box.children = crop_widgets

recording_select.observe(update_crop_options_box, names='value')

##############################################################################
# 8) CHANNEL SELECTION: on_load_mode_change
##############################################################################
def on_load_mode_change(change):
    if change['name'] == 'value':
        load_mode = change['new']
        if load_mode == 'Auto Load':
            # If we have parsed XML, auto-load the "enabled_indices" as channels
            if parsed_xml_metadata is None:
                print("No parsed XML metadata. Please parse settings.xml first or switch to Manual Load.")
                # Optionally, revert to Manual Load
                load_mode_widget.value = 'Manual Load'
                return
            
            enabled_indices = parsed_xml_metadata['enabled_indices']  # e.g. [16,17,18,...,63]
            # Convert each index i to 1-based label "Ch{i+1}"
            auto_selected_channels = [f"Ch{i+1}" for i in enabled_indices]
            
            channel_select_widget.value = auto_selected_channels
            channel_select_widget.disabled = True
            print("Auto Load: Channels have been automatically selected based on settings.xml.")
        else:
            # 'Manual Load'
            channel_select_widget.disabled = False
            print("Manual Load: You can manually select channels from the enabled channels.")

load_mode_widget.observe(on_load_mode_change, names='value')

##############################################################################
# 9) VERIFICATION WINDOW
##############################################################################
def update_verification_window():
    with verification_output:
        clear_output()
        selected_channels = list(channel_select_widget.value)
        if len(selected_channels) != 16:
            print(f"ERROR: Exactly 16 channels must be selected. Currently: {len(selected_channels)}")
            return
        
        display_label = Label("Verify your channel selection. Remove any incorrect channels:")
        display(display_label)
        
        channel_buttons = []
        for ch in selected_channels:
            btn = Button(
                description='Remove',
                button_style='danger',
                tooltip=f'Remove {ch}',
                layout=widgets.Layout(width='80px')
            )
            btn.on_click(lambda b, ch=ch: on_remove_channel_clicked(ch))
            channel_buttons.append(HBox([Label(ch), btn]))
        
        display(VBox(channel_buttons))
        print("\nAfter removing channels, re-select to maintain exactly 16 channels.")

def on_remove_channel_clicked(channel):
    current_selection = list(channel_select_widget.value)
    if channel in current_selection:
        current_selection.remove(channel)
        channel_select_widget.value = current_selection
        print(f"Removed {channel} from selection.")
        update_verification_window()
    else:
        print(f"Channel {channel} not in selection.")

def on_confirm_verification_clicked(b):
    with verification_output:
        clear_output()
        selected_channels = list(channel_select_widget.value)
        if len(selected_channels) != 16:
            print(f"ERROR: Exactly 16 channels must be selected. Currently: {len(selected_channels)}")
            return
        print("Channel selection verified successfully.")

confirm_verification_button = Button(
    description='Confirm Selection',
    button_style='success',
    tooltip='Confirm that the selected channels are correct'
)
confirm_verification_button.on_click(on_confirm_verification_clicked)

def on_channel_selection_change(change):
    if change['name'] == 'value':
        update_verification_window()

channel_select_widget.observe(on_channel_selection_change, names='value')

verification_section = VBox([
    Label("STEP K) Verify Selected Channels:"),
    verification_output,
    confirm_verification_button
])

##############################################################################
# 10) PROCESSING LOGIC (updated to use parsed_xml_metadata)
##############################################################################
def on_process_button_clicked(b):
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
        
        selected_channels = list(channel_select_widget.value)
        if len(selected_channels) != 16:
            print(f"ERROR: Exactly 16 channels must be selected. Currently {len(selected_channels)}")
            return
        
        # **New**: Map selected channels to data indices
        # Since channel_select_widget options are the enabled channels, indices are 0-15
        channel_label_to_data_index = { ch: idx for idx, ch in enumerate(selected_channels) }
        selected_channel_indices = list(range(len(selected_channels)))  # [0,1,...15]
        
        # Use parsed XML if available:
        if parsed_xml_metadata is not None:
            sfreq = parsed_xml_metadata["sampling_rate"]        # e.g. 10000.0
            n_channels_total = parsed_xml_metadata["n_channels"] # e.g. 16
        else:
            sfreq = 10000  # fallback
            n_channels_total = 64  # fallback
    
        apply_crop_dict = {}
        crop_times_dict = {}
        
        for path in selected_paths:
            wbox = crop_settings_widgets.get(path, None)
            if wbox:
                apply_crop_toggle = wbox.children[1].children[0]
                start_time_w = wbox.children[2].children[0]
                end_time_w = wbox.children[2].children[1]
                
                apply_crop = (apply_crop_toggle.value == 'Yes')
                if apply_crop:
                    start_s = start_time_w.value
                    end_s = end_time_w.value
                    if start_s < 0:
                        start_s = 0
                    if end_s <= start_s:
                        print(f"WARNING: For '{os.path.basename(path)}', end <= start. Ignoring crop.")
                        apply_crop = False
                else:
                    start_s, end_s = None, None
                apply_crop_dict[path] = apply_crop
                crop_times_dict[path] = (start_s, end_s) if apply_crop else (None, None)
        
        concatenate_selected = (concatenate_toggle.value == 'Yes')
        do_split_10min = (split_10min_toggle.value == 'Yes')
        
        prefix = output_filename_prefix_widget.value.strip()
        suffix = output_filename_suffix_widget.value.strip()

        def get_stripped_basename(path):
            return strip_datetime_prefix(os.path.basename(path))

        # Verify final channel selection again
        with verification_output:
            current_sel = list(channel_select_widget.value)
            if len(current_sel) != 16:
                print("ERROR: Channel selection not verified. Must have 16 channels.")
                return

        # Concatenation path
        if concatenate_selected and len(selected_paths) >= 2:
            print("Concatenation selected. Processing recordings to concatenate...")
            epoched_list = []
            final_labels = []

            for path in selected_paths:
                dat_file = os.path.join(path, "continuous.dat")
                if not os.path.exists(dat_file):
                    print(f"SKIP: 'continuous.dat' not found in '{path}'")
                    continue

                try:
                    raw_data = np.fromfile(dat_file, dtype=np.int16)
                except Exception as e:
                    print(f"  ERROR: Failed to load '{dat_file}': {e}")
                    continue

                try:
                    reshaped_data = raw_data.reshape(-1, n_channels_total)
                except Exception as e:
                    print(f"  ERROR: Reshape failed for '{dat_file}': {e}")
                    continue

                # Sub-select user-chosen channels (if applicable)
                if n_channels_total == 64:
                    reshaped_data = reshaped_data[:, selected_channel_indices]
                elif n_channels_total == 16:
                    reshaped_data = reshaped_data[:, selected_channel_indices]
                else:
                    print(f"  WARNING: Unexpected number of channels ({n_channels_total}) in '{dat_file}'. Skipping channel selection.")
                
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
                    else:
                        cropped_data = reshaped_data
                else:
                    cropped_data = reshaped_data

                epoched_data = segment_data_into_5s_epochs(cropped_data, sfreq=sfreq, n_channels=len(selected_channel_indices))
                if epoched_data.size == 0:
                    print(f"  WARNING: No epochs for '{os.path.basename(path)}'. Skipping.")
                    continue

                epoched_list.append(epoched_data)
                final_labels.append(get_stripped_basename(path))
                print(f"Loaded & epoched => '{get_stripped_basename(path)}' shape={epoched_data.shape}, Crop={apply_crop}")
    
            if not epoched_list:
                print("No valid recordings to concatenate.")
            else:
                try:
                    concatenated_epoched = np.concatenate(epoched_list, axis=0)
                    concatenated_label = "&".join(final_labels)
                    print(f"Concatenated shape {concatenated_epoched.shape}")
                except Exception as e:
                    print(f"ERROR: Failed to concatenate: {e}")
                    concatenated_epoched = None

                if concatenated_epoched is not None:
                    if do_split_10min:
                        parts_dict = split_into_10min_blocks(concatenated_epoched)
                        if not parts_dict:
                            print("* No 10min blocks found.")
                        else:
                            filename_base = f"{prefix}{concatenated_label}{suffix}" if (prefix or suffix) else concatenated_label
                            pkl_filename = f"{filename_base}_concat_with_10min_split.pkl"
                            save_path = os.path.join(out_dir, pkl_filename)
                            try:
                                with open(save_path, 'wb') as f:
                                    pickle.dump(parts_dict, f)
                                print(f"Saved 10min blocks => '{pkl_filename}'")
                            except Exception as e:
                                print(f"ERROR saving 10min blocks: {e}")
                    else:
                        filename_base = f"{prefix}{concatenated_label}{suffix}" if (prefix or suffix) else concatenated_label
                        pkl_filename = f"{filename_base}_concat.pkl"
                        save_path = os.path.join(out_dir, pkl_filename)
                        try:
                            with open(save_path, 'wb') as f:
                                pickle.dump(concatenated_epoched, f)
                            print(f"Saved concatenated epoched => '{pkl_filename}'")
                        except Exception as e:
                            print(f"ERROR saving concatenated: {e}")

        # If not concatenating or only one recording
        if not concatenate_selected or len(selected_paths) < 2:
            print("Processing recordings individually...")
            for path in selected_paths:
                dat_file = os.path.join(path, "continuous.dat")
                if not os.path.exists(dat_file):
                    print(f"SKIP: 'continuous.dat' not found in '{path}'")
                    continue

                try:
                    raw_data = np.fromfile(dat_file, dtype=np.int16)
                except Exception as e:
                    print(f"  ERROR: Failed to load '{dat_file}': {e}")
                    continue

                try:
                    reshaped_data = raw_data.reshape(-1, n_channels_total)
                except Exception as e:
                    print(f"  ERROR: Reshape failed for '{dat_file}': {e}")
                    continue

                # Sub-select user-chosen channels
                if n_channels_total in [64, 16]:
                    reshaped_data = reshaped_data[:, selected_channel_indices]
                else:
                    print(f"  WARNING: Unexpected number of channels ({n_channels_total}) in '{dat_file}'. Skipping channel selection.")
                
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
                    else:
                        cropped_data = reshaped_data
                else:
                    cropped_data = reshaped_data

                epoched_data = segment_data_into_5s_epochs(cropped_data, sfreq=sfreq, n_channels=len(selected_channel_indices))
                if epoched_data.size == 0:
                    print(f"  WARNING: No epochs for '{os.path.basename(path)}'. Skipping.")
                    continue

                if do_split_10min:
                    parts_dict = split_into_10min_blocks(epoched_data)
                    if not parts_dict:
                        print(f"  * No 10min blocks for '{os.path.basename(path)}'")
                    else:
                        stripped_basename = get_stripped_basename(path)
                        filename_base = f"{prefix}{stripped_basename}{suffix}" if (prefix or suffix) else stripped_basename
                        pkl_filename = f"{filename_base}_concat_with_10min_split.pkl"
                        save_path = os.path.join(out_dir, pkl_filename)
                        try:
                            with open(save_path, 'wb') as f:
                                pickle.dump(parts_dict, f)
                            print(f"  -> Saved 10min blocks => '{pkl_filename}'")
                        except Exception as e:
                            print(f"  ERROR: {e}")
                else:
                    stripped_basename = get_stripped_basename(path)
                    filename_base = f"{prefix}{stripped_basename}{suffix}" if (prefix or suffix) else stripped_basename
                    pkl_filename = f"{filename_base}_epoched.pkl"
                    save_path = os.path.join(out_dir, pkl_filename)
                    try:
                        with open(save_path, 'wb') as f:
                            pickle.dump(epoched_data, f)
                        print(f"  -> Saved epoched => '{pkl_filename}'")
                    except Exception as e:
                        print(f"  ERROR: {e}")

        print("\nDONE processing selected recordings.")

process_button.on_click(on_process_button_clicked)

##############################################################################
# 11) BUILD & DISPLAY THE UI
##############################################################################
ui = VBox([
    # STEP 0) XML parse
    xml_parse_box,
    
    # STEP A) Add directory
    Label("STEP A) Use the file chooser below to select a directory, then click 'Add Selected Directory':"),
    fc,
    add_directory_button,

    # STEP B) Remove directories
    Label("STEP B) If needed, remove directories here:"),
    dir_remove_select,
    remove_directory_button,

    # STEP C) Load subfolders
    Label("STEP C) Load subfolders from the directories:"),
    load_dirs_button,
    directory_accordion,

    # STEP D) Select recordings
    Label("STEP D) Choose which recordings to process:"),
    recording_select,

    # STEP E) Crop options
    Label("STEP E) Per-recording time crop options:"),
    crop_options_box,

    # STEP F) Concatenation
    Label("STEP F) Concatenate selected recordings?"),
    concatenate_toggle,

    # STEP G) Output directory & filename prefix/suffix
    Label("STEP G) Specify output folder + optional filename prefix/suffix:"),
    HBox([
        output_dir_text,
        VBox([
            output_filename_prefix_widget,
            output_filename_suffix_widget
        ])
    ]),
    
    # STEP H) Split final data into 10-min blocks
    Label("STEP H) Split final data into 10-minute blocks (optional)?"),
    split_10min_toggle,

    # STEP I) Load Mode
    Label("STEP I) Channel Load Mode:"),
    load_mode_widget,

    # STEP J) Channel Selection
    channel_selection_label,
    channel_select_widget,

    # STEP K) Verification
    verification_section,

    # STEP L) Process button
    Label("STEP L) Process recordings:"),
    process_button,

    # Log Output
    log_output
])

display(ui)

# Initialize verification window if channels are already selected
update_verification_window()
