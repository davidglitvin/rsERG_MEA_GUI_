##############################################################################
# 0) IMPORTS
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
    epoch_length = sfreq * epoch_duration  # e.g., 10,000 * 5 = 50,000
    total_samples = data.shape[0]
    n_epochs = total_samples // epoch_length

    if n_epochs == 0:
        return np.empty((0, n_channels, epoch_length))

    truncated_data = data[: n_epochs * epoch_length, :]
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
# 2) PLOTTING FUNCTIONS (PSD-RELATED)
#    [These remain in place if you need them later for analyzing PSDs]
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
# 3) GLOBAL VARIABLES & WIDGETS
##############################################################################
# Instead of scanning directories, we will select multiple .dat files directly:
# A) FileChooser for multiple `.dat` files
file_chooser = FileChooser(os.getcwd())
file_chooser.title = 'Select one or more .dat files'
file_chooser.show_only_dirs = False
file_chooser.filter_pattern = '*.dat'
file_chooser.use_dir_icons = True
file_chooser.multiple = True  # Allow multiple file selection

# B) Button to confirm selection of files
add_files_button = Button(
    description='Load Selected .dat Files',
    button_style='info'
)

# We will keep a "recording_select" style widget to display chosen files (just for continuity)
recording_select = SelectMultiple(
    options=[],
    description='Selected Files:',
    layout=widgets.Layout(width='600px', height='200px'),
    style={'description_width': 'initial'}
)

# Container for per-recording crop options
crop_options_box = VBox()

# Toggle to optionally split concatenated data into 10-minute blocks
split_10min_toggle = ToggleButtons(
    options=['No', 'Yes'],
    value='No',
    description='Split 10min?',
    button_style='',
    layout=widgets.Layout(width='150px')
)

# Toggle to decide whether to concatenate selected recordings
concatenate_toggle = ToggleButtons(
    options=['No', 'Yes'],
    value='No',
    description='Concatenate Files?',
    button_style='',
    layout=widgets.Layout(width='200px')
)

# Text widget to specify output directory
output_dir_text = widgets.Text(
    value='',
    placeholder='Type or paste output folder path here',
    description='Output Directory:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='600px')
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

# The channel selection widget
channel_select_widget = SelectMultiple(
    options=[f"Ch{ch}" for ch in range(1, 65)],
    value=[f"Ch{ch}" for ch in range(1, 17)],  # Default selection: first 16 channels
    description='Select Channels:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='600px', height='200px')
)

# A label to prompt the user
channel_selection_label = Label(
    value="STEP E) Select exactly 16 channels to include in processing:"
)

# ToggleButtons to select Load Mode (Auto Load vs Manual Load)
load_mode_widget = ToggleButtons(
    options=['Auto Load', 'Manual Load'],
    value='Auto Load',
    description='Load Mode:',
    button_style='',
    layout=widgets.Layout(width='200px')
)

# Verification Window (VBox) to confirm selected channels
verification_output = Output()

# Finally, a process button
process_button = Button(
    description='Process Selected Files',
    button_style='success'
)

# A log output area
log_output = Output()

##############################################################################
# 4) CROPPING WIDGETS CREATION
##############################################################################
crop_settings_widgets = {}  # Key: file path, Value: widget VBox

def create_crop_widgets_for_recording(recording_path):
    """
    Creates crop option widgets for a single .dat file.

    Returns a VBox containing:
      - ToggleButtons: 'Apply Crop? Yes/No'
      - IntText: start_time, end_time
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
        Label(f"File: {strip_datetime_prefix(basename)}"),
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

recording_select.observe(update_crop_options_box, names='value')

##############################################################################
# 5) CHANNEL SELECTION: LOAD MODE + VERIFICATION
##############################################################################
def on_load_mode_change(change):
    """
    Adjusts the channel selection widget based on the selected load mode.
    """
    if change['name'] == 'value':
        load_mode = change['new']
        if load_mode == 'Auto Load':
            # Example: pick Ch17-Ch24 and Ch41-Ch48
            auto_selected_channels = [f"Ch{ch}" for ch in range(17, 25)] + [f"Ch{ch}" for ch in range(41, 49)]
            channel_select_widget.value = auto_selected_channels
            channel_select_widget.disabled = True
            with verification_output:
                clear_output()
                print("Auto Load: Predefined 16 channels have been selected, manual selection disabled.")
        elif load_mode == 'Manual Load':
            channel_select_widget.disabled = False
            # Optionally reset to a default
            if len(channel_select_widget.value) != 16:
                channel_select_widget.value = [f"Ch{ch}" for ch in range(1, 17)]
            with verification_output:
                clear_output()
                print("Manual Load: Please select 16 channels from the list.")

load_mode_widget.observe(on_load_mode_change, names='value')

def update_verification_window():
    """
    Updates the verification window to display currently selected channels.
    """
    with verification_output:
        clear_output()
        selected_channels = list(channel_select_widget.value)
        if len(selected_channels) != 16:
            print(f"ERROR: Exactly 16 channels must be selected. Currently selected: {len(selected_channels)}")
            return
        
        # Display the list of selected channels with "Remove" buttons
        display_label = Label("Verify your channel selection. Remove any channel if needed:")
        display(display_label)
        
        channel_buttons = []
        for ch in selected_channels:
            btn = Button(
                description='Remove',
                button_style='danger',
                tooltip=f'Remove {ch}',
                layout=widgets.Layout(width='70px')
            )
            # Attach callback with closure to capture channel name
            btn.on_click(lambda b, ch=ch: on_remove_channel_clicked(ch))
            channel_buttons.append(HBox([Label(ch), btn]))
        
        display(VBox(channel_buttons))
        print("\nAfter removing a channel, please re-select channels so you end up with exactly 16.")

def on_remove_channel_clicked(channel):
    """
    Removes a channel from the selection.
    """
    current_selection = list(channel_select_widget.value)
    if channel in current_selection:
        current_selection.remove(channel)
        channel_select_widget.value = current_selection
        with verification_output:
            print(f"Removed {channel} from selection.")
        update_verification_window()

def on_channel_selection_change(change):
    """
    Updates the verification window whenever the channel selection changes.
    """
    if change['name'] == 'value':
        update_verification_window()

channel_select_widget.observe(on_channel_selection_change, names='value')

# A Confirm button for channel verification
confirm_verification_button = Button(
    description='Confirm Selection',
    button_style='success',
    tooltip='Confirm that the selected channels are correct'
)

def on_confirm_verification_clicked(b):
    with verification_output:
        clear_output()
        selected_channels = list(channel_select_widget.value)
        if len(selected_channels) != 16:
            print(f"ERROR: Exactly 16 channels must be selected. Currently selected: {len(selected_channels)}")
            return
        print("Channel selection verified successfully.")

confirm_verification_button.on_click(on_confirm_verification_clicked)

##############################################################################
# 6) MULTI-FILE SELECTION LOGIC
##############################################################################
selected_files = []

def on_add_files_button_clicked(b):
    """
    When you click 'Load Selected .dat Files', it populates the recording_select
    widget with whichever .dat files the user picked in the FileChooser.
    """
    with log_output:
        clear_output()
        chosen = file_chooser.selected
        if not chosen:
            print("No files selected. Please select at least one `.dat` file.")
            return
        
        # Overwrite our global list
        selected_files.clear()
        selected_files.extend(chosen)
        
        # Update the "recording_select" widget
        # (We'll store the absolute paths in 'options'.)
        recording_select.options = selected_files
        
        print(f"Loaded {len(chosen)} file(s):")
        for fpath in chosen:
            print(" -", fpath)

add_files_button.on_click(on_add_files_button_clicked)

##############################################################################
# 7) PROCESSING LOGIC
##############################################################################
def on_process_button_clicked(b):
    """
    Processes the selected recordings:
    - Loads data
    - Optionally applies per-recording time crops
    - Selects 16 user-chosen channels
    - Epochs data into 5-second epochs
    - Optionally concatenates recordings
    - Optionally splits data into 10-min blocks
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
            print("No files selected.")
            return
        
        # Check channels
        selected_channels = list(channel_select_widget.value)
        if len(selected_channels) != 16:
            print(f"ERROR: Exactly 16 channels must be selected. Currently selected: {len(selected_channels)}")
            return
        
        # Convert 'Ch5' => index 4, etc.
        try:
            selected_channel_indices = [int(ch.replace('Ch', '')) - 1 for ch in selected_channels]
        except ValueError:
            print("ERROR: Channel labels must be in the format 'Ch1' to 'Ch64'.")
            return
        
        # Retrieve user toggles
        concatenate_selected = (concatenate_toggle.value == 'Yes')
        do_split_10min = (split_10min_toggle.value == 'Yes')
        prefix = output_filename_prefix_widget.value.strip()
        suffix = output_filename_suffix_widget.value.strip()
        
        # Check channel verification done
        with verification_output:
            current_ch_selection = list(channel_select_widget.value)
            if len(current_ch_selection) != 16:
                print("ERROR: Channel verification incomplete. Must have exactly 16 channels selected.")
                return
        
        # If concatenation is chosen and multiple files are selected,
        # we will merge them into one epoched array:
        if concatenate_selected and len(selected_paths) >= 2:
            print("Concatenation selected. All chosen files will be combined.")
            epoched_list = []
            final_labels = []
            
            for path in selected_paths:
                if not os.path.exists(path):
                    print(f"SKIP: File not found => {path}")
                    continue
                try:
                    raw_data = np.fromfile(path, dtype=np.int16)
                except Exception as e:
                    print(f"ERROR loading {path}: {e}")
                    continue
                
                # Reshape to [samples, 64], then pick channels
                n_channels_total = 64
                sfreq = 10000
                try:
                    reshaped_data = raw_data.reshape(-1, n_channels_total)
                except Exception as e:
                    print(f"ERROR reshaping data from {path}: {e}")
                    continue
                
                # select channels
                reshaped_data = reshaped_data[:, selected_channel_indices]
                
                # Crop if user-specified
                apply_crop = False
                start_s = None
                end_s   = None
                widget_box = crop_settings_widgets.get(path, None)
                if widget_box is not None:
                    apply_crop_toggle = widget_box.children[1].children[0]
                    apply_crop = (apply_crop_toggle.value == 'Yes')
                    start_time_widget = widget_box.children[2].children[0]
                    end_time_widget   = widget_box.children[2].children[1]
                    if apply_crop:
                        start_s = start_time_widget.value
                        end_s   = end_time_widget.value
                
                if apply_crop and (start_s is not None) and (end_s is not None):
                    sidx = int(start_s * sfreq)
                    eidx = int(end_s * sfreq)
                    sidx = max(0, sidx)
                    eidx = min(eidx, reshaped_data.shape[0])
                    if sidx < eidx:
                        reshaped_data = reshaped_data[sidx:eidx, :]
                        actual_duration_s = (eidx - sidx)/sfreq
                        if actual_duration_s < (end_s - start_s):
                            print(f"  Note: Adjusted end time because the recording was shorter.")
                
                # Epoch
                epoched_data = segment_data_into_5s_epochs(reshaped_data, sfreq=sfreq, n_channels=16)
                if epoched_data.size == 0:
                    print(f"WARNING: 0 epochs for file {os.path.basename(path)}. Skipping.")
                    continue
                
                epoched_list.append(epoched_data)
                final_labels.append(strip_datetime_prefix(os.path.basename(path)))
                print(f"Loaded => {os.path.basename(path)}  shape={epoched_data.shape}")
            
            if not epoched_list:
                print("No valid epoched data to concatenate.")
                return
            
            # Concatenate
            try:
                concatenated_epoched = np.concatenate(epoched_list, axis=0)
            except Exception as e:
                print(f"ERROR concatenating: {e}")
                return
            
            combined_label = "&".join(final_labels)
            print(f"Concatenated shape: {concatenated_epoched.shape}")
            
            # Optionally split
            if do_split_10min:
                parts_dict = split_into_10min_blocks(concatenated_epoched)
                if not parts_dict:
                    print("No full 10-min blocks found in concatenated data.")
                else:
                    filename_base = f"{prefix}{combined_label}{suffix}" if (prefix or suffix) else combined_label
                    pkl_filename = f"{filename_base}_concat_10min.pkl"
                    save_path = os.path.join(out_dir, pkl_filename)
                    try:
                        with open(save_path, 'wb') as f:
                            pickle.dump(parts_dict, f)
                        print(f"Saved 10-min splitted concatenated => {pkl_filename}")
                    except Exception as e:
                        print(f"ERROR saving splitted data: {e}")
            else:
                # Save entire concatenated array
                filename_base = f"{prefix}{combined_label}{suffix}" if (prefix or suffix) else combined_label
                pkl_filename = f"{filename_base}_concat.pkl"
                save_path = os.path.join(out_dir, pkl_filename)
                try:
                    with open(save_path, 'wb') as f:
                        pickle.dump(concatenated_epoched, f)
                    print(f"Saved concatenated => {pkl_filename}")
                except Exception as e:
                    print(f"ERROR saving concatenated data: {e}")
        
        # If not concatenating or only one file:
        if not concatenate_selected or len(selected_paths) < 2:
            print("Processing each file individually...")
            for path in selected_paths:
                if not os.path.exists(path):
                    print(f"SKIP: File not found => {path}")
                    continue
                
                try:
                    raw_data = np.fromfile(path, dtype=np.int16)
                except Exception as e:
                    print(f"ERROR loading {path}: {e}")
                    continue
                
                n_channels_total = 64
                sfreq = 10000
                try:
                    reshaped_data = raw_data.reshape(-1, n_channels_total)
                except Exception as e:
                    print(f"ERROR reshaping data from {path}: {e}")
                    continue
                
                # select channels
                reshaped_data = reshaped_data[:, selected_channel_indices]
                
                # crop if needed
                apply_crop = False
                start_s = None
                end_s = None
                widget_box = crop_settings_widgets.get(path, None)
                if widget_box is not None:
                    apply_crop_toggle = widget_box.children[1].children[0]
                    apply_crop = (apply_crop_toggle.value == 'Yes')
                    start_time_widget = widget_box.children[2].children[0]
                    end_time_widget   = widget_box.children[2].children[1]
                    if apply_crop:
                        start_s = start_time_widget.value
                        end_s   = end_time_widget.value

                if apply_crop and (start_s is not None) and (end_s is not None):
                    sidx = int(start_s * sfreq)
                    eidx = int(end_s * sfreq)
                    sidx = max(0, sidx)
                    eidx = min(eidx, reshaped_data.shape[0])
                    if sidx < eidx:
                        reshaped_data = reshaped_data[sidx:eidx, :]
                        actual_s = (eidx - sidx)/sfreq
                        if actual_s < (end_s - start_s):
                            print(f"  Notice: Adjusted end time for {os.path.basename(path)}.")
                
                # epoch
                epoched_data = segment_data_into_5s_epochs(reshaped_data, sfreq=sfreq, n_channels=16)
                if epoched_data.size == 0:
                    print(f"WARNING: 0 epochs => {os.path.basename(path)}")
                    continue
                
                # optionally split
                stripped_basename = strip_datetime_prefix(os.path.basename(path))
                if do_split_10min:
                    parts_dict = split_into_10min_blocks(epoched_data)
                    if not parts_dict:
                        print(f"No 10-min blocks for {os.path.basename(path)}")
                    else:
                        filename_base = f"{prefix}{stripped_basename}{suffix}" if (prefix or suffix) else stripped_basename
                        pkl_filename = f"{filename_base}_10min.pkl"
                        save_path = os.path.join(out_dir, pkl_filename)
                        try:
                            with open(save_path, 'wb') as f:
                                pickle.dump(parts_dict, f)
                            print(f"Saved 10-min splitted => {pkl_filename}")
                        except Exception as e:
                            print(f"ERROR saving splitted data: {e}")
                else:
                    # save epoched
                    filename_base = f"{prefix}{stripped_basename}{suffix}" if (prefix or suffix) else stripped_basename
                    pkl_filename = f"{filename_base}_epoched.pkl"
                    save_path = os.path.join(out_dir, pkl_filename)
                    try:
                        with open(save_path, 'wb') as f:
                            pickle.dump(epoched_data, f)
                        print(f"Saved epoched => {pkl_filename}")
                    except Exception as e:
                        print(f"ERROR saving epoched data: {e}")
        
        print("\nDONE processing selected files.")

process_button.on_click(on_process_button_clicked)

##############################################################################
# 8) BUILD & DISPLAY THE UI
##############################################################################
verification_section = VBox([
    Label("STEP F) Verify Selected Channels:"),
    verification_output,
    confirm_verification_button
])

ui = VBox([
    # STEP A: Multi-file selection
    Label("STEP A) Use the file chooser below to select one or more .dat files, then click 'Load Selected .dat Files':"),
    file_chooser,
    add_files_button,
    
    # STEP B: Display chosen files
    Label("STEP B) Which files are loaded?"),
    recording_select,
    
    # STEP C: Set Per-File Crop Options
    Label("STEP C) Set per-file time crop options if desired:"),
    crop_options_box,

    # STEP D: Decide whether to concatenate selected files
    Label("STEP D) Concatenate all selected files into a single dataset?"),
    concatenate_toggle,

    # STEP E: Output Directory + filename prefix/suffix
    Label("STEP E) Output folder and optional filename prefix/suffix:"),
    HBox([
        output_dir_text,
        VBox([
            output_filename_prefix_widget,
            output_filename_suffix_widget
        ])
    ]),

    # STEP F: Split final data into 10-minute blocks?
    HBox([
        Label("STEP F) Split final data into 10-min blocks (optional)?"),
        split_10min_toggle
    ]),
    
    # STEP G: Channel Load Mode
    Label("STEP G) Choose Load Mode (Auto/Manual) and then select EXACTLY 16 channels:"),
    load_mode_widget,
    channel_selection_label,
    channel_select_widget,
    
    # STEP H: Channel Verification
    verification_section,
    
    # STEP I: Process
    Label("STEP I) Process your files:"),
    process_button,

    # Log Output
    log_output
])

display(ui)

# Initialize the verification window based on default selections
update_verification_window()
