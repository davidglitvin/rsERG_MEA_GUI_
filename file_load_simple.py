import os
import pickle
import numpy as np

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
    Output
)
from IPython.display import display, clear_output
from tqdm.notebook import tqdm

from ipyfilechooser import FileChooser

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
# 2) GLOBAL VARIABLES & WIDGETS
##############################################################################
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
    layout=widgets.Layout(width='600px', height='250px'),
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
    layout=widgets.Layout(width='600px')
)

# H) Container for per-recording crop options
crop_options_box = VBox()

# I) Toggle to optionally split concatenated data into 10-minute blocks
split_10min_toggle = ToggleButtons(
    options=['No', 'Yes'],
    value='No',
    description='Split 10min?'
)

# J) Toggle to decide whether to concatenate selected recordings
concatenate_toggle = ToggleButtons(
    options=['No', 'Yes'],
    value='No',
    description='Concatenate Selected Recordings?'
)

# K) Button to start processing
process_button = Button(
    description='Process Selected Recordings',
    button_style='success'
)

# L) Output area for logs
log_output = Output()

##############################################################################
# 3) DIRECTORY MANIPULATION CALLBACKS
##############################################################################
def on_add_directory_clicked(b):
    """
    Adds the selected directory from FileChooser to directories_list and updates the UI.
    """
    with log_output:
        clear_output()
        selected_dir = fc.selected_path
        if not selected_dir:
            print("No directory selected.")
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
# 4) LOGIC TO FIND OPEN EPHYS RECORDINGS
##############################################################################
def find_open_ephys_recordings_in_dir(base_dir):
    """
    Finds **all folders** within base_dir that directly contain 'continuous.dat'.

    Returns:
    - matches: a list of folder paths where 'continuous.dat' is found.
    """
    matches = []
    for root, dirs, files in os.walk(base_dir):
        if "continuous.dat" in files:
            matches.append(root)
    return matches

def on_load_dirs_clicked(b):
    """
    Scans added directories to find folders with continuous.dat and updates the UI.
    Also logs # of epochs based on file size for each found folder.
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
                        # Calculate number of 5-second epochs
                        file_size_bytes = os.path.getsize(dat_file)
                        bytes_per_sample = 2  # int16
                        n_channels = 16
                        sfreq = 10000
                        epoch_duration = 5  # seconds
                        epoch_length = sfreq * epoch_duration  # samples per epoch

                        total_samples = file_size_bytes / bytes_per_sample
                        number_of_channels = n_channels
                        number_of_frames = total_samples / number_of_channels
                        number_of_epochs = int(number_of_frames // epoch_length)

                        print(f"{strip_datetime_prefix(basename)} => {number_of_epochs} epochs (folder: {path})")
                else:
                    print("No folders with continuous.dat found here.")
            
            acc = Accordion(children=[out])
            acc.set_title(0, d)
            children_for_accordion.append(acc)

        directory_accordion.children = children_for_accordion
        
        # Deduplicate and sort
        all_recordings = list(set(all_recordings))
        all_recordings.sort()

        recording_select.options = all_recordings
        print(f"Found {len(all_recordings)} valid folder(s) from {len(directories_list)} directories.")

load_dirs_button.on_click(on_load_dirs_clicked)

##############################################################################
# 5) PER-RECORDING CROP OPTIONS CALLBACKS
##############################################################################
crop_settings_widgets = {}  # Key: recording path, Value: widget VBox

def create_crop_widgets_for_recording(recording_path):
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
    
    # Clear existing widgets
    crop_options_box.children = []
    
    # Remove widgets for recordings no longer selected
    to_remove = [path for path in crop_settings_widgets if path not in selected_recordings]
    for path in to_remove:
        del crop_settings_widgets[path]
    
    # Add widgets for newly selected recordings
    for path in selected_recordings:
        if path not in crop_settings_widgets:
            crop_settings_widgets[path] = create_crop_widgets_for_recording(path)
    
    # Display
    crop_widgets = [crop_settings_widgets[path] for path in selected_recordings]
    crop_options_box.children = crop_widgets

recording_select.observe(update_crop_options_box, names='value')

##############################################################################
# 6) PROCESSING LOGIC
##############################################################################
def on_process_button_clicked(b):
    """
    For each selected path (which directly contains continuous.dat):
      - Loads data from 'continuous.dat' (16 channels, int16, 10kHz)
      - Optional crop
      - Epochs into 5s intervals
      - If concatenation is selected, we combine them
      - If 10-min split is selected, we split and pickle accordingly
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
        
        # Gather crop settings
        apply_crop_dict = {}
        crop_times_dict = {}
        
        for path in selected_paths:
            widgets_box = crop_settings_widgets.get(path, None)
            if widgets_box:
                apply_crop_toggle = widgets_box.children[1].children[0]
                start_time = widgets_box.children[2].children[0]
                end_time = widgets_box.children[2].children[1]
                
                apply_crop = (apply_crop_toggle.value == 'Yes')
                if apply_crop:
                    s_time = start_time.value
                    e_time = end_time.value
                    if s_time < 0:
                        s_time = 0
                    if e_time <= s_time:
                        print(f"WARNING: For '{os.path.basename(path)}', end time <= start time. Ignoring crop.")
                        apply_crop = False
                else:
                    s_time, e_time = None, None
                apply_crop_dict[path] = apply_crop
                crop_times_dict[path] = (s_time, e_time) if apply_crop else (None, None)
        
        concatenate_selected = (concatenate_toggle.value == 'Yes')
        do_split_10min = (split_10min_toggle.value == 'Yes')

        def get_stripped_basename(path):
            return strip_datetime_prefix(os.path.basename(path))
        
        # We'll store epoched data if we want to combine
        epoched_list = []
        final_labels = []
        
        # If we want to combine, or if we have single recordings, we do one pass
        if concatenate_selected and len(selected_paths) >= 2:
            print("Concatenation selected. Processing recordings to concatenate...")
        else:
            print("Processing recordings individually (or only one).")
        
        # We'll handle all recordings in one loop
        for path in selected_paths:
            dat_file = os.path.join(path, "continuous.dat")
            if not os.path.exists(dat_file):
                print(f"SKIP: 'continuous.dat' not found under '{path}'")
                continue

            # Load the raw data
            try:
                raw_data = np.fromfile(dat_file, dtype=np.int16)
            except Exception as e:
                print(f"  ERROR: Failed to load '{dat_file}': {e}")
                continue

            n_channels = 16
            sfreq = 10000
            try:
                reshaped_data = raw_data.reshape(-1, n_channels)
            except Exception as e:
                print(f"  ERROR: Failed to reshape data from '{dat_file}': {e}")
                continue

            total_samples = reshaped_data.shape[0]
            apply_crop = apply_crop_dict.get(path, False)
            start_s, end_s = crop_times_dict.get(path, (None, None))

            if apply_crop and (start_s is not None) and (end_s is not None):
                sidx = int(start_s * sfreq)
                eidx = int(end_s * sfreq)
                sidx = max(0, sidx)
                eidx = min(eidx, total_samples)
                if sidx >= eidx:
                    print(f"  WARNING: For '{os.path.basename(path)}', start_idx >= end_idx. Skipping crop.")
                    cropped_data = reshaped_data
                else:
                    cropped_data = reshaped_data[sidx:eidx, :]
                    actual_duration = eidx - sidx
                    actual_duration_s = actual_duration / sfreq
                    if actual_duration_s < (end_s - start_s):
                        print(f"  NOTICE: For '{os.path.basename(path)}', end time exceeds file length. "
                              f"Adjusted end time to {actual_duration_s:.2f} s.")
            else:
                cropped_data = reshaped_data

            # Epoch
            epoched_data = segment_data_into_5s_epochs(cropped_data, sfreq=sfreq, n_channels=n_channels)
            if epoched_data.size == 0:
                print(f"  WARNING: No epochs created for '{os.path.basename(path)}'. Skipping.")
                continue

            # If we are concatenating:
            if concatenate_selected and len(selected_paths) >= 2:
                epoched_list.append(epoched_data)
                final_labels.append(get_stripped_basename(path))
                print(f"Loaded & epoched => '{get_stripped_basename(path)}'  Shape={epoched_data.shape}, Crop={apply_crop}")
            else:
                # Process individually
                basename_str = get_stripped_basename(path)
                # Optionally split into 10-min
                if do_split_10min:
                    parts_dict = split_into_10min_blocks(epoched_data)
                    if not parts_dict:
                        print(f"  * No full 10min blocks found for '{os.path.basename(path)}'")
                    else:
                        pkl_filename = f"{basename_str}_concat_with_10min_split.pkl"
                        save_path = os.path.join(out_dir, pkl_filename)
                        try:
                            with open(save_path, 'wb') as f:
                                pickle.dump(parts_dict, f)
                            print(f"  -> Saved 10min blocks: {list(parts_dict.keys())} => '{pkl_filename}'")
                        except Exception as e:
                            print(f"  ERROR: Failed to save 10min blocks: {e}")
                else:
                    pkl_filename = f"{basename_str}_epoched.pkl"
                    save_path = os.path.join(out_dir, pkl_filename)
                    try:
                        with open(save_path, 'wb') as f:
                            pickle.dump(epoched_data, f)
                        print(f"  -> Saved epoched array => '{pkl_filename}'")
                    except Exception as e:
                        print(f"  ERROR: Failed to save epoched data: {e}")

        # Now handle the concatenated case if needed
        if concatenate_selected and len(selected_paths) >= 2:
            if not epoched_list:
                print("No valid recordings to concatenate.")
            else:
                # Merge all epoched
                try:
                    concatenated_epoched = np.concatenate(epoched_list, axis=0)
                    concatenated_label = "&".join(final_labels)
                    print(f"Concatenated => '{concatenated_label}', shape={concatenated_epoched.shape}")
                except Exception as e:
                    print(f"ERROR: Failed to concatenate: {e}")
                    concatenated_epoched = None

                if concatenated_epoched is not None:
                    if do_split_10min:
                        parts_dict = split_into_10min_blocks(concatenated_epoched)
                        if not parts_dict:
                            print(f"  * No full 10min blocks found in concatenated data.")
                        else:
                            pkl_filename = f"{concatenated_label}_concat_with_10min_split.pkl"
                            save_path = os.path.join(out_dir, pkl_filename)
                            try:
                                with open(save_path, 'wb') as f:
                                    pickle.dump(parts_dict, f)
                                print(f"  -> Saved 10min blocks => '{pkl_filename}'")
                            except Exception as e:
                                print(f"  ERROR: Failed to save 10min blocks: {e}")
                    else:
                        pkl_filename = f"{concatenated_label}_concat.pkl"
                        save_path = os.path.join(out_dir, pkl_filename)
                        try:
                            with open(save_path, 'wb') as f:
                                pickle.dump(concatenated_epoched, f)
                            print(f"  -> Saved concatenated epoched => '{pkl_filename}'")
                        except Exception as e:
                            print(f"  ERROR: Failed to save concatenated data: {e}")

        print("\nDONE processing selected recordings.")

process_button.on_click(on_process_button_clicked)

##############################################################################
# 7) BUILD & DISPLAY THE UI
##############################################################################
ui = VBox([
    # STEP A: Add Directory
    Label("STEP A) Use the file chooser to select a top-level directory, then click 'Add Selected Directory':"),
    fc,
    add_directory_button,

    # STEP B: Remove Directories
    Label("STEP B) Remove directories if needed:"),
    dir_remove_select,
    remove_directory_button,
    
    # STEP C: Load Recordings
    Label("STEP C) Search subdirectories for folders containing continuous.dat:"),
    load_dirs_button,
    directory_accordion,
    
    # STEP D: Select Recordings to Process
    Label("STEP D) Choose which recordings to process (each folder has its own continuous.dat):"),
    recording_select,

    # STEP E: Set Per-Recording Crop Options
    Label("STEP E) Set per-recording time crop options (optional):"),
    crop_options_box,

    # STEP F: Concatenation Option
    Label("STEP F) Concatenate the selected recordings?"),
    concatenate_toggle,

    # STEP G: Output Directory
    Label("STEP G) Output folder:"),
    output_dir_text,

    # STEP H: Split final data into 10-minute blocks?
    split_10min_toggle,

    # STEP I: Process Button
    process_button,

    # Log Output
    log_output
])

display(ui)
