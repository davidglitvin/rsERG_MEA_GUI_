import os
import pickle
import numpy as np
import mne
from tqdm.notebook import tqdm

import ipywidgets as widgets
from ipywidgets import (
    SelectMultiple,
    Button,
    VBox,
    HBox,
    Label,
    Output
)
from IPython.display import display, clear_output

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

def load_pickle(file_path):
    """Loads a pickle file and returns the data."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

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
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    epochs = mne.EpochsArray(data, info)
    return epochs

def process_pickle(file_path):
    """
    Processes a single pickle file and returns an MNE Epochs object.
    Handles both single concatenated arrays and dictionaries of 10-minute blocks.
    """
    data = load_pickle(file_path)
    
    # Determine if data is a dictionary (split into 10min blocks) or a single array
    if isinstance(data, dict):
        print(f"Detected split blocks in {os.path.basename(file_path)}")
        all_epochs = []
        for block_name, block_data in data.items():
            print(f"  Processing block: {block_name} with shape {block_data.shape}")
            # Assume block_data shape is (n_epochs, n_channels, n_times)
            ch_names = [f"Ch{i+1}" for i in range(block_data.shape[1])]
            epochs = create_epochs_array(block_data, sfreq=10000, ch_names=ch_names)
            all_epochs.append(epochs)
        if all_epochs:
            combined_epochs = mne.concatenate_epochs(all_epochs)
            return combined_epochs
        else:
            print(f"No valid blocks found in {file_path}")
            return None
    elif isinstance(data, np.ndarray):
        print(f"Detected single concatenated array in {os.path.basename(file_path)} with shape {data.shape}")
        # Assume data shape is (n_epochs, n_channels, n_times)
        ch_names = [f"Ch{i+1}" for i in range(data.shape[1])]
        epochs = create_epochs_array(data, sfreq=10000, ch_names=ch_names)
        return epochs
    else:
        print(f"Unrecognized data format in {file_path}")
        return None

def concatenate_all_epochs(epochs_list):
    """
    Concatenates a list of MNE Epochs objects into a single Epochs object.
    """
    if not epochs_list:
        print("No epochs to concatenate.")
        return None
    combined_epochs = mne.concatenate_epochs(epochs_list)
    return combined_epochs

##############################################################################
# 2) GLOBAL VARIABLES & WIDGETS
##############################################################################

# A) FileChooser widget for selecting the directory containing .pkl files
file_chooser = FileChooser(
    os.path.expanduser("~"),  # Starting directory, adjust as needed
    title='Select Directory with Pickle Files',
    show_only_dirs=True,
    select_multiple=False
)

load_dir_button = Button(
    description='Load Directory',
    button_style='info'
)

# B) A file selector widget. We use SelectMultiple so you can pick more than one.
pkl_select = widgets.SelectMultiple(
    options=[],
    description='Pickle(s):',
    layout=widgets.Layout(width='800px', height='300px')
)

remove_selected_button = widgets.Button(
    description='Remove Selected',
    button_style='danger',
    tooltip='Remove highlighted items from the list above'
)

# C) Buttons to convert
convert_button = widgets.Button(
    description='Convert Selected to MNE',
    button_style='success'
)
convert_all_button = widgets.Button(
    description='Convert ALL to MNE',
    button_style='warning'
)

# D) Output area
output_area = widgets.Output()

##############################################################################
# 3) CALLBACK FUNCTIONS
##############################################################################

def list_pkl_files_in_directory(directory):
    """Return a sorted list of .pkl files in the given directory."""
    if not os.path.isdir(directory):
        return []
    all_files = os.listdir(directory)
    pkl_files = [f for f in all_files if f.endswith('.pkl')]
    return sorted(pkl_files)

def on_load_dir_clicked(b):
    """Populate pkl_select.options with all .pkl files in the chosen directory."""
    with output_area:
        clear_output()
        directory = file_chooser.selected_path
        if not directory:
            print("No directory selected.")
            return
        if not os.path.isdir(directory):
            print(f"Invalid directory: {directory}")
            return
        
        pkl_files = list_pkl_files_in_directory(directory)
        if not pkl_files:
            print(f"No .pkl files found in {directory}.")
            pkl_select.options = []
            return
        
        pkl_select.options = pkl_files
        print(f"Found {len(pkl_files)} pickle files in {directory}:")
        for pkl in pkl_files:
            print(f"  - {pkl}")
        print("\nSelect the pickle files you want to load and convert.")

def on_remove_selected_clicked(b):
    """Remove from pkl_select.options whatever is currently highlighted."""
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

def on_convert_button_clicked(b):
    """Convert the SELECTED .pkl files into MNE Epochs."""
    from IPython import get_ipython  # We'll need this to set the var in the notebook

    with output_area:
        clear_output()
        directory = file_chooser.selected_path
        selected_files = list(pkl_select.value)
        
        if not directory:
            print("No directory selected.")
            return
        if not selected_files:
            print("No pickle files selected for conversion.")
            return
        
        print(f"Converting {len(selected_files)} selected pickle files into MNE Epochs...\n")
        
        epochs_list = []
        for pkl_file in tqdm(selected_files, desc="Processing pickles"):
            pkl_path = os.path.join(directory, pkl_file)
            if not os.path.isfile(pkl_path):
                print(f"  SKIP: File not found - {pkl_file}")
                continue
            epochs = process_pickle(pkl_path)
            if epochs:
                epochs_list.append(epochs)
        
        if not epochs_list:
            print("No valid epochs were loaded from the selected pickles.")
            return
        
        # Concatenate all epochs into a single Epochs object
        combined_epochs = concatenate_all_epochs(epochs_list)
        if combined_epochs:
            print(f"\nSuccessfully combined all epochs into a single Epochs object.")
            print(f"Total epochs: {len(combined_epochs)}")
            
            # Store the combined epochs in the NOTEBOOK's namespace
            get_ipython().user_ns["mne_combined_epochs"] = combined_epochs
            print("Assigned 'mne_combined_epochs' in the notebook's global namespace.")
        else:
            print("Failed to concatenate epochs.")

def on_convert_all_clicked(b):
    """Convert ALL .pkl files into MNE Epochs."""
    from IPython import get_ipython

    with output_area:
        clear_output()
        directory = file_chooser.selected_path
        all_files = list(pkl_select.options)
        
        if not directory:
            print("No directory selected.")
            return
        if not all_files:
            print("No pickle files available for conversion.")
            return
        
        print(f"Converting ALL {len(all_files)} pickle files into MNE Epochs...\n")
        
        epochs_list = []
        for pkl_file in tqdm(all_files, desc="Processing pickles"):
            pkl_path = os.path.join(directory, pkl_file)
            if not os.path.isfile(pkl_path):
                print(f"  SKIP: File not found - {pkl_file}")
                continue
            epochs = process_pickle(pkl_path)
            if epochs:
                epochs_list.append(epochs)
        
        if not epochs_list:
            print("No valid epochs were loaded from the pickle files.")
            return
        
        combined_epochs = concatenate_all_epochs(epochs_list)
        if combined_epochs:
            print(f"\nSuccessfully combined all epochs into a single Epochs object.")
            print(f"Total epochs: {len(combined_epochs)}")
            
            # Store in the notebook namespace
            get_ipython().user_ns["mne_combined_epochs"] = combined_epochs
            print("Assigned 'mne_combined_epochs' in the notebook's global namespace.")
        else:
            print("Failed to concatenate epochs.")

##############################################################################
# 4) WIDGET EVENTS
##############################################################################
load_dir_button.on_click(on_load_dir_clicked)
remove_selected_button.on_click(on_remove_selected_clicked)
convert_button.on_click(on_convert_button_clicked)
convert_all_button.on_click(on_convert_all_clicked)

##############################################################################
# 5) BUILD & DISPLAY THE UI
##############################################################################

ui = VBox([
    # STEP A: File Chooser for Directory
    VBox([
        Label("Select Directory Containing Pickle Files:"),
        file_chooser,
        load_dir_button
    ]),
    
    # STEP B: File Selection and Removal
    HBox([
        VBox([
            Label("Pickle Files:"),
            pkl_select,
            remove_selected_button
        ]),
    ]),
    
    # STEP C: Conversion Buttons
    VBox([
        Label("Conversion Options:"),
        HBox([
            convert_button,
            convert_all_button
        ])
    ]),
    
    # STEP D: Output Area
    output_area
])

display(ui)
