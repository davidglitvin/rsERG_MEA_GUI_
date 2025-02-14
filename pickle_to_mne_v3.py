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
    Output,
    ToggleButtons
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

def process_pickle(file_path, combine_blocks=True):
    """
    Processes a single pickle file and returns either:
      - A SINGLE mne.Epochs (if combine_blocks=True or data is a single array)
      - A DICTIONARY of {block_name: mne.Epochs} if combine_blocks=False AND data is a dict
      - None if unrecognized or no valid data

    Handles both single concatenated arrays and dictionaries of 10-minute blocks.
    """
    data = load_pickle(file_path)

    if isinstance(data, dict):
        print(f"Detected split blocks in {os.path.basename(file_path)}")
        # data is { block_name:  (n_epochs, n_channels, n_times) }
        if not data:
            print(f"No valid blocks found in {file_path}")
            return None

        all_epochs = []
        block_epochs_dict = {}  # if user wants to keep them separate
        for block_name, block_data in data.items():
            print(f"  Processing block: {block_name} with shape {block_data.shape}")
            # block_data shape: (n_epochs, n_channels, n_times)
            ch_names = [f"Ch{i+1}" for i in range(block_data.shape[1])]
            block_epochs = create_epochs_array(block_data, sfreq=10000, ch_names=ch_names)
            if combine_blocks:
                all_epochs.append(block_epochs)
            else:
                # store separately
                block_epochs_dict[block_name] = block_epochs

        if combine_blocks:
            # Combine all blocks into a single mne.Epochs
            if all_epochs:
                combined_epochs = mne.concatenate_epochs(all_epochs)
                return combined_epochs
            else:
                return None
        else:
            # Return a dictionary of block_name -> mne.Epochs
            return block_epochs_dict

    elif isinstance(data, np.ndarray):
        print(f"Detected single concatenated array in {os.path.basename(file_path)} with shape {data.shape}")
        # data shape: (n_epochs, n_channels, n_times)
        ch_names = [f"Ch{i+1}" for i in range(data.shape[1])]
        epochs = create_epochs_array(data, sfreq=10000, ch_names=ch_names)
        return epochs

    else:
        print(f"Unrecognized data format in {file_path}")
        return None

def concatenate_all_epochs(epochs_list):
    """
    Concatenates a list of MNE Epochs objects into a single Epochs object.
    Returns None if empty.
    """
    if not epochs_list:
        print("No epochs to concatenate.")
        return None
    combined_epochs = mne.concatenate_epochs(epochs_list)
    return combined_epochs


##############################################################################
# 2) GLOBAL VARIABLES & WIDGETS
##############################################################################

file_chooser = FileChooser(
    os.path.expanduser("~"),  
    title='Select Directory with Pickle Files',
    show_only_dirs=True,
    select_multiple=False
)

load_dir_button = Button(
    description='Load Directory',
    button_style='info'
)

# A) A file selector widget for .pkl
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

# B) Buttons to convert
convert_button = widgets.Button(
    description='Convert Selected to MNE',
    button_style='success'
)
convert_all_button = widgets.Button(
    description='Convert ALL to MNE',
    button_style='warning'
)

# **New**: Toggle to decide how we handle 10-min block dictionaries
block_handling_toggle = ToggleButtons(
    options=['Combine Blocks', 'Keep Blocks Separate'],
    value='Combine Blocks',
    description='Block Handling:'
)

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
        for pklf in pkl_files:
            print(f"  - {pklf}")
        print("\nSelect the pickle files you want to load and convert.")

def on_remove_selected_clicked(b):
    """Remove from pkl_select.options whatever is currently highlighted."""
    with output_area:
        clear_output()
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
    """
    Convert the SELECTED .pkl files into MNE data structures.
    If block_handling_toggle = 'Combine Blocks', dictionaries are merged into one Epochs.
    If 'Keep Blocks Separate', returns a dict of block_name->Epochs.
    """
    from IPython import get_ipython

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
        
        # Decide how to handle block dictionaries
        combine_blocks = (block_handling_toggle.value == 'Combine Blocks')
        
        print(f"Converting {len(selected_files)} selected file(s). Block Handling => {block_handling_toggle.value}\n")

        mne_data_list = []  # We'll store each file's result here
        for pkl_file in tqdm(selected_files, desc="Processing pickles"):
            pkl_path = os.path.join(directory, pkl_file)
            if not os.path.isfile(pkl_path):
                print(f"  SKIP: File not found - {pkl_file}")
                continue

            # load, either single MNE Epochs or dict of block_name->Epochs
            result = process_pickle(pkl_path, combine_blocks=combine_blocks)
            if result is not None:
                mne_data_list.append(result)

        if not mne_data_list:
            print("No valid data were loaded from the selected pickles.")
            return
        
        # If the user asked to combine multiple files *and* each file returned a single Epochs,
        # we can combine them all into one. If they want to keep blocks separate, we do not
        # automatically combine across multiple files.
        # For now, let's handle the 'Combine Blocks' scenario:
        if combine_blocks:
            # Check if all are single MNE Epochs objects
            if all(isinstance(item, mne.Epochs) for item in mne_data_list):
                # Combine them into one big epochs
                all_combined = concatenate_all_epochs(mne_data_list)
                if all_combined:
                    print(f"\nSuccessfully combined {len(mne_data_list)} files into ONE Epochs object.")
                    print(f"Total epochs: {len(all_combined)}")
                    get_ipython().user_ns["mne_data_objects"] = all_combined
                    print("Stored as 'mne_data_objects' in the notebook's namespace (single mne.Epochs).")
                    return
                else:
                    print("Failed to concatenate the MNE Epochs across files.")
                    return
            else:
                # Some file returned a dictionary => let's just store them all in a list
                print("Some files returned dictionaries or had no data. Storing results in a list.")
                get_ipython().user_ns["mne_data_objects"] = mne_data_list
                print("Stored as 'mne_data_objects' in the notebook's namespace (list of items).")
                return
        else:
            # 'Keep Blocks Separate': we have either
            # - a dictionary for each file (block_name->Epochs), or
            # - a single mne.Epochs if the file was a normal array, or
            # - a mix
            print("Keeping blocks separate. Storing each file's result in a list.")
            get_ipython().user_ns["mne_data_objects"] = mne_data_list
            print("Stored as 'mne_data_objects' in the notebook's namespace (list of items).")
            return

def on_convert_all_clicked(b):
    """
    Convert ALL .pkl files in the folder using the same block handling logic.
    """
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
        
        combine_blocks = (block_handling_toggle.value == 'Combine Blocks')
        print(f"Converting ALL {len(all_files)} file(s). Block Handling => {block_handling_toggle.value}\n")

        mne_data_list = []
        for pkl_file in tqdm(all_files, desc="Processing pickles"):
            pkl_path = os.path.join(directory, pkl_file)
            if not os.path.isfile(pkl_path):
                print(f"  SKIP: File not found - {pkl_file}")
                continue

            result = process_pickle(pkl_path, combine_blocks=combine_blocks)
            if result is not None:
                mne_data_list.append(result)

        if not mne_data_list:
            print("No valid data were loaded from the pickle files.")
            return
        
        # If user says combine blocks, let's see if we can also combine multiple files
        if combine_blocks:
            if all(isinstance(item, mne.Epochs) for item in mne_data_list):
                all_combined = concatenate_all_epochs(mne_data_list)
                if all_combined:
                    print(f"\nSuccessfully combined {len(mne_data_list)} files into ONE Epochs object.")
                    print(f"Total epochs: {len(all_combined)}")
                    get_ipython().user_ns["mne_data_objects"] = all_combined
                    print("Stored as 'mne_data_objects' in the notebook's namespace (single mne.Epochs).")
                    return
                else:
                    print("Failed to concatenate the MNE Epochs across files.")
                    return
            else:
                print("Some files returned dictionaries or had no data. Storing results in a list.")
                get_ipython().user_ns["mne_data_objects"] = mne_data_list
                print("Stored as 'mne_data_objects' in the notebook's namespace (list of items).")
                return
        else:
            # Keep blocks separate
            print("Keeping blocks separate. Storing each file's result in a list.")
            get_ipython().user_ns["mne_data_objects"] = mne_data_list
            print("Stored as 'mne_data_objects' in the notebook's namespace (list of items).")
            return


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
    VBox([
        Label("Select Directory Containing Pickle Files:"),
        file_chooser,
        load_dir_button
    ]),
    HBox([
        VBox([
            Label("Pickle Files:"),
            pkl_select,
            remove_selected_button
        ]),
    ]),
    VBox([
        Label("Conversion Options:"),
        HBox([convert_button, convert_all_button]),
        Label("How to handle 10-min block dictionaries?"),
        block_handling_toggle
    ]),
    output_area
])

display(ui)
