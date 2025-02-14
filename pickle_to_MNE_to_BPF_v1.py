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
    epochs = mne.EpochsArray(data, info, verbose=False)
    return epochs

def process_pickle(file_path):
    """
    Processes a single pickle file and returns an MNE Epochs object.
    Handles both single concatenated arrays and dictionaries of 10-minute blocks.
    """
    data = load_pickle(file_path)
    if isinstance(data, dict):
        print(f"Detected split blocks in {os.path.basename(file_path)}")
        all_epochs = []
        for block_name, block_data in data.items():
            print(f"  Processing block: {block_name} with shape {block_data.shape}")
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
        ch_names = [f"Ch{i+1}" for i in range(data.shape[1])]
        epochs = create_epochs_array(data, sfreq=10000, ch_names=ch_names)
        return epochs
    else:
        print(f"Unrecognized data format in {file_path}")
        return None

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
        print(f"Processing {len(file_list)} file(s)...\n")
    
    for pkl_file in tqdm(file_list, desc="Processing files"):
        pkl_path = os.path.join(input_directory, pkl_file)
        with output_area:
            print(f"\nProcessing file: {pkl_file}")
        if not os.path.isfile(pkl_path):
            with output_area:
                print(f"  SKIP: File not found - {pkl_file}")
            continue
        try:
            epochs = process_pickle(pkl_path)
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
                verbose=False
            )
            with output_area:
                print("  Bandpass filter applied successfully.")
        except Exception as e:
            with output_area:
                print(f"  ERROR: Failed to apply filter to {pkl_file}: {e}")
            continue
        
        # --- Build a Unique Output Filename ---
        output_path = get_unique_output_path(output_directory, pkl_file)
        
        # --- Save Filtered Epochs ---
        try:
            epochs_filtered.save(output_path, overwrite=True)
            with output_area:
                print(f"  Filtered epochs saved to: {output_path}")
        except Exception as e:
            with output_area:
                print(f"  ERROR: Failed to save filtered epochs for {pkl_file}: {e}")
            continue

def on_process_selected_clicked(b):
    """Process and filter only the selected pickle files."""
    selected_files = list(pkl_select.value)
    if not selected_files:
        with output_area:
            clear_output()
            print("No pickle files selected for processing.")
        return
    with output_area:
        clear_output()
    process_and_filter_files(selected_files)

def on_process_all_clicked(b):
    """Process and filter all pickle files in the list."""
    all_files = list(pkl_select.options)
    if not all_files:
        with output_area:
            clear_output()
            print("No pickle files available for processing.")
        return
    with output_area:
        clear_output()
    process_and_filter_files(all_files)

# --- Connect Callbacks to Buttons ---
load_input_dir_button.on_click(on_load_input_dir_clicked)
remove_selected_button.on_click(on_remove_selected_clicked)
process_selected_button.on_click(on_process_selected_clicked)
process_all_button.on_click(on_process_all_clicked)

##############################################################################
# BUILD & DISPLAY THE UI
##############################################################################

input_ui = VBox([
    Label("Step 1: Select Input Directory Containing Pickle Files:"),
    input_dir_chooser,
    load_input_dir_button,
    Label("Available Pickle Files:"),
    pkl_select,
    remove_selected_button
])

filter_ui = VBox([
    Label("Step 2: Set Bandpass Filter Parameters:"),
    HBox([bandpass_l_freq_widget, bandpass_h_freq_widget, fir_design_widget, filter_length_widget])
])

output_ui = VBox([
    Label("Step 3: Select Output Directory for Filtered FIF Files:"),
    output_dir_chooser
])

process_ui = HBox([process_selected_button, process_all_button])

main_ui = VBox([
    input_ui,
    filter_ui,
    output_ui,
    process_ui,
    output_area
])

display(main_ui)
