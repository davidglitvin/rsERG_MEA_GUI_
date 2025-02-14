import os
import mne
from tqdm.notebook import tqdm

import ipywidgets as widgets
from ipyfilechooser import FileChooser
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

from IPython import get_ipython  # We read from the notebook's namespace

# 1. Low & High Cutoff
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

# 2. FIR Design & Filter Length
fir_design_widget = widgets.Dropdown(
    options=['firwin', 'firwin2'],
    value='firwin',
    description='FIR design:',
    layout=widgets.Layout(width='200px'),
    tooltip='Select FIR filter design method'
)

filter_length_widget = widgets.Text(
    value='auto',
    description='Filter Len:',
    layout=widgets.Layout(width='160px'),
    tooltip='Enter # of taps (odd integer) or "auto"'
)

# 3. Output FileChooser
output_fif_chooser = FileChooser(
    os.getcwd(),
    title='Select Base .fif File to Save',
    select_default=False
)
output_fif_chooser.show_only_files = False
output_fif_chooser.filter_pattern = ['*.fif', '*.fif.gz']
output_fif_chooser.default_filename = 'filtered_epochs.fif'

# 4. Apply Filter Button
apply_filter_button = widgets.Button(
    description='Apply Bandpass Filter',
    button_style='success',
    tooltip='Apply bandpass filter to "mne_data_objects" and save as .fif'
)

# 5. Output Area
bandpass_output_area = widgets.Output()


def filter_single_epochs(
    epochs: mne.Epochs,
    l_freq: float,
    h_freq: float,
    fir_design: str,
    filter_length: str
) -> mne.Epochs:
    """Utility to filter a single mne.Epochs object."""
    filtered = epochs.copy().filter(
        l_freq=l_freq,
        h_freq=h_freq,
        fir_design=fir_design,
        filter_length=filter_length,
        verbose=False
    )
    return filtered


def on_apply_filter_clicked(b):
    with bandpass_output_area:
        clear_output()
        
        # Retrieve "mne_data_objects" from notebook environment
        try:
            data_obj = get_ipython().user_ns["mne_data_objects"]
        except KeyError:
            print("ERROR: 'mne_data_objects' is not defined in the notebook namespace.")
            print("Please load your data first via the updated pickle-to-mne step.")
            return
        
        # Parse filter parameters
        def parse_freq(value_text):
            vt = value_text.strip().lower()
            if vt == 'none':
                return None
            else:
                return float(vt)

        try:
            l_freq = parse_freq(bandpass_l_freq_widget.value)
            if l_freq is not None and l_freq < 0:
                print(f"ERROR: Low frequency must be >= 0. Got: {l_freq}")
                return
        except ValueError:
            print(f"ERROR: Could not parse low freq '{bandpass_l_freq_widget.value}' as float/None.")
            return

        try:
            h_freq = parse_freq(bandpass_h_freq_widget.value)
            if h_freq is not None and h_freq < 0:
                print(f"ERROR: High frequency must be >= 0. Got: {h_freq}")
                return
        except ValueError:
            print(f"ERROR: Could not parse high freq '{bandpass_h_freq_widget.value}' as float/None.")
            return
        
        fir_design = fir_design_widget.value
        
        filter_len_text = filter_length_widget.value.strip().lower()
        if filter_len_text == 'auto':
            filter_length = 'auto'
        else:
            try:
                flen = int(filter_len_text)
                if flen % 2 == 0 or flen < 1:
                    print(f"ERROR: Filter length must be a positive odd integer. Got: {flen}")
                    return
                filter_length = flen
            except ValueError:
                print(f"ERROR: Could not parse filter length '{filter_len_text}' as integer or 'auto'.")
                return
        
        # Validate output file
        if not output_fif_chooser.selected:
            print("ERROR: No output file selected. Please choose an output file.")
            return
        out_base_path = output_fif_chooser.selected
        # Example: user picks "filtered_epochs.fif" => we might store block-based files with a suffix

        # Make sure the base path ends with .fif or .fif.gz
        if not (out_base_path.endswith('.fif') or out_base_path.endswith('.fif.gz')):
            print("ERROR: Output file must end with .fif or .fif.gz.")
            return
        
        # Summarize parameters
        print(f"Bandpass filter parameters:")
        print(f"  - Low Freq: {l_freq}")
        print(f"  - High Freq: {h_freq}")
        print(f"  - FIR Design: {fir_design}")
        print(f"  - Filter Length: {filter_length}")
        print(f"  - Output Base File: {out_base_path}\n")
        
        # We'll define a helper to build an output filename from the user's base
        # If the user wants to keep blocks separate or multiple items, we add suffixes
        base_root, base_ext = os.path.splitext(out_base_path)
        def build_output_filename(suffix: str):
            # If original was .fif.gz, base_ext = '.fif.gz'
            # We can handle that by checking if .gz in base_ext
            if base_ext == '.gz':
                # means we likely have something like .fif.gz
                # We can do a little trick:
                base_root2, ext2 = os.path.splitext(base_root)
                # base_root2 => 'filtered_epochs'
                # ext2 => '.fif'
                return f"{base_root2}_{suffix}{ext2}{base_ext}"
            else:
                # normal approach
                return f"{base_root}_{suffix}{base_ext}"
        
        # We apply the filter to "data_obj" which could be:
        # 1) A single mne.Epochs
        # 2) A dictionary: {block_name: mne.Epochs}
        # 3) A list (of single Epochs or dictionaries)
        # We'll store the results in a new var => "filtered_data"
        
        def filter_and_save_epochs(epochs_obj, out_filename):
            """Filter a single mne.Epochs and save to out_filename."""
            print(f"  Filtering single Epochs => saving to {out_filename}")
            try:
                filtered_ep = filter_single_epochs(
                    epochs_obj, l_freq, h_freq, fir_design, filter_length
                )
                filtered_ep.save(out_filename, overwrite=True)
                print(f"    -> Done.")
            except Exception as e:
                print(f"    ERROR filtering or saving: {e}")
        
        def filter_dict_of_epochs(block_dict, base_filename):
            """Filter each block in a dictionary -> each block saved as separate .fif."""
            for block_name, block_epochs in block_dict.items():
                block_out = build_output_filename(block_name)
                filter_and_save_epochs(block_epochs, block_out)
        
        def process_item(item, idx=0):
            """Handle a single item that might be an Epochs or a dict of block_name->Epochs."""
            if isinstance(item, mne.Epochs):
                # single
                suffix = f"item{idx}"
                out_file = build_output_filename(suffix)
                filter_and_save_epochs(item, out_file)
            elif isinstance(item, dict):
                # multiple blocks
                print(f"Processing dict of blocks (item{idx}).")
                filter_dict_of_epochs(item, out_base_path)
            else:
                print(f"SKIP: Unrecognized item type {type(item)} at index {idx}.")
        
        # Main logic
        #-----------
        if isinstance(data_obj, mne.Epochs):
            # A single epochs
            print("Filtering a single mne.Epochs object.")
            filter_and_save_epochs(data_obj, out_base_path)
        elif isinstance(data_obj, dict):
            # Dictionary of blocks
            print("Filtering a dictionary {block_name: mne.Epochs}.")
            filter_dict_of_epochs(data_obj, out_base_path)
        elif isinstance(data_obj, list):
            # Could be a list of single or dict
            print(f"Filtering a list of {len(data_obj)} item(s).")
            for i, val in enumerate(data_obj):
                process_item(val, i)
        else:
            print(f"ERROR: Unrecognized data type => {type(data_obj)}. Nothing to filter.")
        
        print("\nDONE filtering & saving. Check the log above for details.")

apply_filter_button.on_click(on_apply_filter_clicked)

# Layout
filter_params_box = widgets.HBox([
    bandpass_l_freq_widget,
    bandpass_h_freq_widget,
    fir_design_widget,
    filter_length_widget
])

bandpass_ui = widgets.VBox([
    widgets.HTML("<h2>Bandpass Filter Application</h2>"),
    widgets.Label("### Filter Parameters:"),
    filter_params_box,
    widgets.Label("### Select Base Output .fif File:"),
    output_fif_chooser,
    apply_filter_button,
    bandpass_output_area
])

display(bandpass_ui)
