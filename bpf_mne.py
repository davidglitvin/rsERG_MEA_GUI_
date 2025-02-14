import os
import mne
from tqdm.notebook import tqdm

import ipywidgets as widgets
from ipyfilechooser import FileChooser
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

# We will read from the notebook namespace (where "pickle_to_mne" put the variable).
from IPython import get_ipython

# 1. Low Cutoff Frequency Widget
bandpass_l_freq_widget = widgets.Text(
    value='0',
    description='Low freq (Hz):',
    layout=widgets.Layout(width='200px'),
    tooltip='Enter low cutoff frequency in Hz or "None" to disable'
)

# 2. High Cutoff Frequency Widget
bandpass_h_freq_widget = widgets.Text(
    value='45',
    description='High freq (Hz):',
    layout=widgets.Layout(width='200px'),
    tooltip='Enter high cutoff frequency in Hz or "None" to disable'
)

# 3. FIR Design Dropdown
fir_design_widget = widgets.Dropdown(
    options=['firwin', 'firwin2'],
    value='firwin',
    description='FIR design:',
    layout=widgets.Layout(width='200px'),
    tooltip='Select FIR filter design method'
)

# 4. Filter Length Widget
filter_length_widget = widgets.Text(
    value='auto',
    description='Filter Length (taps):',
    layout=widgets.Layout(width='250px'),
    tooltip='Enter number of taps (odd integer) or "auto" for automatic selection'
)

# 5. Output FileChooser Widget
output_fif_chooser = FileChooser(
    os.getcwd(),
    title='Select Output .fif File',
    select_default=False
)
output_fif_chooser.show_only_files = False
output_fif_chooser.filter_pattern = ['*.fif', '*.fif.gz']
output_fif_chooser.default_filename = 'filtered_epochs.fif'

# 6. Apply Filter Button
apply_filter_button = widgets.Button(
    description='Apply Bandpass Filter',
    button_style='success',
    tooltip='Apply bandpass filter to mne_combined_epochs and save as .fif'
)

# 7. Output Area
bandpass_output_area = widgets.Output()

def on_apply_filter_clicked(b):
    with bandpass_output_area:
        clear_output()
        
        # Attempt to retrieve the 'mne_combined_epochs' from the notebook's user namespace
        try:
            # This reads from the interactive notebook environment
            epochs = get_ipython().user_ns["mne_combined_epochs"]
        except KeyError:
            print("ERROR: 'mne_combined_epochs' is not defined in the notebook namespace.")
            print("Please load your Epochs object first via the 'pickle_to_mne' step.")
            return
        
        # Now 'epochs' should be your combined MNE Epochs object
        # ------------------------------------------------------
        # (1) Parse Low Frequency
        l_freq_text = bandpass_l_freq_widget.value.strip()
        if l_freq_text.lower() == "none":
            l_freq = None
        else:
            try:
                l_freq = float(l_freq_text)
                if l_freq < 0:
                    print(f"ERROR: Low frequency must be >= 0. Got: {l_freq}")
                    return
            except ValueError:
                print(f"ERROR: Could not parse low freq '{l_freq_text}' as a float or 'None'.")
                return
        
        # (2) Parse High Frequency
        h_freq_text = bandpass_h_freq_widget.value.strip()
        if h_freq_text.lower() == "none":
            h_freq = None
        else:
            try:
                h_freq = float(h_freq_text)
                if h_freq < 0:
                    print(f"ERROR: High frequency must be >= 0. Got: {h_freq}")
                    return
            except ValueError:
                print(f"ERROR: Could not parse high freq '{h_freq_text}' as float or 'None'.")
                return
        
        # (3) FIR Design
        fir_design = fir_design_widget.value
        
        # (4) Filter Length
        filter_length_text = filter_length_widget.value.strip()
        if filter_length_text.lower() == "auto":
            filter_length = 'auto'
        else:
            try:
                filter_length = int(filter_length_text)
                if filter_length % 2 == 0:
                    print(f"ERROR: Filter length must be an odd integer. Got: {filter_length}")
                    return
                if filter_length < 1:
                    print(f"ERROR: Filter length must be positive. Got: {filter_length}")
                    return
            except ValueError:
                print(f"ERROR: Could not parse filter length '{filter_length_text}' as integer or 'auto'.")
                return
        
        # (5) Validate Output .fif File
        if not output_fif_chooser.selected:
            print("ERROR: No output .fif file selected. Please choose an output file.")
            return
        
        output_fif_path = output_fif_chooser.selected
        if not (output_fif_path.endswith('.fif') or output_fif_path.endswith('.fif.gz')):
            print("ERROR: Output file must end with .fif or .fif.gz.")
            return
        
        # (6) Apply Bandpass Filter
        print(f"Applying bandpass filter with parameters:")
        print(f"  - Low Frequency: {l_freq} Hz")
        print(f"  - High Frequency: {h_freq} Hz")
        print(f"  - FIR Design: {fir_design}")
        print(f"  - Filter Length: {filter_length}")
        print(f"  - Output File: {output_fif_path}\n")
        
        try:
            epochs_filtered = epochs.copy().filter(
                l_freq=l_freq,
                h_freq=h_freq,
                fir_design=fir_design,
                filter_length=filter_length,
                verbose=False
            )
            print("Filter applied successfully.")
        except Exception as e:
            print(f"ERROR: Failed to apply filter: {e}")
            return
        
        # (7) Save Filtered Epochs
        try:
            epochs_filtered.save(output_fif_path, overwrite=True)
            print(f"Filtered epochs saved successfully to '{output_fif_path}'.")
        except Exception as e:
            print(f"ERROR: Failed to save filtered epochs: {e}")
            return

apply_filter_button.on_click(on_apply_filter_clicked)

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
    widgets.Label("### Select Output .fif File:"),
    output_fif_chooser,
    apply_filter_button,
    bandpass_output_area
])

display(bandpass_ui)
