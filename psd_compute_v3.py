import os
import pickle
import numpy as np
import mne
from scipy import signal
from tqdm.notebook import tqdm

import ipywidgets as widgets
from ipywidgets import (
    FloatText,
    FloatSlider,
    IntProgress,
    Button,
    VBox,
    HBox,
    Label,
    Output,
    SelectMultiple,
    ToggleButtons
)
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

from ipyfilechooser import FileChooser
from IPython import get_ipython

##############################################################################
# 1) WIDGETS & GLOBALS
##############################################################################

# File chooser for .fif input
input_fif_chooser = FileChooser(
    os.getcwd(),
    title='Select Filtered .fif File',
    select_default=False
)
input_fif_chooser.show_only_files = True
input_fif_chooser.filter_pattern = ['*.fif', '*.fif.gz']

# Segment Toggle: Whether to split into 6 blocks or not
segment_toggle = ToggleButtons(
    options=['No', 'Yes'],
    value='No',
    description='Segment into 6 blocks?',
    style={'description_width': 'initial'}
)

# Window length and overlap
window_length_widget = FloatText(
    value=2.0,
    description='Window (s):',
    layout=widgets.Layout(width='150px')
)
overlap_widget = FloatSlider(
    value=50.0,
    min=0.0,
    max=100.0,
    step=1.0,
    description='Overlap (%):',
    continuous_update=False,
    layout=widgets.Layout(width='300px')
)

# Output pickle chooser
output_pickle_chooser = FileChooser(
    os.getcwd(),
    title='Select Output Pickle File',
    select_default=False
)
output_pickle_chooser.show_only_files = False
output_pickle_chooser.filter_pattern = ['*.pkl']
output_pickle_chooser.default_filename = 'psd_results.pkl'

# Compute PSD button and progress
compute_psd_button = Button(
    description='Compute PSD',
    button_style='success',
    tooltip='Compute PSD on loaded data and save results'
)
psd_progress = IntProgress(
    value=0,
    min=0,
    max=100,
    step=1,
    description='Progress:',
    bar_style='info',
    orientation='horizontal',
    layout=widgets.Layout(width='400px')
)

# Output area
psd_output_area = Output()

# Global dictionary to store PSD results
psd_results = {}  # Will be { "block1": { "Ch1": {"freqs", "psd"}, ...}, "block2": ...}
# If segment_toggle=No => "block1" is the entire dataset

# Channel selection & plot mode
plot_channel_widget = SelectMultiple(
    options=[],
    description='Select Channels:',
    layout=widgets.Layout(width='300px', height='120px')
)
plot_mode_widget = ToggleButtons(
    options=[('Mean & +Std', 'mean_std'), ('Individual + Mean', 'individual_mean')],
    description='Plot Mode:',
    value='mean_std',
    button_style='info',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='300px')
)

# Axis Range
x_min_widget = FloatText(value=0.0, description='X min (Hz):', layout=widgets.Layout(width='130px'))
x_max_widget = FloatText(value=45.0, description='X max (Hz):', layout=widgets.Layout(width='130px'))
y_min_widget = FloatText(value=0.0, description='Y min:', layout=widgets.Layout(width='130px'))
y_max_widget = FloatText(value=1.0, description='Y max:', layout=widgets.Layout(width='130px'))

# Plot Button
plot_psd_button = Button(
    description='Plot PSD',
    button_style='info',
    tooltip='Plot PSD side by side for each block'
)

##############################################################################
# 2) HELPER FUNCTIONS
##############################################################################

def load_epochs_from_fif(filepath):
    """Load .fif or .fif.gz epochs and return mne.Epochs object."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File does not exist: {filepath}")
    epochs_loaded = mne.read_epochs(filepath, preload=True)
    if not isinstance(epochs_loaded, mne.Epochs):
        data_array = epochs_loaded.get_data()
        info = epochs_loaded.info
        epochs_obj = mne.EpochsArray(data_array, info)
        return epochs_obj
    return epochs_loaded

def compute_welch_psd_for_epochs(epochs, window_s=2.0, overlap_pct=50.0, fmin=0.0, fmax=45.0):
    """
    Compute PSD for all channels in a single mne.Epochs object using Welch's method.
    Returns dict {channel_name: {"freqs": <1D array>, "psd": <2D array (n_epochs, n_freqs)>}}
    """
    psd_dict = {}
    sfreq = epochs.info['sfreq']
    nperseg = int(window_s * sfreq)
    noverlap = int(nperseg * (overlap_pct / 100.0))

    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = data.shape

    if nperseg > n_times:
        raise ValueError(f"nperseg ({nperseg}) cannot exceed each epoch’s length in samples ({n_times}).")

    # Precompute window
    window_fn = signal.hamming(nperseg, sym=False)

    for ch_idx, ch_name in enumerate(epochs.ch_names):
        ch_data = data[:, ch_idx, :]  # shape => (n_epochs, n_times)

        freqs, psd_all = signal.welch(
            ch_data,
            fs=sfreq,
            window=window_fn,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='density',
            axis=1
        )
        # Filter freq range
        mask = (freqs >= fmin) & (freqs <= fmax)
        freqs_f = freqs[mask]
        psd_f = psd_all[:, mask]

        # if no freqs remain, skip
        if freqs_f.size == 0:
            psd_dict[ch_name] = {"freqs": None, "psd": None}
        else:
            psd_dict[ch_name] = {"freqs": freqs_f, "psd": psd_f}
    return psd_dict

##############################################################################
# 3) COMPUTE PSD CALLBACK
##############################################################################

def on_compute_psd_clicked(b):
    global psd_results
    psd_results = {}
    with psd_output_area:
        clear_output()
        
        # 1. Validate & load .fif
        fif_path = input_fif_chooser.selected
        if not fif_path:
            print("ERROR: No .fif file selected.")
            return
        if not os.path.isfile(fif_path):
            print(f"ERROR: File not found => {fif_path}")
            return
        
        try:
            print(f"Loading epochs from: {fif_path}")
            epochs = load_epochs_from_fif(fif_path)
            print(f"Loaded. n_epochs={len(epochs)}, n_channels={len(epochs.ch_names)}")
        except Exception as e:
            print(f"ERROR loading epochs: {e}")
            return
        
        # 2. Check segment toggle
        segment_choice = segment_toggle.value  # "Yes" or "No"
        
        # 3. Parse PSD parameters
        window_s = window_length_widget.value
        overlap_pct = overlap_widget.value
        if window_s <= 0:
            print("ERROR: Window length must be positive.")
            return
        if not (0 <= overlap_pct < 100):
            print("ERROR: Overlap must be 0-100.")
            return
        
        # 4. Output pickle file
        out_pkl = output_pickle_chooser.selected
        if not out_pkl or not out_pkl.endswith('.pkl'):
            print("ERROR: Output file must be .pkl.")
            return
        
        # 5. If segment=Yes => split into 6 blocks, each with 120 epochs => store in psd_results["block1"], etc.
        n_total_epochs = len(epochs)
        if segment_choice == "Yes":
            if n_total_epochs < 6 * 120:
                print(f"ERROR: We need at least 720 epochs, but found {n_total_epochs}. Can't split into 6 x 120.")
                return
            # We'll create 6 parted epochs
            print("Segmenting into 6 blocks of 120 epochs each.")
            for i in range(6):
                start_ep = i * 120
                end_ep = (i + 1) * 120
                block_epochs = epochs[start_ep:end_ep]  # subselect
                block_name = f"block{i+1}"
                print(f"  Computing PSD for {block_name} => epochs {start_ep}..{end_ep-1}")
                try:
                    block_psd = compute_welch_psd_for_epochs(
                        block_epochs,
                        window_s=window_s,
                        overlap_pct=overlap_pct,
                        fmin=0.0,
                        fmax=45.0
                    )
                    psd_results[block_name] = block_psd
                except Exception as e:
                    print(f"ERROR computing PSD for block {block_name}: {e}")
                    psd_results[block_name] = {}
        else:
            # No segmentation => treat entire epochs as one block
            print("No segmentation => single block with all epochs.")
            try:
                single_psd = compute_welch_psd_for_epochs(
                    epochs,
                    window_s=window_s,
                    overlap_pct=overlap_pct,
                    fmin=0.0,
                    fmax=45.0
                )
                psd_results["block1"] = single_psd
            except Exception as e:
                print(f"ERROR computing PSD for single block: {e}")
                return
        
        # 6. Check the structure of psd_results
        # e.g. { "block1": {...}, "block2": {...}, ...}
        if not psd_results:
            print("ERROR: No PSD results were computed.")
            return
        
        # 7. We gather all channels from the FIRST block to populate the channel widget
        # If user wants to see all channels that exist in every block, they'd do a cross-check, but let's keep it simple
        first_block_name = list(psd_results.keys())[0]
        block_data = psd_results[first_block_name]  # e.g. {ch: {freqs, psd}}
        # only keep valid channels
        valid_channels = [ch for ch, dd in block_data.items() if dd["freqs"] is not None]
        plot_channel_widget.options = valid_channels
        print(f"\nPopulated channel widget with {len(valid_channels)} valid channels from {first_block_name}.\n")
        
        # 8. Save PSD results to pickle
        try:
            with open(out_pkl, 'wb') as f:
                pickle.dump(psd_results, f)
            print(f"Saved PSD results to '{out_pkl}'")
        except Exception as e:
            print(f"ERROR saving PSD results: {e}")
            return
        
        # 9. Done
        print("Done computing PSD! Ready to plot.")

##############################################################################
# 4) PLOT PSD
##############################################################################

def on_plot_psd_clicked(b):
    with psd_output_area:
        clear_output()
        
        if not psd_results:
            print("ERROR: No PSD results found. Please compute PSD first.")
            return
        
        selected_channels = list(plot_channel_widget.value)
        if not selected_channels:
            print("ERROR: No channels selected for plotting.")
            return
        
        plot_mode = plot_mode_widget.value  # "mean_std" or "individual_mean"
        
        x_min = x_min_widget.value
        x_max = x_max_widget.value
        y_min = y_min_widget.value
        y_max = y_max_widget.value
        
        if x_min >= x_max or y_min >= y_max:
            print("ERROR: Invalid axis limits.")
            return
        
        # We expect psd_results = { "block1": {ch => {...}}, "block2": {...}, ...}
        # If user didn't segment => there's just "block1".
        block_names = sorted(psd_results.keys())  # e.g. ["block1"] or ["block1","block2",...,"block6"]
        
        n_blocks = len(block_names)
        n_channels = len(selected_channels)
        
        print(f"Plotting {n_channels} channel(s) across {n_blocks} block(s).")
        
        # We'll make a figure with (n_channels) rows, (n_blocks) columns
        fig, axes = plt.subplots(n_channels, n_blocks, figsize=(4*n_blocks, 2.5*n_channels), squeeze=False)
        
        for row_idx, channel in enumerate(selected_channels):
            for col_idx, block_name in enumerate(block_names):
                ax = axes[row_idx, col_idx]
                
                block_data = psd_results[block_name]  # dict {ch => {"freqs","psd"}}
                if channel not in block_data or block_data[channel]["freqs"] is None:
                    ax.text(0.5,0.5, f"No data\n{block_name}\n{channel}", va='center', ha='center')
                    ax.set_xlim(0,1)
                    ax.set_ylim(0,1)
                    continue
                
                freqs = block_data[channel]["freqs"]
                psd_array = block_data[channel]["psd"]  # shape => (n_epochs_in_block, n_freqs)
                
                psd_mean = np.nanmean(psd_array, axis=0)
                psd_std = np.nanstd(psd_array, axis=0)
                
                if plot_mode == 'mean_std':
                    ax.plot(freqs, psd_mean, color='blue', label='Mean')
                    ax.fill_between(freqs, psd_mean, psd_mean+psd_std, color='blue', alpha=0.3, label='+Std')
                else:
                    # "individual_mean"
                    for ep_idx in range(psd_array.shape[0]):
                        ax.plot(freqs, psd_array[ep_idx], color='lightgray', linewidth=0.5)
                    ax.plot(freqs, psd_mean, color='blue', label='Mean')
                
                if row_idx==0:
                    ax.set_title(block_name, fontsize=10)
                if row_idx==n_channels-1:
                    ax.set_xlabel("Freq (Hz)")
                if col_idx==0:
                    ax.set_ylabel(f"{channel}\nPSD (V²/Hz)")
                
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        plt.show()
        print("Done plotting PSD side by side.")

##############################################################################
# 5) GUI LAYOUT
##############################################################################

compute_box = VBox([
    Label("STEP 1: Select Filtered .fif"),
    input_fif_chooser,
    segment_toggle,
    HBox([window_length_widget, overlap_widget]),
    output_pickle_chooser,
    compute_psd_button,
    psd_progress,
    psd_output_area
])

plot_box = VBox([
    Label("STEP 2: Plot PSD (Side by Side if multiple blocks)"),
    HBox([Label("Channels:"), plot_channel_widget]),
    HBox([Label("Plot Mode:"), plot_mode_widget]),
    HBox([
        Label("X Range:"),
        x_min_widget,
        x_max_widget,
        Label("Y Range:"),
        y_min_widget,
        y_max_widget
    ]),
    plot_psd_button
])

main_ui = VBox([
    widgets.HTML("<h2>PSD Computation with Optional Segmentation</h2>"),
    compute_box,
    plot_box
])

# Attach callbacks
compute_psd_button.on_click(on_compute_psd_clicked)
plot_psd_button.on_click(on_plot_psd_clicked)

# Display UI
display(main_ui)
