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
# 1) GUI WIDGETS & GLOBALS
##############################################################################

# File chooser for .fif
input_fif_chooser = FileChooser(
    os.getcwd(),
    title='Select Filtered .fif File',
    select_default=False
)
input_fif_chooser.show_only_files = True
input_fif_chooser.filter_pattern = ['*.fif', '*.fif.gz']

# Segment toggle (Yes=split into 6 blocks, No=all epochs)
segment_toggle = ToggleButtons(
    options=['No', 'Yes'],
    value='No',
    description='Segment 6 blocks?',
    style={'description_width': 'initial'}
)

# PSD cleaning toggle
psd_clean_toggle = ToggleButtons(
    options=['No', 'Yes'],
    value='No',
    description='Apply PSD Dropping?',
    style={'description_width': 'initial'}
)

# Example threshold widgets for cleaning
low_band_threshold_widget = FloatText(
    value=3.0,
    description='LowBandTh:',
    layout=widgets.Layout(width='120px'),
    tooltip='Threshold for low-frequency outlier'
)
test_band_threshold_widget = FloatText(
    value=10.0,
    description='TestBandTh:',
    layout=widgets.Layout(width='120px'),
    tooltip='Threshold for band outlier checks'
)

# PSD Window length & overlap
window_length_widget = FloatText(
    value=2.0,
    description='Window (s):',
    layout=widgets.Layout(width='120px'),
    tooltip='Welch window length'
)
overlap_widget = FloatSlider(
    value=50.0,
    min=0.0,
    max=100.0,
    step=1.0,
    description='Overlap(%):',
    continuous_update=False,
    layout=widgets.Layout(width='250px')
)

# Output pickle
output_pickle_chooser = FileChooser(
    os.getcwd(),
    title='Select Output Pickle File',
    select_default=False
)
output_pickle_chooser.show_only_files = False
output_pickle_chooser.filter_pattern = ['*.pkl']
output_pickle_chooser.default_filename = 'psd_results.pkl'

# Compute PSD button & progress
compute_psd_button = Button(
    description='Compute PSD',
    button_style='success',
    tooltip='Compute PSD on loaded data, optionally split into blocks, drop bad epochs, and save'
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

# Global to store final PSD results: { "block1": {ch => {...}}, "block2": {...}, ...}
psd_results = {}

# Channel selection & plot mode
plot_channel_widget = SelectMultiple(
    options=[],
    description='Channels:',
    layout=widgets.Layout(width='300px', height='120px')
)
plot_mode_widget = ToggleButtons(
    options=[('Mean & +Std', 'mean_std'), ('Individual + Mean', 'individual_mean')],
    description='Plot Mode:',
    value='mean_std',
    button_style='info',
    layout=widgets.Layout(width='250px')
)

# Axis range
x_min_widget = FloatText(value=0.0, description='X min:', layout=widgets.Layout(width='100px'))
x_max_widget = FloatText(value=45.0, description='X max:', layout=widgets.Layout(width='100px'))
y_min_widget = FloatText(value=0.0, description='Y min:', layout=widgets.Layout(width='100px'))
y_max_widget = FloatText(value=1.0, description='Y max:', layout=widgets.Layout(width='100px'))

# Plot button
plot_psd_button = Button(
    description='Plot PSD',
    button_style='info',
    tooltip='Plot PSD side by side for each block'
)

##############################################################################
# 2) HELPER FUNCTIONS
##############################################################################

def load_epochs_from_fif(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    epochs_loaded = mne.read_epochs(filepath, preload=True)
    if not isinstance(epochs_loaded, mne.Epochs):
        data_array = epochs_loaded.get_data()
        info = epochs_loaded.info
        new_ep = mne.EpochsArray(data_array, info)
        return new_ep
    return epochs_loaded

def compute_welch_psd_for_epochs(
    epochs: mne.Epochs, window_s=2.0, overlap_pct=50.0, fmin=0.0, fmax=45.0
):
    """
    Compute PSD for each channel of an mne.Epochs using Welch.
    Returns dict {ch: {"freqs":<1D>, "psd":<2D [n_epochs, n_freqs]>}}.
    """
    psd_dict = {}
    sfreq = epochs.info['sfreq']
    nperseg = int(window_s * sfreq)
    noverlap = int(nperseg * (overlap_pct / 100.0))

    data = epochs.get_data()  # shape=(n_ep, n_ch, n_time)
    n_ep, n_ch, n_time = data.shape

    if nperseg > n_time:
        raise ValueError(f"nperseg ({nperseg}) > epoch length in samples ({n_time}).")

    window_fn = signal.hamming(nperseg, sym=False)

    for ch_idx, ch_name in enumerate(epochs.ch_names):
        ch_data = data[:, ch_idx, :]  # (n_ep, n_time)

        freqs, psd_all = signal.welch(
            ch_data,
            fs=sfreq,
            window=window_fn,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='density',
            axis=1
        )
        mask = (freqs >= fmin) & (freqs <= fmax)
        freqs_f = freqs[mask]
        psd_f = psd_all[:, mask]

        if freqs_f.size == 0:
            psd_dict[ch_name] = {"freqs": None, "psd": None}
        else:
            psd_dict[ch_name] = {"freqs": freqs_f, "psd": psd_f}
    return psd_dict

def exclude_psd_epochs(psd_dict_block, low_band=(1,3), low_th=3.0,
                       test_bands=[(7,9),(9,11),(11,13),(13,15),(15,17),(17,19),(19,21)],
                       test_th=10.0, test_count_th=None):
    """
    Example function to drop "bad" epochs from each channel’s PSD array, in a single block.
    We remove rows from psd_f that fail checks:
      - any row with PSD in low_band > low_th * mean
      - row with repeated test_band outliers
    Returns a new psd_dict with some epochs removed for each channel.
    """
    # if test_count_th not specified, default to half the test bands
    if test_count_th is None:
        test_count_th = len(test_bands) // 2

    # We'll do everything in freq space once for all channels
    # But each channel can have a different psd => we define a function to exclude in arrays
    def exclude_traces(psd_array, freqs):
        mean_psd = np.nanmean(psd_array, axis=0)
        low_mask = (freqs >= low_band[0]) & (freqs <= low_band[1])
        band_masks = [(freqs >= b[0]) & (freqs <= b[1]) for b in test_bands]

        keep_idx = []
        for ep_idx, row in enumerate(psd_array):
            # 1) check low band outlier
            if np.any(row[low_mask] > low_th * mean_psd[low_mask]):
                continue
            # 2) check repeated band outliers
            count_test = 0
            for bm in band_masks:
                if np.any(row[bm] > test_th * mean_psd[bm]):
                    count_test += 1
            if count_test >= test_count_th:
                continue
            # If not excluded
            keep_idx.append(ep_idx)
        return keep_idx

    new_block_dict = {}
    for ch, dd in psd_dict_block.items():
        if dd["freqs"] is None or dd["psd"] is None:
            new_block_dict[ch] = dd
            continue
        freqs_ = dd["freqs"]
        psd_ = dd["psd"]  # shape => (n_epochs, n_freqs)
        keep_idx = exclude_traces(psd_, freqs_)
        if len(keep_idx) == 0:
            new_block_dict[ch] = {"freqs": freqs_, "psd": None}
        else:
            new_block_dict[ch] = {"freqs": freqs_, "psd": psd_[keep_idx,:]}
    return new_block_dict


##############################################################################
# 3) MAIN LOGIC
##############################################################################

def on_compute_psd_clicked(b):
    global psd_results
    psd_results = {}
    with psd_output_area:
        clear_output()

        fif_path = input_fif_chooser.selected
        if not fif_path:
            print("ERROR: No .fif file selected.")
            return
        if not os.path.isfile(fif_path):
            print(f"ERROR: File not found => {fif_path}")
            return
        
        try:
            epochs_all = load_epochs_from_fif(fif_path)
            n_epochs = len(epochs_all)
            print(f"Loaded => {n_epochs} epochs, {len(epochs_all.ch_names)} channels.")
        except Exception as e:
            print(f"ERROR loading epochs: {e}")
            return

        do_segment = (segment_toggle.value == 'Yes')
        do_clean = (psd_clean_toggle.value == 'Yes')

        # Welch param
        w_len = window_length_widget.value
        overlap = overlap_widget.value

        # Validate
        if w_len <= 0:
            print("ERROR: Window length must be > 0.")
            return
        if not (0 <= overlap < 100):
            print("ERROR: Overlap must be 0-100.")
            return

        out_path = output_pickle_chooser.selected
        if not out_path or not out_path.endswith('.pkl'):
            print("ERROR: Output must be .pkl.")
            return

        # Segment if chosen
        if do_segment:
            # We'll assume user wants 6 blocks => 120 each if 720 total
            print("Segmenting into 6 blocks of 120 epochs each.")
            if n_epochs < 720:
                print(f"WARNING: We only have {n_epochs} epochs, not enough for 6 full blocks (720). Using partial blocks.")
            block_size = 120
            n_blocks = (n_epochs // block_size)  # might be 6 if exactly 720, or fewer
            for i in range(n_blocks):
                start_ep = i * block_size
                end_ep = (i+1)*block_size
                print(f"  Block{i+1}: epochs {start_ep}..{end_ep-1}")
                block_ep = epochs_all[start_ep:end_ep]
                block_name = f"block{i+1}"
                # Compute PSD
                block_psd = compute_welch_psd_for_epochs(
                    block_ep, window_s=w_len, overlap_pct=overlap, fmin=0.0, fmax=45.0
                )
                # optional cleaning
                if do_clean:
                    print(f"  PSD dropping for block {block_name}")
                    block_psd = exclude_psd_epochs(
                        block_psd,
                        low_band=(1,3),
                        low_th=low_band_threshold_widget.value,
                        test_bands=[(7,9),(9,11),(11,13),(13,15),(15,17),(17,19),(19,21)],
                        test_th=test_band_threshold_widget.value,
                        test_count_th=None
                    )
                psd_results[block_name] = block_psd
            if n_blocks == 0:
                print("ERROR: No blocks created. Possibly not enough epochs.")
                return
        else:
            # Single block
            print("No segmentation => single block of all epochs.")
            block_psd = compute_welch_psd_for_epochs(
                epochs_all,
                window_s=w_len,
                overlap_pct=overlap,
                fmin=0.0,
                fmax=45.0
            )
            if do_clean:
                print("Applying PSD dropping to single block.")
                block_psd = exclude_psd_epochs(
                    block_psd,
                    low_band=(1,3),
                    low_th=low_band_threshold_widget.value,
                    test_bands=[(7,9),(9,11),(11,13),(13,15),(15,17),(17,19),(19,21)],
                    test_th=test_band_threshold_widget.value,
                    test_count_th=None
                )
            psd_results["block1"] = block_psd

        # For channel selection, we use first block
        first_block = list(psd_results.keys())[0]
        block_data = psd_results[first_block]  # dict of channels
        # gather valid channels
        valid_chs = [ch for ch, dd in block_data.items() if dd["psd"] is not None and dd["freqs"] is not None]
        plot_channel_widget.options = valid_chs

        # Save to pickle
        try:
            with open(out_path, 'wb') as f:
                pickle.dump(psd_results, f)
            print(f"Saved PSD data to '{out_path}'.")
        except Exception as e:
            print(f"ERROR saving PSD data: {e}")
            return

        print("Done. PSD computed (and possibly cleaned). You may now plot.")

##############################################################################
# 4) PLOT LOGIC
##############################################################################

def on_plot_psd_clicked(b):
    with psd_output_area:
        clear_output()
        if not psd_results:
            print("ERROR: No PSD results. Please compute PSD first.")
            return
        
        sel_channels = list(plot_channel_widget.value)
        if not sel_channels:
            print("ERROR: No channels selected.")
            return
        mode = plot_mode_widget.value
        x_min, x_max = x_min_widget.value, x_max_widget.value
        y_min, y_max = y_min_widget.value, y_max_widget.value
        if x_min >= x_max or y_min >= y_max:
            print("ERROR: Invalid axis range.")
            return

        block_names = sorted(psd_results.keys())
        n_blocks = len(block_names)
        n_ch = len(sel_channels)
        print(f"Plotting {n_ch} channels x {n_blocks} blocks.")
        
        fig, axes = plt.subplots(n_ch, n_blocks, figsize=(4*n_blocks, 2.5*n_ch), squeeze=False)
        
        for r_idx, ch in enumerate(sel_channels):
            for c_idx, blk in enumerate(block_names):
                ax = axes[r_idx, c_idx]
                block_dict = psd_results[blk]
                if ch not in block_dict or block_dict[ch]["freqs"] is None or block_dict[ch]["psd"] is None:
                    ax.text(0.5,0.5,f"No data\n{blk}:{ch}",ha='center',va='center')
                    ax.set_xlim(0,1)
                    ax.set_ylim(0,1)
                    continue
                freqs = block_dict[ch]["freqs"]
                psd_arr = block_dict[ch]["psd"]  # shape => (#kept_epochs, freq_len)
                psd_mean = np.nanmean(psd_arr, axis=0)
                psd_std = np.nanstd(psd_arr, axis=0)
                
                if mode == 'mean_std':
                    ax.plot(freqs, psd_mean, color='blue')
                    ax.fill_between(freqs, psd_mean, psd_mean+psd_std, color='blue', alpha=0.3)
                else:
                    # "individual + mean"
                    for i in range(psd_arr.shape[0]):
                        ax.plot(freqs, psd_arr[i], color='lightgray', linewidth=0.5)
                    ax.plot(freqs, psd_mean, color='blue')
                
                if r_idx == 0:
                    ax.set_title(blk, fontsize=9)
                if r_idx == n_ch - 1:
                    ax.set_xlabel("Freq (Hz)")
                if c_idx == 0:
                    ax.set_ylabel(f"{ch}\nPSD (V²/Hz)")
                
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        plt.show()
        print("Done plotting PSD side by side.")

##############################################################################
# 5) GUI LAYOUT
##############################################################################

# Row for PSD cleaning thresholds
clean_params = HBox([
    psd_clean_toggle,
    Label("LowBandTh:"),
    low_band_threshold_widget,
    Label("TestBandTh:"),
    test_band_threshold_widget
])

param_box = VBox([
    HBox([segment_toggle, clean_params]),
    HBox([window_length_widget, overlap_widget]),
])

compute_box = VBox([
    Label("STEP A: Load Filtered .fif File:"),
    input_fif_chooser,
    param_box,
    output_pickle_chooser,
    compute_psd_button,
    psd_progress,
    psd_output_area
])

plot_box = VBox([
    Label("STEP B: Plot PSD"),
    HBox([Label("Channels:"), plot_channel_widget]),
    HBox([Label("Plot Mode:"), plot_mode_widget]),
    HBox([
        Label("X Range:"), x_min_widget, x_max_widget,
        Label("Y Range:"), y_min_widget, y_max_widget
    ]),
    plot_psd_button
])

main_ui = VBox([
    widgets.HTML("<h2>PSD Computation with Optional Segmentation & Cleaning</h2>"),
    compute_box,
    plot_box
])

# Attach callbacks
compute_psd_button.on_click(on_compute_psd_clicked)
plot_psd_button.on_click(on_plot_psd_clicked)

# Display
display(main_ui)
