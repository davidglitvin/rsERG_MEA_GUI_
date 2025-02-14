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
    ToggleButtons,
    Dropdown
)
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

from ipyfilechooser import FileChooser
from IPython import get_ipython

##############################################################################
# 1) WIDGETS & GLOBALS
##############################################################################

# A) Load Mode: either from .fif or from mne_data_objects
load_mode_widget = ToggleButtons(
    options=['Load from .fif', 'Use mne_data_objects'],
    value='Load from .fif',
    description='Load Mode:',
    style={'description_width': 'initial'}
)

# B) Input .fif chooser
input_fif_chooser = FileChooser(
    os.getcwd(),
    title='Select Filtered .fif File',
    select_default=False
)
input_fif_chooser.show_only_files = True
input_fif_chooser.filter_pattern = ['*.fif', '*.fif.gz']

# C) PSD Parameters
window_length_widget = FloatText(
    value=2.0,
    description='Window (s):',
    layout=widgets.Layout(width='200px'),
    tooltip='Window length in seconds for Welch'
)

overlap_widget = FloatSlider(
    value=50.0,
    min=0.0,
    max=100.0,
    step=1.0,
    description='Overlap (%):',
    continuous_update=False,
    layout=widgets.Layout(width='300px'),
    tooltip='Percentage overlap between windows'
)

# D) PSD Progress
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

# E) Compute PSD Button
compute_psd_button = Button(
    description='Compute PSD',
    button_style='success',
    tooltip='Compute PSD for either the loaded .fif or mne_data_objects'
)

# F) Output area
psd_output_area = Output()

# We'll store the final PSD data in a global variable
psd_data_objects = None

# G) Plot Controls
item_select_widget = Dropdown(
    options=[],
    description='Item/Part:',
    layout=widgets.Layout(width='250px')
)

plot_channel_widget = SelectMultiple(
    options=[],
    description='Channels:',
    layout=widgets.Layout(width='300px', height='100px')
)

plot_mode_widget = ToggleButtons(
    options=[('Mean & +Std', 'mean_std'), ('Indiv & Mean', 'individual_mean')],
    value='mean_std',
    description='Plot Mode:',
    button_style='info',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='300px')
)

x_min_widget = FloatText(value=0.0, description='X min (Hz):', layout=widgets.Layout(width='150px'))
x_max_widget = FloatText(value=45.0, description='X max (Hz):', layout=widgets.Layout(width='150px'))
y_min_widget = FloatText(value=0.0, description='Y min:', layout=widgets.Layout(width='150px'))
y_max_widget = FloatText(value=1.0, description='Y max:', layout=widgets.Layout(width='150px'))

plot_psd_button = Button(
    description='Plot PSD',
    button_style='info',
    tooltip='Plot PSD for the selected item/part & channel(s)'
)

##############################################################################
# 2) HELPER FUNCTIONS
##############################################################################

def compute_welch_psd(epochs: mne.Epochs, window_s: float, overlap_pct: float, fmin=0.0, fmax=45.0):
    """
    Compute PSD for a single mne.Epochs object using Welch's method, returning
    a dict: {channel: {"freqs": freqs, "psd": psd_array}}.
    psd_array shape => (n_epochs, n_freqs).
    """
    psd_result = {}
    sfreq = epochs.info['sfreq']

    nperseg = int(window_s * sfreq)
    noverlap = int(nperseg * (overlap_pct / 100.0))

    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = data.shape

    if nperseg > n_times:
        raise ValueError(f"nperseg ({nperseg}) cannot exceed epoch length in samples ({n_times}).")

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
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        freqs_filtered = freqs[freq_mask]
        psd_filtered = psd_all[:, freq_mask]

        if freqs_filtered.size == 0:
            psd_result[ch_name] = {"freqs": None, "psd": None}
        else:
            psd_result[ch_name] = {"freqs": freqs_filtered, "psd": psd_filtered}

    return psd_result

def load_single_fif(input_path):
    """Loads a single .fif epochs file and returns an mne.Epochs object."""
    if not input_path or not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    epochs_loaded = mne.read_epochs(input_path, preload=True)
    if not isinstance(epochs_loaded, mne.Epochs):
        # reconstruct
        data_array = epochs_loaded.get_data()
        info = epochs_loaded.info
        epochs = mne.EpochsArray(data_array, info)
        return epochs
    else:
        return epochs_loaded

##############################################################################
# 3) MAIN PSD COMPUTE CALLBACK
##############################################################################

def on_compute_psd_clicked(b):
    global psd_data_objects
    psd_data_objects = None  # reset

    with psd_output_area:
        clear_output()
        
        load_mode = load_mode_widget.value  # "Load from .fif" or "Use mne_data_objects"

        window_s = window_length_widget.value
        overlap_pct = overlap_widget.value
        if window_s <= 0:
            print("ERROR: Window length must be positive.")
            return
        if not (0 <= overlap_pct < 100):
            print("ERROR: Overlap must be between 0 and 100.")
            return
        
        # 1) If user selected "Load from .fif", we do the original approach
        if load_mode == 'Load from .fif':
            fif_path = input_fif_chooser.selected
            if not fif_path:
                print("ERROR: No .fif file selected. Please pick a file or switch to 'Use mne_data_objects'.")
                return
            try:
                print(f"Loading single .fif => {fif_path}")
                epochs_obj = load_single_fif(fif_path)
                print(f"Loaded epochs. Now computing PSD.")
                psd_result = compute_welch_psd(epochs_obj, window_s, overlap_pct, fmin=0, fmax=45)
                psd_data_objects = psd_result
                print("PSD done for single .fif. Storing in 'psd_data_objects'.")
                # Set in notebook
                get_ipython().user_ns["psd_data_objects"] = psd_data_objects
                # item_select => "Single"
                item_select_widget.options = ["Single"]
                item_select_widget.value = "Single"
                # fill channel
                valid_chs = [ch for ch, d in psd_result.items() if d["freqs"] is not None]
                plot_channel_widget.options = valid_chs
                print(f"Found {len(valid_chs)} valid channels to plot.")
            except Exception as e:
                print(f"ERROR loading or computing PSD from .fif: {e}")
            return

        # 2) If user selected "Use mne_data_objects"
        else:
            try:
                data_obj = get_ipython().user_ns["mne_data_objects"]
            except KeyError:
                print("ERROR: 'mne_data_objects' not found in the notebook. Please load data first.")
                return
            
            print("Using 'mne_data_objects' from the notebook environment.\n")
            
            def compute_for_item(item):
                if not isinstance(item, mne.Epochs):
                    print("WARNING: Attempted compute_for_item on non-Epochs object.")
                    return None
                return compute_welch_psd(item, window_s, overlap_pct, fmin=0, fmax=45)
            
            if isinstance(data_obj, mne.Epochs):
                # single
                print("Detected a single mne.Epochs in 'mne_data_objects'. Computing PSD...")
                psd_res = compute_for_item(data_obj)
                psd_data_objects = psd_res
                get_ipython().user_ns["psd_data_objects"] = psd_data_objects
                # fill
                item_select_widget.options = ["Single"]
                item_select_widget.value = "Single"
                valid_chs = [ch for ch, dd in psd_res.items() if dd["freqs"] is not None]
                plot_channel_widget.options = valid_chs
                print(f"  => Done. Found {len(valid_chs)} channels.")
                return
            
            elif isinstance(data_obj, dict):
                # parted => {part_name: epochs}
                print("Detected dict => parted data.")
                parted_dict = {}
                for part_name, part_epochs in data_obj.items():
                    if not isinstance(part_epochs, mne.Epochs):
                        parted_dict[part_name] = None
                        print(f"  WARNING: '{part_name}' not an mne.Epochs. Skipping PSD.")
                    else:
                        parted_dict[part_name] = compute_for_item(part_epochs)
                psd_data_objects = parted_dict
                get_ipython().user_ns["psd_data_objects"] = psd_data_objects
                # item_select => each key
                part_names = list(parted_dict.keys())
                item_select_widget.options = part_names
                if part_names:
                    item_select_widget.value = part_names[0]
                else:
                    item_select_widget.value = None
                print(f"  => Done parted. Found {len(part_names)} parts.")
                return
            
            elif isinstance(data_obj, list):
                # multiple items
                print(f"Detected list => {len(data_obj)} items.")
                psd_list = []
                for i, val in enumerate(data_obj):
                    if isinstance(val, mne.Epochs):
                        tmp_psd = compute_for_item(val)
                        psd_list.append(tmp_psd)
                    elif isinstance(val, dict):
                        # parted
                        tmp_dict = {}
                        for pn, ep in val.items():
                            if not isinstance(ep, mne.Epochs):
                                tmp_dict[pn] = None
                            else:
                                tmp_dict[pn] = compute_for_item(ep)
                        psd_list.append(tmp_dict)
                    else:
                        psd_list.append(None)
                        print(f"  WARNING: item {i} is not recognized => {type(val)}.")
                psd_data_objects = psd_list
                get_ipython().user_ns["psd_data_objects"] = psd_data_objects
                # Build item_select
                item_options = []
                for i, val in enumerate(psd_list):
                    if isinstance(val, dict):
                        # parted
                        for part_name in val.keys():
                            item_options.append(f"Item{i}:{part_name}")
                    elif isinstance(val, dict) or val is None:
                        continue
                    else:
                        # single
                        item_options.append(f"Item{i}:Single")
                item_select_widget.options = item_options
                if item_options:
                    item_select_widget.value = item_options[0]
                else:
                    item_select_widget.value = None
                print("  => Done list. PSD computed for each item.")
                return
            
            else:
                print(f"ERROR: 'mne_data_objects' is type {type(data_obj)} not recognized.")
                return

##############################################################################
# 4) ITEM SELECTION => POPULATE CHANNELS
##############################################################################

def on_item_select_change(change):
    """Called when user picks a new "item/part" from the dropdown."""
    if change['name'] != 'value':
        return
    new_val = change['new']
    if not new_val:
        return

    with psd_output_area:
        clear_output()
        print(f"Selected item/part: {new_val}")
        
        try:
            psd_obj = get_ipython().user_ns["psd_data_objects"]
        except KeyError:
            print("ERROR: 'psd_data_objects' not found. Please compute PSD first.")
            return
        
        # We'll define a helper to fetch the PSD dict
        def fetch_psd_dict(root_obj, label):
            # If root_obj is a single dictionary of channels => user picks "Single"
            #   we return the entire dictionary
            if isinstance(root_obj, dict) and not any(isinstance(v, dict) for v in root_obj.values()):
                # single parted or single
                # if label is in root_obj, that means user typed an actual channel name
                # but we want the entire dictionary if label == 'Single'
                if label in root_obj:
                    return root_obj[label]
                else:
                    # 'Single' or something else => return entire dict
                    return root_obj
            
            # parted => {partA: {ch:...}, partB: {ch:...}}
            elif isinstance(root_obj, dict):
                if label in root_obj:
                    return root_obj[label]
                else:
                    return None
            
            # If we have a list
            elif isinstance(root_obj, list):
                # parse itemX:partName
                if ":" in label:
                    item_str, part_str = label.split(":", 1)
                    idx = int(item_str.replace("Item", ""))
                    if idx < 0 or idx >= len(root_obj):
                        return None
                    item_val = root_obj[idx]
                    if isinstance(item_val, dict):
                        # parted
                        if part_str in item_val:
                            return item_val[part_str]
                        elif part_str == "Single":
                            return item_val
                        else:
                            return None
                    else:
                        # single
                        if part_str == "Single":
                            return item_val
                        else:
                            return None
                else:
                    # might be single
                    return None
            else:
                # fallback => might be single parted
                return None
        
        found_psd = fetch_psd_dict(psd_obj, new_val)
        if not found_psd or not isinstance(found_psd, dict):
            print(f"ERROR: PSD dict not found for '{new_val}'.")
            plot_channel_widget.options = []
            return
        
        # found_psd => {channel: {"freqs":..., "psd":...}}
        valid_chs = [ch for ch, d in found_psd.items() if d and d["freqs"] is not None]
        plot_channel_widget.options = valid_chs
        print(f"Channels populated => {len(valid_chs)} found.")


##############################################################################
# 5) PLOT PSD
##############################################################################

def on_plot_psd_clicked(b):
    with psd_output_area:
        clear_output()

        item_val = item_select_widget.value
        channels = plot_channel_widget.value
        if not item_val:
            print("ERROR: No item/part selected.")
            return
        if not channels:
            print("ERROR: No channels selected.")
            return
        
        try:
            psd_obj = get_ipython().user_ns["psd_data_objects"]
        except KeyError:
            print("ERROR: 'psd_data_objects' not found. Please compute PSD first.")
            return
        
        # Helper to retrieve the PSD for the chosen item/part
        def fetch_psd_dict(root_obj, label):
            # If single parted or single
            if isinstance(root_obj, dict) and not any(isinstance(v, dict) for v in root_obj.values()):
                # If label is in root_obj => user typed a channel name
                # else => might be "Single" => return the entire dict
                if label in root_obj:
                    return root_obj[label]
                else:
                    return root_obj
            
            # parted => {partA: {ch:...}, partB: {ch:...}}
            elif isinstance(root_obj, dict):
                return root_obj.get(label, None)
            
            # list => multiple items
            elif isinstance(root_obj, list):
                if ":" not in label:
                    return None
                item_str, part_str = label.split(":", 1)
                idx = int(item_str.replace("Item",""))
                if idx < 0 or idx >= len(root_obj):
                    return None
                val_ = root_obj[idx]
                if isinstance(val_, dict):
                    # parted
                    if part_str in val_:
                        return val_[part_str]
                    elif part_str == "Single":
                        return val_
                    else:
                        return None
                else:
                    # single
                    if part_str == "Single":
                        return val_
                    else:
                        return None
            else:
                return None
        
        found_psd = fetch_psd_dict(psd_obj, item_val)
        if not found_psd or not isinstance(found_psd, dict):
            print(f"ERROR: PSD data not found for '{item_val}'.")
            return
        
        x_min, x_max = x_min_widget.value, x_max_widget.value
        y_min, y_max = y_min_widget.value, y_max_widget.value
        if x_min >= x_max or y_min >= y_max:
            print("ERROR: Invalid axis limits.")
            return
        
        mode = plot_mode_widget.value
        
        for ch in channels:
            if ch not in found_psd or found_psd[ch]["freqs"] is None:
                print(f"Skipping channel '{ch}' - no valid PSD.")
                continue
            
            freqs = found_psd[ch]["freqs"]
            psd_data = found_psd[ch]["psd"]  # shape => (n_epochs, n_freqs)
            
            psd_mean = np.nanmean(psd_data, axis=0)
            psd_std = np.nanstd(psd_data, axis=0)
            
            plt.figure(figsize=(8,5))
            
            if mode == 'mean_std':
                plt.plot(freqs, psd_mean, color='blue', label=f"{ch} Mean PSD")
                plt.fill_between(freqs, psd_mean, psd_mean + psd_std, color='blue', alpha=0.3, label='+Std')
            elif mode == 'individual_mean':
                for i in range(psd_data.shape[0]):
                    plt.plot(freqs, psd_data[i], color='lightgray', linewidth=0.5)
                plt.plot(freqs, psd_mean, color='blue', label=f"{ch} Mean PSD")
            else:
                print(f"ERROR: unknown plot mode '{mode}'")
                plt.close()
                continue
            
            plt.title(f"PSD: {item_val}, {ch}")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("PSD (V²/Hz)")
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.show()


##############################################################################
# 6) EVENT CONNECTIONS
##############################################################################

compute_psd_button.on_click(on_compute_psd_clicked)
item_select_widget.observe(on_item_select_change, names='value')
plot_psd_button.on_click(on_plot_psd_clicked)

##############################################################################
# 7) UI LAYOUT
##############################################################################

load_box = VBox([
    Label("Load Mode:"),
    load_mode_widget,
    Label("If loading a single .fif:"),
    input_fif_chooser
])

psd_params_box = HBox([window_length_widget, overlap_widget])
axis_box = HBox([x_min_widget, x_max_widget, y_min_widget, y_max_widget])

psd_ui = VBox([
    widgets.HTML("<h2>PSD Computation (Multi-Part or Single .fif)</h2>"),
    load_box,
    Label("Step 2: PSD Parameters:"),
    psd_params_box,
    compute_psd_button,
    psd_progress,
    psd_output_area,
    
    widgets.HTML("<h3>Plot PSD</h3>"),
    HBox([Label("Item/Part:"), item_select_widget]),
    HBox([Label("Channels:"), plot_channel_widget]),
    HBox([Label("Plot Mode:"), plot_mode_widget]),
    HBox([Label("Axis Limits:"), axis_box]),
    plot_psd_button
])

# Display the GUI if running standalone
display(psd_ui)
