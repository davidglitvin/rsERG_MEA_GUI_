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
    Text
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
            num_blocks = 6
            epochs_per_block = 120
            required_epochs = num_blocks * epochs_per_block
            if n_total_epochs < required_epochs:
                print(f"ERROR: Need at least {required_epochs} epochs for segmentation, but found {n_total_epochs}.")
                return
            print(f"Segmenting into {num_blocks} blocks of {epochs_per_block} epochs each.")
            for i in range(num_blocks):
                start_ep = i * epochs_per_block
                end_ep = (i + 1) * epochs_per_block if i < num_blocks -1 else n_total_epochs
                block_epochs = epochs[start_ep:end_ep]  # subselect
                block_name = f"block{i+1}"
                print(f"  Computing PSD for {block_name} => epochs {start_ep} to {end_ep-1}")
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
                    print(f"ERROR computing PSD for {block_name}: {e}")
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
        
        # 7. Populate the channel selection widget
        # If segmented, keys will be "blockX:ChY", else "block1:ChY" or "ChY"
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
# 4) PLOT PSD CALLBACK
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
        
        # Axis Ranges
        x_min = x_min_widget.value
        x_max = x_max_widget.value
        y_min = y_min_widget.value
        y_max = y_max_widget.value
        
        # Validate axis ranges
        if (x_min is not None and x_max is not None) and (x_min >= x_max):
            print("ERROR: X min must be less than X max.")
            return
        if (y_min is not None and y_max is not None) and (y_min >= y_max):
            print("ERROR: Y min must be less than Y max.")
            return
        
        # PSD Parameters
        # These can be added as needed for further customization
        
        # Test bands (can be made into widgets if needed)
        # Currently, using fixed test bands from the original exclude_traces function
        test_bands_list = [(7,9),(9,11),(11,13),(13,15),(15,17),(17,19),(19,21)]
        
        # Exclusion criteria can also be adjusted here if needed
        
        # Extract psd_data for selected channels
        # If segmented, keys might be "blockX:ChY"
        # We'll need to parse the block and channel names
        # For simplicity, assume all blocks have the same channels
        
        # Determine if data is segmented based on block names
        # If block names contain "block" and channels are unique per block, then segmented
        # Else, assume single block
        is_segmented = any('block' in blk.lower() for blk in psd_results.keys())
        
        if is_segmented:
            # Collect all unique blocks and channels
            blocks = sorted(psd_results.keys())
            channels = sorted(selected_channels)
            n_blocks = len(blocks)
            n_channels = len(channels)
            
            print(f"Plotting {n_channels} channel(s) across {n_blocks} block(s).")
            
            # Create subplot grid: channels as rows, blocks as columns
            fig, axes = plt.subplots(n_channels, n_blocks, figsize=(4*n_blocks, 3*n_channels), squeeze=False)
            plt.subplots_adjust(hspace=0.5, wspace=0.4)
            
            for row_idx, ch in enumerate(channels):
                for col_idx, blk in enumerate(blocks):
                    ax = axes[row_idx, col_idx]
                    psd_info = psd_results[blk].get(ch, {})
                    freqs = psd_info.get('freqs')
                    psd = psd_info.get('psd')
                    
                    if psd is None or freqs is None:
                        ax.text(0.5, 0.5, f"No PSD data\n{blk}:{ch}", ha='center', va='center')
                        ax.axis('off')
                        continue
                    
                    # Compute mean and std
                    psd_mean = np.nanmean(psd, axis=0)
                    psd_std = np.nanstd(psd, axis=0)
                    
                    if plot_mode == 'mean_std':
                        ax.plot(freqs, psd_mean, color='blue', label='Mean')
                        ax.fill_between(freqs, psd_mean - psd_std, psd_mean + psd_std, color='blue', alpha=0.3, label='±Std')
                    else:
                        # Plot individual PSDs
                        for ep_idx in range(psd.shape[0]):
                            ax.plot(freqs, psd[ep_idx], color='lightgray', linewidth=0.5)
                        ax.plot(freqs, psd_mean, color='blue', label='Mean')
                    
                    # Set titles and labels
                    if row_idx == 0:
                        ax.set_title(blk, fontsize=10)
                    if col_idx == 0:
                        ax.set_ylabel(ch, fontsize=10)
                    
                    # Set axis ranges
                    if x_min is not None and x_max is not None:
                        ax.set_xlim(x_min, x_max)
                    if y_min is not None and y_max is not None:
                        ax.set_ylim(y_min, y_max)
                    
                    # Add legend only to first subplot to avoid clutter
                    if row_idx == 0 and col_idx == 0:
                        ax.legend(loc='upper right', fontsize=8)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.suptitle("PSD Plots: Channels vs Blocks", fontsize=12)
            plt.show()
            print("Done plotting PSDs.")
        else:
            # Single block plotting
            block = list(psd_results.keys())[0]
            print(f"Plotting channels from single block: {block}")
            
            n_channels = len(selected_channels)
            fig, axes = plt.subplots(n_channels, 1, figsize=(6, 3*n_channels), squeeze=False)
            plt.subplots_adjust(hspace=0.5)
            
            for row_idx, ch in enumerate(selected_channels):
                ax = axes[row_idx, 0]
                psd_info = psd_results[block].get(ch, {})
                freqs = psd_info.get('freqs')
                psd = psd_info.get('psd')
                
                if psd is None or freqs is None:
                    ax.text(0.5, 0.5, f"No PSD data\n{block}:{ch}", ha='center', va='center')
                    ax.axis('off')
                    continue
                
                # Compute mean and std
                psd_mean = np.nanmean(psd, axis=0)
                psd_std = np.nanstd(psd, axis=0)
                
                if plot_mode == 'mean_std':
                    ax.plot(freqs, psd_mean, color='blue', label='Mean')
                    ax.fill_between(freqs, psd_mean - psd_std, psd_mean + psd_std, color='blue', alpha=0.3, label='±Std')
                else:
                    # Plot individual PSDs
                    for ep_idx in range(psd.shape[0]):
                        ax.plot(freqs, psd[ep_idx], color='lightgray', linewidth=0.5)
                    ax.plot(freqs, psd_mean, color='blue', label='Mean')
                
                # Set titles and labels
                ax.set_title(ch, fontsize=10)
                ax.set_xlabel("Frequency (Hz)", fontsize=10)
                ax.set_ylabel("PSD (V²/Hz)", fontsize=10)
                
                # Set axis ranges
                if x_min is not None and x_max is not None:
                    ax.set_xlim(x_min, x_max)
                if y_min is not None and y_max is not None:
                    ax.set_ylim(y_min, y_max)
                
                # Add legend only to first subplot to avoid clutter
                if row_idx == 0:
                    ax.legend(loc='upper right', fontsize=8)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.suptitle(f"PSD Plots: Channels from {block}", fontsize=12)
            plt.show()
            print("Done plotting PSDs.")

##############################################################################
# 5) EXPORT PSD PLOTS AS SVG CALLBACK
##############################################################################

def on_export_svg_clicked(b):
    with psd_output_area:
        clear_output()
        
        if not psd_results:
            print("ERROR: No PSD results found. Please compute PSD first.")
            return
        
        # Collect the plots that have been generated
        # In this code, plots are displayed but not stored.
        # To export, we need to generate the plots again and save them.
        # Alternatively, modify the plotting function to store figures.
        # Here, we'll assume that we have to regenerate the plots for exporting.
        
        # However, to make it efficient, we can modify the plotting function to store figures.
        # For simplicity, let's re-use the plotting code to save the figures as SVG.
        
        # Prompt the user to select an export directory
        export_dir_chooser = FileChooser(
            os.getcwd(),
            title='Select Export Directory for SVG Plots',
            select_default=False
        )
        export_dir_chooser.show_only_dirs = True
        display(export_dir_chooser)
        
        # Wait for user to select directory
        import time
        time.sleep(1)  # Allow some time for user to select
        
        export_dir = export_dir_chooser.selected
        if not export_dir:
            print("No export directory selected.")
            return
        if not os.path.isdir(export_dir):
            print(f"Selected path is not a directory: {export_dir}")
            return
        
        # Ask for a base filename
        base_filename = widgets.Text(
            value='psd_plot',
            description='Base Filename:',
            layout=widgets.Layout(width='300px')
        )
        display(base_filename)
        
        # Wait for user to input base filename
        def save_svg_plots(change):
            fname_base = base_filename.value.strip()
            if not fname_base:
                print("No base filename provided.")
                return
            
            # Check if data is segmented
            is_segmented = any('block' in blk.lower() for blk in psd_results.keys())
            
            if is_segmented:
                blocks = sorted(psd_results.keys())
                channels = sorted([ch for blk in blocks for ch in psd_results[blk] if psd_results[blk][ch]["freqs"] is not None])
                n_blocks = len(blocks)
                n_channels = len(channels)
                
                # Create a figure with subplots
                fig, axes = plt.subplots(n_channels, n_blocks, figsize=(4*n_blocks, 3*n_channels), squeeze=False)
                plt.subplots_adjust(hspace=0.5, wspace=0.4)
                
                for row_idx, ch in enumerate(channels):
                    for col_idx, blk in enumerate(blocks):
                        ax = axes[row_idx, col_idx]
                        psd_info = psd_results[blk].get(ch, {})
                        freqs = psd_info.get('freqs')
                        psd = psd_info.get('psd')
                        
                        if psd is None or freqs is None:
                            ax.text(0.5, 0.5, f"No PSD data\n{blk}:{ch}", ha='center', va='center')
                            ax.axis('off')
                            continue
                        
                        # Compute mean and std
                        psd_mean = np.nanmean(psd, axis=0)
                        psd_std = np.nanstd(psd, axis=0)
                        
                        ax.plot(freqs, psd_mean, color='blue', label='Mean')
                        ax.fill_between(freqs, psd_mean - psd_std, psd_mean + psd_std, color='blue', alpha=0.3, label='±Std')
                        
                        # Set titles and labels
                        if row_idx == 0:
                            ax.set_title(blk, fontsize=10)
                        if col_idx == 0:
                            ax.set_ylabel(ch, fontsize=10)
                        
                        # Set axis ranges
                        if x_min_widget.value is not None and x_max_widget.value is not None:
                            ax.set_xlim(x_min_widget.value, x_max_widget.value)
                        if y_min_widget.value is not None and y_max_widget.value is not None:
                            ax.set_ylim(y_min_widget.value, y_max_widget.value)
                        
                        # Add legend only to first subplot to avoid clutter
                        if row_idx == 0 and col_idx == 0:
                            ax.legend(loc='upper right', fontsize=8)
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.suptitle("PSD Plots: Channels vs Blocks", fontsize=12)
                
                # Save the figure as SVG
                svg_filename = os.path.join(export_dir, f"{fname_base}.svg")
                try:
                    fig.savefig(svg_filename, format='svg')
                    print(f"Saved SVG plot to '{svg_filename}'")
                except Exception as e:
                    print(f"ERROR saving SVG plot: {e}")
                plt.close(fig)
            else:
                # Single block plotting
                block = list(psd_results.keys())[0]
                print(f"Exporting plots for single block: {block}")
                
                channels = sorted(selected_channels)
                n_channels = len(channels)
                
                fig, axes = plt.subplots(n_channels, 1, figsize=(6, 3*n_channels), squeeze=False)
                plt.subplots_adjust(hspace=0.5)
                
                for row_idx, ch in enumerate(channels):
                    ax = axes[row_idx, 0]
                    psd_info = psd_results[block].get(ch, {})
                    freqs = psd_info.get('freqs')
                    psd = psd_info.get('psd')
                    
                    if psd is None or freqs is None:
                        ax.text(0.5, 0.5, f"No PSD data\n{block}:{ch}", ha='center', va='center')
                        ax.axis('off')
                        continue
                    
                    # Compute mean and std
                    psd_mean = np.nanmean(psd, axis=0)
                    psd_std = np.nanstd(psd, axis=0)
                    
                    ax.plot(freqs, psd_mean, color='blue', label='Mean')
                    ax.fill_between(freqs, psd_mean - psd_std, psd_mean + psd_std, color='blue', alpha=0.3, label='±Std')
                    
                    # Set titles and labels
                    ax.set_title(ch, fontsize=10)
                    ax.set_xlabel("Frequency (Hz)", fontsize=10)
                    ax.set_ylabel("PSD (V²/Hz)", fontsize=10)
                    
                    # Set axis ranges
                    if x_min_widget.value is not None and x_max_widget.value is not None:
                        ax.set_xlim(x_min_widget.value, x_max_widget.value)
                    if y_min_widget.value is not None and y_max_widget.value is not None:
                        ax.set_ylim(y_min_widget.value, y_max_widget.value)
                    
                    # Add legend only to first subplot to avoid clutter
                    if row_idx == 0:
                        ax.legend(loc='upper right', fontsize=8)
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.suptitle(f"PSD Plots: Channels from {block}", fontsize=12)
                
                # Save the figure as SVG
                svg_filename = os.path.join(export_dir, f"{fname_base}.svg")
                try:
                    fig.savefig(svg_filename, format='svg')
                    print(f"Saved SVG plot to '{svg_filename}'")
                except Exception as e:
                    print(f"ERROR saving SVG plot: {e}")
                plt.close(fig)
        
        # Link the save function to the base_filename widget
        base_filename.observe(save_svg_plots, names='value')
    
    # 6) EXPORT SVG BUTTON CALLBACK
    def on_export_svg_clicked(b):
        with psd_output_area:
            clear_output()
            print("Please select an export directory and provide a base filename.")
            # Trigger the export process by displaying the necessary widgets
            # Create a new set of widgets for export
            export_dir_chooser = FileChooser(
                os.getcwd(),
                title='Select Export Directory for SVG Plots',
                select_default=False
            )
            export_dir_chooser.show_only_dirs = True
            display(export_dir_chooser)
            
            base_filename = widgets.Text(
                value='psd_plot',
                description='Base Filename:',
                layout=widgets.Layout(width='300px')
            )
            display(base_filename)
            
            save_button = Button(
                description='Save SVG Plots',
                button_style='success',
                tooltip='Save the generated PSD plots as SVG files'
            )
            display(save_button)
            
            def save_svg_plots(b_save):
                export_dir = export_dir_chooser.selected
                if not export_dir:
                    print("No export directory selected.")
                    return
                if not os.path.isdir(export_dir):
                    print(f"Selected path is not a directory: {export_dir}")
                    return
                
                fname_base = base_filename.value.strip()
                if not fname_base:
                    print("No base filename provided.")
                    return
                
                # Determine if data is segmented
                is_segmented = any('block' in blk.lower() for blk in psd_results.keys())
                
                if is_segmented:
                    blocks = sorted(psd_results.keys())
                    channels = sorted([ch for blk in blocks for ch in psd_results[blk] if psd_results[blk][ch]["freqs"] is not None])
                    n_blocks = len(blocks)
                    n_channels = len(channels)
                    
                    print(f"Exporting {n_channels} channel(s) across {n_blocks} block(s) as SVG.")
                    
                    # Create a figure with subplots
                    fig, axes = plt.subplots(n_channels, n_blocks, figsize=(4*n_blocks, 3*n_channels), squeeze=False)
                    plt.subplots_adjust(hspace=0.5, wspace=0.4)
                    
                    for row_idx, ch in enumerate(channels):
                        for col_idx, blk in enumerate(blocks):
                            ax = axes[row_idx, col_idx]
                            psd_info = psd_results[blk].get(ch, {})
                            freqs = psd_info.get('freqs')
                            psd = psd_info.get('psd')
                            
                            if psd is None or freqs is None:
                                ax.text(0.5, 0.5, f"No PSD data\n{blk}:{ch}", ha='center', va='center')
                                ax.axis('off')
                                continue
                            
                            # Compute mean and std
                            psd_mean = np.nanmean(psd, axis=0)
                            psd_std = np.nanstd(psd, axis=0)
                            
                            ax.plot(freqs, psd_mean, color='blue', label='Mean')
                            ax.fill_between(freqs, psd_mean - psd_std, psd_mean + psd_std, color='blue', alpha=0.3, label='±Std')
                            
                            # Set titles and labels
                            if row_idx == 0:
                                ax.set_title(blk, fontsize=10)
                            if col_idx == 0:
                                ax.set_ylabel(ch, fontsize=10)
                            
                            # Set axis ranges
                            if x_min_widget.value is not None and x_max_widget.value is not None:
                                ax.set_xlim(x_min_widget.value, x_max_widget.value)
                            if y_min_widget.value is not None and y_max_widget.value is not None:
                                ax.set_ylim(y_min_widget.value, y_max_widget.value)
                            
                            # Add legend only to first subplot to avoid clutter
                            if row_idx == 0 and col_idx == 0:
                                ax.legend(loc='upper right', fontsize=8)
                    
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plt.suptitle("PSD Plots: Channels vs Blocks", fontsize=12)
                    
                    # Save the figure as SVG
                    svg_filename = os.path.join(export_dir, f"{fname_base}.svg")
                    try:
                        fig.savefig(svg_filename, format='svg')
                        print(f"Saved SVG plot to '{svg_filename}'")
                    except Exception as e:
                        print(f"ERROR saving SVG plot: {e}")
                    plt.close(fig)
                else:
                    # Single block plotting
                    block = list(psd_results.keys())[0]
                    print(f"Exporting plots for single block: {block}")
                    
                    channels = sorted(selected_channels)
                    n_channels = len(channels)
                    
                    fig, axes = plt.subplots(n_channels, 1, figsize=(6, 3*n_channels), squeeze=False)
                    plt.subplots_adjust(hspace=0.5)
                    
                    for row_idx, ch in enumerate(channels):
                        ax = axes[row_idx, 0]
                        psd_info = psd_results[block].get(ch, {})
                        freqs = psd_info.get('freqs')
                        psd = psd_info.get('psd')
                        
                        if psd is None or freqs is None:
                            ax.text(0.5, 0.5, f"No PSD data\n{block}:{ch}", ha='center', va='center')
                            ax.axis('off')
                            continue
                        
                        # Compute mean and std
                        psd_mean = np.nanmean(psd, axis=0)
                        psd_std = np.nanstd(psd, axis=0)
                        
                        ax.plot(freqs, psd_mean, color='blue', label='Mean')
                        ax.fill_between(freqs, psd_mean - psd_std, psd_mean + psd_std, color='blue', alpha=0.3, label='±Std')
                        
                        # Set titles and labels
                        ax.set_title(ch, fontsize=10)
                        ax.set_xlabel("Frequency (Hz)", fontsize=10)
                        ax.set_ylabel("PSD (V²/Hz)", fontsize=10)
                        
                        # Set axis ranges
                        if x_min_widget.value is not None and x_max_widget.value is not None:
                            ax.set_xlim(x_min_widget.value, x_max_widget.value)
                        if y_min_widget.value is not None and y_max_widget.value is not None:
                            ax.set_ylim(y_min_widget.value, y_max_widget.value)
                        
                        # Add legend only to first subplot to avoid clutter
                        if row_idx == 0:
                            ax.legend(loc='upper right', fontsize=8)
                    
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plt.suptitle(f"PSD Plots: Channels from {block}", fontsize=12)
                    
                    # Save the figure as SVG
                    svg_filename = os.path.join(export_dir, f"{fname_base}.svg")
                    try:
                        fig.savefig(svg_filename, format='svg')
                        print(f"Saved SVG plot to '{svg_filename}'")
                    except Exception as e:
                        print(f"ERROR saving SVG plot: {e}")
                    plt.close(fig)
            
        save_button.on_click(save_svg_plots)

##############################################################################
# 6) GUI LAYOUT
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

# New Export SVG Widgets
export_svg_dir_chooser = FileChooser(
    os.getcwd(),
    title='Select Export Directory for SVG Plots',
    select_default=False
)
export_svg_dir_chooser.show_only_dirs = True

export_svg_filename = widgets.Text(
    value='psd_plot',
    description='SVG Base Filename:',
    layout=widgets.Layout(width='300px')
)

export_svg_button = Button(
    description='Export SVG Plots',
    button_style='warning',
    tooltip='Export the plotted PSDs as SVG files'
)

export_svg_output = Output()

def on_export_svg_button_clicked(b):
    with export_svg_output:
        clear_output()
        # Trigger the export process
        on_export_svg_clicked(b)

export_svg_button.on_click(on_export_svg_button_clicked)

export_svg_box = VBox([
    Label("STEP 3: Export Plots as SVG"),
    export_svg_dir_chooser,
    export_svg_filename,
    export_svg_button,
    export_svg_output
])

main_ui = VBox([
    widgets.HTML("<h2>PSD Computation with Optional Segmentation and Export</h2>"),
    compute_box,
    plot_box,
    export_svg_box
])

# Attach callbacks
compute_psd_button.on_click(on_compute_psd_clicked)
plot_psd_button.on_click(on_plot_psd_clicked)

# Display UI
display(main_ui)

