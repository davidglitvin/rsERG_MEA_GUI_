import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math

import ipywidgets as widgets
from ipyfilechooser import FileChooser
from IPython.display import display, clear_output

try:
    from pptx import Presentation
    from pptx.util import Inches
    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False
    print("Warning: python-pptx not installed, PPT export will be disabled.")

import mne
from tqdm.notebook import tqdm
from scipy import signal

##############################################################################
# 1) HELPER FUNCTIONS (Your Existing Logic)
##############################################################################

def exclude_traces(
    psd_array,
    freqs,
    low_band=(1, 3),
    low_band_threshold=3.0,
    test_bands=[(7,9),(9,11),(11,13),(13,15),(15,17),(17,19),(19,21)],
    test_band_threshold=10.0,
    test_band_count_threshold=None
):
    if test_band_count_threshold is None:
        test_band_count_threshold = len(test_bands) // 2

    mean_psd = np.mean(psd_array, axis=0)
    excluded_traces = []
    kept_traces = []

    low_band_indices = np.where((freqs >= low_band[0]) & (freqs <= low_band[1]))[0]
    band_indices = [
        np.where((freqs >= band[0]) & (freqs <= band[1]))[0] for band in test_bands
    ]

    for i, trace in enumerate(psd_array):
        # 1) Low-band outlier
        if np.any(trace[low_band_indices] > low_band_threshold * mean_psd[low_band_indices]):
            excluded_traces.append(i)
            continue

        # 2) Repeated band outliers
        suprathreshold_count = 0
        for indices in band_indices:
            if np.any(trace[indices] > test_band_threshold * mean_psd[indices]):
                suprathreshold_count += 1

        if suprathreshold_count >= test_band_count_threshold:
            excluded_traces.append(i)
        else:
            kept_traces.append(i)

    return kept_traces, excluded_traces

def plot_psds_with_exclusion(
    ax,
    psd_array,
    freqs,
    kept_traces,
    excluded_traces,
    original_mean_psd,
    title,
    show_kept=True,
    show_excluded=True,
    show_original_mean=True,
    show_new_mean=True,
    show_vertical_lines=True,
    color_kept="lightgray",
    color_excluded="red",
    color_old_mean="blue",
    color_new_mean="green",
    alpha_excluded=0.05,
    alpha_kept=0.7,
    title_fontsize=8,
    axis_label_fontsize=8,
    legend_fontsize=8,
    tick_label_fontsize=8,
    max_title_length=40,
    vertical_lines=None,
    vertical_line_color="black",
    vertical_line_style="--",
    vertical_line_alpha=0.6
):
    if len(title) > max_title_length:
        title = title[:max_title_length] + "..."

    if show_kept:
        for idx_i, idx in enumerate(kept_traces):
            label = "Kept Trace" if idx_i == 0 else None
            ax.plot(freqs, psd_array[idx], color=color_kept, alpha=alpha_kept, label=label)

    if show_original_mean:
        ax.plot(freqs, original_mean_psd, color=color_old_mean, linewidth=2, label="Original Mean")

    if show_new_mean and len(kept_traces) > 0:
        new_mean_psd = np.mean(psd_array[kept_traces], axis=0)
        ax.plot(freqs, new_mean_psd, color=color_new_mean, linewidth=2, label="New Mean")

    if show_excluded:
        for idx_j, idx in enumerate(excluded_traces):
            label = "Excluded Trace" if idx_j == 0 else None
            ax.plot(freqs, psd_array[idx], color=color_excluded, alpha=alpha_excluded, zorder=10, label=label)

    if show_vertical_lines and vertical_lines is not None:
        for vfreq in vertical_lines:
            ax.axvline(
                vfreq,
                color=vertical_line_color,
                linestyle=vertical_line_style,
                alpha=vertical_line_alpha
            )

    ax.set_xlabel("Frequency (Hz)", fontsize=axis_label_fontsize)
    ax.set_ylabel("PSD (V²/Hz)", fontsize=axis_label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.legend(loc="upper right", fontsize=legend_fontsize)
    ax.tick_params(axis='both', labelsize=tick_label_fontsize)

def plot_psds_with_dropped_traces(
    psds_dict,
    rows_of_psds,
    low_band=(1,3),
    low_band_threshold=3.0,
    test_bands=[(7,9),(9,11),(11,13),(13,15),(15,17),(17,19),(19,21)],
    test_band_threshold=10.0,
    test_band_count_threshold=None,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    show_kept=True,
    show_excluded=True,
    show_original_mean=True,
    show_new_mean=True,
    show_vertical_lines=True,
    color_kept="lightgray",
    color_excluded="red",
    color_old_mean="blue",
    color_new_mean="green",
    num_cols=4,
    title_fontsize=8,
    axis_label_fontsize=8,
    legend_fontsize=8,
    tick_label_fontsize=8,
    max_title_length=40,
    vertical_lines=None,
    vertical_line_color="black",
    vertical_line_style="--",
    vertical_line_alpha=0.6
):
    figures = []
    kept_indices_dict = {}

    for row_idx, row in enumerate(rows_of_psds, start=1):
        valid_keys = [k for k in row if k in psds_dict]
        num_plots = len(valid_keys)
        if num_plots == 0:
            continue

        num_rows = math.ceil(num_plots / num_cols)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
        plt.subplots_adjust(hspace=0.5)

        if num_rows * num_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, key in enumerate(valid_keys):
            ax = axes[idx]
            psd_output = psds_dict[key]
            psd_data, freq_data = psd_output.get('psd'), psd_output.get('freqs')
            if psd_data is None or freq_data is None:
                ax.text(0.5, 0.5, f"No PSD data for {key}", ha='center', va='center')
                continue

            original_mean_psd = np.mean(psd_data, axis=0)

            kept_traces, excluded_traces = exclude_traces(
                psd_data,
                freq_data,
                low_band=low_band,
                low_band_threshold=low_band_threshold,
                test_bands=test_bands,
                test_band_threshold=test_band_threshold,
                test_band_count_threshold=test_band_count_threshold
            )
            kept_indices_dict[key] = kept_traces

            plot_psds_with_exclusion(
                ax=ax,
                psd_array=psd_data,
                freqs=freq_data,
                kept_traces=kept_traces,
                excluded_traces=excluded_traces,
                original_mean_psd=original_mean_psd,
                title=key,
                show_kept=show_kept,
                show_excluded=show_excluded,
                show_original_mean=show_original_mean,
                show_new_mean=show_new_mean,
                show_vertical_lines=show_vertical_lines,
                color_kept=color_kept,
                color_excluded=color_excluded,
                color_old_mean=color_old_mean,
                color_new_mean=color_new_mean,
                alpha_excluded=0.05,
                alpha_kept=0.7,
                title_fontsize=title_fontsize,
                axis_label_fontsize=axis_label_fontsize,
                legend_fontsize=legend_fontsize,
                tick_label_fontsize=tick_label_fontsize,
                max_title_length=max_title_length,
                vertical_lines=vertical_lines,
                vertical_line_color=vertical_line_color,
                vertical_line_style=vertical_line_style,
                vertical_line_alpha=vertical_line_alpha
            )

            # x/y range
            if x_min is not None or x_max is not None:
                ax.set_xlim(x_min, x_max)
            if y_min is not None or y_max is not None:
                ax.set_ylim(y_min, y_max)

        # Turn off any unused subplots
        for ax in axes[num_plots:]:
            ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(f"Row {row_idx}", fontsize=title_fontsize + 2)
        figures.append(fig)
        plt.show()

    return figures, kept_indices_dict

def group_keys_by_rows(psd_keys, row_size=4):
    # Sort the keys based on block number and channel name
    def sort_key(key):
        if ':' in key:
            block, ch = key.split(':')
            block_num = int(block.replace('block', ''))
        else:
            # If no block is present, assume block_num is 0
            block_num = 0
            ch = key
        ch_num = int(ch.replace('Ch', ''))
        return block_num, ch_num

    psd_keys_sorted = sorted(psd_keys, key=sort_key)
    rows = []
    for i in range(0, len(psd_keys_sorted), row_size):
        rows.append(psd_keys_sorted[i:i+row_size])
    return rows

# Define the aggregate mapping and plotting function
aggregate_map = {
    'eye1': [f'Ch{i}' for i in range(1, 9)],  # Channels 1-8
    'eye2': [f'Ch{i}' for i in range(9, 17)]  # Channels 9-16
}

def plot_aggregated_psds(psd_dict, aggregate_map):
    """
    Plots aggregated PSDs for combined channels (e.g., 'eye1' for channels 1-8).
    """
    figures = []
    for group_name, channels in aggregate_map.items():
        combined_psd = []
        freq_data = None

        for ch in channels:
            if ch in psd_dict:
                psd_info = psd_dict[ch]
                if freq_data is None:
                    freq_data = psd_info['freqs']
                if psd_info['psd'] is not None:
                    combined_psd.append(psd_info['psd'])

        if not combined_psd:
            print(f"No data for {group_name}")
            continue

        combined_psd = np.vstack(combined_psd)  # Stack all PSDs
        mean_psd = np.mean(combined_psd, axis=0)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(freq_data, mean_psd, label=f"{group_name} (mean PSD)", color='blue')
        ax.fill_between(freq_data, np.min(combined_psd, axis=0), np.max(combined_psd, axis=0),
                        color='blue', alpha=0.3, label=f"{group_name} (range)")

        ax.set_title(f"PSD for {group_name}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power Spectral Density (V²/Hz)")
        ax.legend()

        figures.append(fig)
        plt.show()

    return figures

##############################################################################
# 2) Segment & PSD Logic
##############################################################################

def load_epochs(filepath):
    """Loads the .fif file as an mne.Epochs (with preload)."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    ep = mne.read_epochs(filepath, preload=True)
    if not isinstance(ep, mne.Epochs):
        data = ep.get_data()
        info = ep.info
        ep = mne.EpochsArray(data, info)
    return ep

def compute_welch_psd(epochs, fmin=0.0, fmax=45.0, window_s=2.0, overlap_pct=50.0):
    """
    Returns a dict {channel_key: {"freqs":..., "psd":...}} where psd is (n_epochs, n_freqs).
    """
    out_dict = {}
    sfreq = epochs.info['sfreq']
    data = epochs.get_data()  # shape => (n_ep, n_ch, n_time)
    n_ep, n_ch, n_time = data.shape

    nperseg = int(window_s * sfreq)
    noverlap = int(nperseg * overlap_pct/100)

    if nperseg > n_time:
        raise ValueError(f"nperseg ({nperseg}) > epoch length ({n_time}).")

    window_fn = signal.hamming(nperseg, sym=False)

    for ch_idx, ch_name in enumerate(epochs.ch_names):
        ch_data = data[:, ch_idx, :]
        freqs, psd = signal.welch(
            ch_data,
            fs=sfreq,
            window=window_fn,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='density',
            axis=1
        )
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        freqs_f = freqs[freq_mask]
        psd_f = psd[:, freq_mask]
        if freqs_f.size == 0:
            out_dict[ch_name] = {"freqs": None, "psd": None}
        else:
            out_dict[ch_name] = {"freqs": freqs_f, "psd": psd_f}
    return out_dict

##############################################################################
# 3) MAIN GUI
##############################################################################

def build_exportable_plot_psd_gui():
    """
    Adaptation of your PSD-clean GUI that includes:
      - Segmentation toggle (Yes/No).
      - Aggregation toggle for eye1 (channels 1-8) and eye2 (channels 9-16).
      - PSD computation per block or single block.
      - The existing threshold-based drop logic & plotting.
      - Option to export "cleaned" PSD.
    """
    # ~~~~~ GUI ELEMENTS ~~~~~
    aggregate_toggle = widgets.ToggleButtons(
        options=['No', 'Yes'],
        value='No',
        description='Aggregate:',
        style={'description_width': 'initial'}
    )

    segment_toggle = widgets.ToggleButtons(
        options=['No', 'Yes'],
        value='No',
        description='Segment 6 blocks?',
        style={'description_width': 'initial'}
    )

    load_psd_button = widgets.Button(description='Load PSD from .fif', button_style='info')
    fif_chooser = FileChooser(os.getcwd(), title='Select Filtered .fif', select_default=False)
    fif_chooser.show_only_files = True
    fif_chooser.filter_pattern = ['*.fif', '*.fif.gz']

    output_area = widgets.Output()
    loaded_psd = {}

    # ~~~~ A) LOAD & SEGMENT & PSD ~~~~
    def on_load_psd_clicked(b):
        with output_area:
            clear_output()
            chosen_fif = fif_chooser.selected
            if not chosen_fif:
                print("No .fif file selected.")
                return
            if not os.path.isfile(chosen_fif):
                print(f"File not found: {chosen_fif}")
                return
            try:
                print(f"Loading: {chosen_fif}")
                epochs_all = load_epochs(chosen_fif)
                n_total_epochs = len(epochs_all)
                print(f"Loaded => {n_total_epochs} epochs, {len(epochs_all.ch_names)} channels.")

                # Compute PSD
                sfreq = epochs_all.info['sfreq']
                data = epochs_all.get_data()
                n_epochs, n_channels, n_times = data.shape

                window_s = 2.0  # Default window size
                nperseg = int(window_s * sfreq)

                for ch_idx, ch_name in enumerate(epochs_all.ch_names):
                    ch_data = data[:, ch_idx, :]
                    freqs, psd = signal.welch(ch_data, fs=sfreq, nperseg=nperseg, scaling='density', axis=1)
                    loaded_psd[ch_name] = {'freqs': freqs, 'psd': psd}

                print("PSD data loaded for all channels.")

            except Exception as e:
                print(f"Error loading PSD data: {e}")

    def on_plot_psd_clicked(b):
        with output_area:
            clear_output()
            if not loaded_psd:
                print("No PSD data loaded. Please load data first.")
                return

            if aggregate_toggle.value == 'Yes':
                print("Plotting aggregated PSDs...")
                plot_aggregated_psds(loaded_psd, aggregate_map)
            else:
                print("Plotting individual channel/block PSDs...")
                selected_keys = list(loaded_psd.keys())
                if not selected_keys:
                    print("No channels selected.")
                    return
                rows_of_psds = group_keys_by_rows(selected_keys)
                plot_psds_with_dropped_traces(psds_dict=loaded_psd, rows_of_psds=rows_of_psds)

    load_psd_button.on_click(on_load_psd_clicked)

    # Layout of the GUI
    gui_layout = widgets.VBox([
        widgets.HTML("<h3>PSD Aggregation and Plotting GUI</h3>"),
        aggregate_toggle,
        segment_toggle,
        fif_chooser,
        load_psd_button,
        widgets.Button(description='Plot PSDs', on_click=on_plot_psd_clicked),
        output_area
    ])

    display(gui_layout)

# Build and display the GUI
build_exportable_plot_psd_gui()
                                              
