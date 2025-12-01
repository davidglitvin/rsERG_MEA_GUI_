# ==== NaN-aware PSD GUI for filtered FIF (EyeAvg, 2 ch, 2000 Hz) ====
# This is an adaptation of your original psd_compute module to:
#   - load a filtered .fif file (with NaNs),
#   - compute PSD per epoch/channel using ONLY the longest finite segment,
#   - save PSD results to a pickle (same structure as before),
#   - provide a GUI to plot mean±std or individual+mean.
#
# It works generically, but is tuned for files like:
#   "/Users/davidlitvin/Desktop/Isoflurane/MNE Array/1_C57_220_Isoflurane0p5per_up_epoched_EyeAvg_manualNaN_bp0p5_100Hz.fif"

import os
import pickle
import numpy as np
import mne
from scipy import signal
from tqdm.notebook import tqdm

import ipywidgets as widgets
from ipyfilechooser import FileChooser
from IPython.display import display, clear_output
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Helper: NaN-aware Welch for ONE epoch
# -------------------------------------------------------------------
def welch_longest_clean_segment(
    x,
    sfreq,
    nperseg,
    noverlap,
    window,
    fmin,
    fmax,
):
    """
    Compute Welch PSD for a 1D epoch x (may contain NaNs),
    using ONLY the longest contiguous segment of finite samples.

    Returns
    -------
    freqs_sel : ndarray or None
    psd_sel   : ndarray or None
        If None, there was not enough clean data (or all NaNs).
    """
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    if mask.sum() < nperseg:
        # Not enough clean samples for this epoch
        return None, None

    finite_idx = np.where(mask)[0]
    # contiguous runs of finite indices
    starts = [finite_idx[0]]
    ends = []
    for i in range(1, len(finite_idx)):
        if finite_idx[i] != finite_idx[i - 1] + 1:
            ends.append(finite_idx[i - 1])
            starts.append(finite_idx[i])
    ends.append(finite_idx[-1])

    # pick longest run
    lengths = [e - s + 1 for s, e in zip(starts, ends)]
    best_idx = int(np.argmax(lengths))
    s = starts[best_idx]
    e = ends[best_idx]
    seg = x[s:e + 1]

    if seg.size < nperseg:
        return None, None

    freqs, psd = signal.welch(
        seg,
        fs=sfreq,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density',
        axis=0
    )

    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_sel = freqs[freq_mask]
    psd_sel = psd[freq_mask]

    return freqs_sel, psd_sel


# -------------------------------------------------------------------
# Widgets (same structure as your original psd_compute)
# -------------------------------------------------------------------

# 1. Input FileChooser for filtered FIF
input_fif_chooser = FileChooser(
    os.getcwd(),
    title='Select Filtered .fif File',
    select_default=False
)
input_fif_chooser.show_only_files = True
input_fif_chooser.filter_pattern = ['*.fif', '*.fif.gz']

# OPTIONAL: pre-point chooser to your typical directory, if you want
# input_fif_chooser.default_path = r"/Users/davidlitvin/Desktop/Isoflurane/MNE Array"

# 2. PSD parameters
window_length_widget = widgets.FloatText(
    value=2.0,
    description='Window (s):',
    layout=widgets.Layout(width='200px'),
    tooltip='Welch window length in seconds'
)

overlap_widget = widgets.FloatSlider(
    value=50.0,
    min=0.0,
    max=100.0,
    step=1.0,
    description='Overlap (%):',
    continuous_update=False,
    layout=widgets.Layout(width='400px'),
    tooltip='Set overlap % (0–100)'
)

# 3. Output pickle chooser
output_pickle_chooser = FileChooser(
    os.getcwd(),
    title='Select Output Pickle File',
    select_default=False
)
output_pickle_chooser.show_only_files = False
output_pickle_chooser.filter_pattern = ['*.pkl']
output_pickle_chooser.default_filename = 'psd_results_EyeAvg_nan.pkl'

# 4. Compute button
compute_psd_button = widgets.Button(
    description='Compute PSD (NaN-aware)',
    button_style='success',
    tooltip='Compute NaN-aware PSD and save to pickle'
)

# 5. Progress bar
psd_progress = widgets.IntProgress(
    value=0,
    min=0,
    max=100,
    step=1,
    description='Progress:',
    bar_style='info',
    orientation='horizontal',
    layout=widgets.Layout(width='800px')
)

# 6. Output area
psd_output_area = widgets.Output()

# 7. Channel selection (multi-select)
plot_channel_widget = widgets.SelectMultiple(
    options=[],
    description='Select Channels:',
    layout=widgets.Layout(width='400px', height='100px'),
    tooltip='Select one or more channels to plot PSD'
)

# 8. Plot mode (same as your extended version)
plot_mode_widget = widgets.ToggleButtons(
    options=[
        ('Mean & +Std', 'mean_std'),
        ('Individual + Mean', 'individual_mean')
    ],
    description='Plot Mode:',
    value='mean_std',
    button_style='info',
    layout=widgets.Layout(width='400px'),
    tooltips=[
        'Mean PSD with positive std shading',
        'Individual PSD traces in gray + mean'
    ]
)

# 9. Axis ranges
x_min_widget = widgets.FloatText(
    value=0.0,
    description='X Min (Hz):',
    layout=widgets.Layout(width='150px')
)
x_max_widget = widgets.FloatText(
    value=45.0,
    description='X Max (Hz):',
    layout=widgets.Layout(width='150px')
)
y_min_widget = widgets.FloatText(
    value=0.0,
    description='Y Min:',
    layout=widgets.Layout(width='150px')
)
y_max_widget = widgets.FloatText(
    value=1.0,
    description='Y Max:',
    layout=widgets.Layout(width='150px')
)

# 10. Plot button
plot_psd_button = widgets.Button(
    description='Plot PSD',
    button_style='info',
    tooltip='Plot PSD for the selected channels'
)

# storage
psd_results = {}


# -------------------------------------------------------------------
# Compute PSD callback (NaN-aware)
# -------------------------------------------------------------------
def on_compute_psd_clicked(b):
    global psd_results
    psd_results = {}

    with psd_output_area:
        clear_output()
        
        # --- Load FIF ---
        input_fif_path = input_fif_chooser.selected
        if not input_fif_path:
            print("ERROR: No input .fif file selected.")
            return
        if not os.path.isfile(input_fif_path):
            print(f"ERROR: Input .fif file not found: {input_fif_path}")
            return
        if not (input_fif_path.endswith('.fif') or input_fif_path.endswith('.fif.gz')):
            print("ERROR: Input must be .fif or .fif.gz")
            return

        try:
            print(f"Loading Epochs from '{input_fif_path}'...")
            epochs_loaded = mne.read_epochs(input_fif_path, preload=True)
            print(f"Loaded: {len(epochs_loaded)} epochs, {len(epochs_loaded.ch_names)} channels")
        except Exception as e:
            print(f"ERROR loading .fif: {e}")
            return

        if not isinstance(epochs_loaded, mne.Epochs):
            try:
                data_array = epochs_loaded.get_data()
                info = epochs_loaded.info
                epochs = mne.EpochsArray(data_array, info)
                print("Reconstructed mne.Epochs from loaded object.")
            except Exception as e:
                print(f"ERROR reconstructing Epochs: {e}")
                return
        else:
            epochs = epochs_loaded

        # --- Parameters ---
        window_length_s = window_length_widget.value
        overlap_percent = overlap_widget.value

        if window_length_s <= 0:
            print("ERROR: Window length must be > 0.")
            return
        if not (0 <= overlap_percent < 100):
            print("ERROR: Overlap % must be in [0, 100).")
            return

        sfreq = epochs.info['sfreq']
        nperseg = int(window_length_s * sfreq)
        noverlap = int(nperseg * (overlap_percent / 100.0))
        n_times = epochs.get_data().shape[2]

        if nperseg > n_times:
            print(f"ERROR: nperseg ({nperseg}) > n_times per epoch ({n_times}).")
            return

        hamming_window = signal.hamming(nperseg, sym=False)

        # You filtered 0.5–100 Hz; for PSD let's default to 0–45 Hz (adjustable in code)
        fmin = 0.0
        fmax = 45.0

        print("\nPSD parameters:")
        print(f"  sfreq       : {sfreq} Hz")
        print(f"  window      : {window_length_s} s  -> nperseg={nperseg}")
        print(f"  overlap     : {overlap_percent}%  -> noverlap={noverlap}")
        print(f"  freq range  : {fmin}–{fmax} Hz")
        print("  NaN handling: longest contiguous clean segment per epoch\n")

        # --- Validate output pickle ---
        output_pickle_path = output_pickle_chooser.selected
        if not output_pickle_path:
            print("ERROR: No output pickle file selected.")
            return
        output_dir = os.path.dirname(output_pickle_path) or "."
        if not os.path.isdir(output_dir):
            print(f"ERROR: Output directory does not exist: {output_dir}")
            return
        if not output_pickle_path.endswith(".pkl"):
            print("ERROR: Output file must end with .pkl")
            return

        # --- Progress bar ---
        psd_progress.value = 0
        psd_progress.max = len(epochs.ch_names)
        psd_progress.bar_style = 'info'
        display(psd_progress)

        print("Starting NaN-aware PSD computation...\n")

        # --- Loop over channels (NaN-aware) ---
        for ch_idx, ch_name in enumerate(tqdm(epochs.ch_names, desc="Channels")):
            try:
                ch_data = epochs.get_data(picks=ch_name).squeeze(axis=1)  # (n_epochs, n_times)

                if ch_data.ndim != 2:
                    print(f"WARNING: '{ch_name}' data has shape {ch_data.shape}, skipping.")
                    psd_results[ch_name] = {'psd': None, 'freqs': None}
                    psd_progress.value += 1
                    continue

                n_epochs = ch_data.shape[0]
                freqs_ref = None
                psd_list = []

                for ep in range(n_epochs):
                    x = ch_data[ep]
                    freqs_ep, psd_ep = welch_longest_clean_segment(
                        x,
                        sfreq=sfreq,
                        nperseg=nperseg,
                        noverlap=noverlap,
                        window=hamming_window,
                        fmin=fmin,
                        fmax=fmax,
                    )

                    if freqs_ep is None:
                        if freqs_ref is None:
                            psd_list.append(None)
                        else:
                            psd_list.append(np.full_like(freqs_ref, np.nan))
                        continue

                    if freqs_ref is None:
                        freqs_ref = freqs_ep
                    else:
                        if freqs_ep.shape != freqs_ref.shape or not np.allclose(freqs_ep, freqs_ref):
                            raise RuntimeError(
                                f"Frequency grid mismatch for channel '{ch_name}' — check parameters."
                            )

                    psd_list.append(psd_ep)

                if freqs_ref is None:
                    print(f"WARNING: Channel '{ch_name}' had no clean segments ≥ nperseg; PSD all-NaN.")
                    psd_results[ch_name] = {'psd': None, 'freqs': None}
                    psd_progress.value += 1
                    continue

                # Replace None entries with NaN rows
                psd_rows = []
                for psd_ep in psd_list:
                    if psd_ep is None:
                        psd_rows.append(np.full_like(freqs_ref, np.nan))
                    else:
                        psd_rows.append(psd_ep)
                psd_arr = np.vstack(psd_rows)  # (n_epochs, n_freqs)

                if np.isnan(psd_arr).all():
                    print(f"WARNING: PSD for '{ch_name}' is all NaNs.")
                    psd_results[ch_name] = {'psd': None, 'freqs': None}
                    psd_progress.value += 1
                    continue

                psd_results[ch_name] = {'psd': psd_arr, 'freqs': freqs_ref}

            except Exception as e:
                print(f"ERROR computing PSD for channel '{ch_name}': {e}")
                psd_results[ch_name] = {'psd': None, 'freqs': None}
            finally:
                psd_progress.value += 1

        psd_progress.bar_style = 'success'
        print("\nNaN-aware PSD computation finished.\n")

        # Populate channel widget with valid channels
        valid_channels = [ch for ch, data in psd_results.items() if data['psd'] is not None]
        if not valid_channels:
            print("ERROR: No valid PSD data for any channel.")
            return
        plot_channel_widget.options = valid_channels

        # Save PSD dict
        try:
            with open(output_pickle_path, "wb") as f:
                pickle.dump(psd_results, f)
            print(f"PSD results saved to '{output_pickle_path}'.")
            print(f"Channels with valid PSD: {len(valid_channels)} / {len(epochs.ch_names)}")
        except Exception as e:
            print(f"ERROR saving PSD pickle: {e}")
            return

        # Set default Y max from data
        try:
            max_psd = max(
                np.nanmax(data['psd']) for data in psd_results.values() if data['psd'] is not None
            )
            y_max_widget.value = float(max_psd * 1.1)
        except Exception:
            y_max_widget.value = 1.0


# -------------------------------------------------------------------
# Plotting callback (same GUI logic, NaN-aware under the hood)
# -------------------------------------------------------------------
def on_plot_psd_clicked(b):
    with psd_output_area:
        plt.close('all')
        clear_output()

        if not psd_results:
            print("ERROR: No PSD results available. Compute PSD first.")
            return

        selected_channels = plot_channel_widget.value
        if not selected_channels:
            print("ERROR: No channels selected.")
            return

        plot_mode = plot_mode_widget.value

        x_min = x_min_widget.value
        x_max = x_max_widget.value
        y_min = y_min_widget.value
        y_max = y_max_widget.value

        if x_min >= x_max:
            print("ERROR: X Min must be < X Max.")
            return
        if y_min >= y_max:
            print("ERROR: Y Min must be < Y Max.")
            return

        for channel in selected_channels:
            psd_data = psd_results.get(channel, {})
            if not psd_data or psd_data["psd"] is None:
                print(f"Skipping '{channel}': no PSD data.")
                continue

            psd = psd_data["psd"]   # (n_epochs, n_freqs)
            freqs = psd_data["freqs"]

            if psd.size == 0:
                print(f"ERROR: PSD data for '{channel}' is empty.")
                continue

            psd_mean = np.nanmean(psd, axis=0)
            psd_std = np.nanstd(psd, axis=0)

            if psd_mean.shape != freqs.shape:
                print("ERROR: PSD mean/freq shape mismatch.")
                continue

            plt.figure(figsize=(10, 6))

            if plot_mode == "mean_std":
                plt.plot(freqs, psd_mean, color="blue", label=f"{channel} mean PSD")
                plt.fill_between(
                    freqs,
                    psd_mean,
                    psd_mean + psd_std,
                    alpha=0.3,
                    color="blue",
                    label="+1 std",
                )
            elif plot_mode == "individual_mean":
                for ep in range(psd.shape[0]):
                    plt.plot(freqs, psd[ep], color="lightgray", linewidth=0.5)
                plt.plot(freqs, psd_mean, color="blue", label=f"{channel} mean PSD")
            else:
                print("Unknown plot mode.")
                continue

            plt.title(f"PSD for {channel} (NaN-aware Welch)")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("PSD (V²/Hz)")
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.show()


# -------------------------------------------------------------------
# Wire up callbacks & layout
# -------------------------------------------------------------------
compute_psd_button.on_click(on_compute_psd_clicked)
plot_psd_button.on_click(on_plot_psd_clicked)

psd_params_box = widgets.HBox([window_length_widget, overlap_widget])
x_axis_box = widgets.HBox([x_min_widget, x_max_widget])
y_axis_box = widgets.HBox([y_min_widget, y_max_widget])

psd_ui = widgets.VBox([
    widgets.HTML("<h2>NaN-aware Power Spectral Density (PSD) from Filtered FIF</h2>"),

    widgets.Label("Step 1: Select filtered Epochs FIF (.fif) with EyeAvg + NaNs"),
    widgets.HBox([widgets.Label("Input .fif:"), input_fif_chooser]),

    widgets.Label("Step 2: Set PSD parameters"),
    psd_params_box,

    widgets.Label("Step 3: Select output PSD pickle"),
    widgets.HBox([widgets.Label("Output .pkl:"), output_pickle_chooser]),

    compute_psd_button,
    psd_progress,
    psd_output_area,

    widgets.Label("Step 4: Plot PSD"),
    widgets.HBox([widgets.Label("Channels:"), plot_channel_widget]),
    widgets.HBox([widgets.Label("Plot mode:"), plot_mode_widget]),
    widgets.HTML("<b>Axis ranges:</b>"),
    widgets.HBox([widgets.Label("X axis (Hz):"), x_axis_box]),
    widgets.HBox([widgets.Label("Y axis:"), y_axis_box]),
    plot_psd_button
])

display(psd_ui)
