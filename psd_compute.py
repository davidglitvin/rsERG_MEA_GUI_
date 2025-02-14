# Step 1: Install Necessary Libraries
# (Uncomment and run this cell if the libraries are not already installed)
# !pip install mne ipywidgets ipyfilechooser tqdm scipy matplotlib joblib

# Step 2: Import Necessary Libraries
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

# Step 3: Define the GUI Widgets

# 1. Input FileChooser Widget for Loading Filtered .fif File
input_fif_chooser = FileChooser(
    os.getcwd(),
    title='Select Filtered .fif File',
    select_default=False
)
input_fif_chooser.show_only_files = True
input_fif_chooser.filter_pattern = ['*.fif', '*.fif.gz']

# 2. PSD Parameters (Window Length and Overlap)
window_length_widget = widgets.FloatText(
    value=2.0,
    description='Window Length (s):',
    layout=widgets.Layout(width='200px'),
    tooltip='Enter window length in seconds'
)

overlap_widget = widgets.FloatSlider(
    value=50.0,
    min=0.0,
    max=100.0,
    step=1.0,
    description='Overlap (%):',
    continuous_update=False,
    layout=widgets.Layout(width='400px'),
    tooltip='Set overlap percentage (0-100%)'
)

# 3. Output FileChooser Widget for Saving PSD Pickle File
output_pickle_chooser = FileChooser(
    os.getcwd(),
    title='Select Output Pickle File',
    select_default=False
)
output_pickle_chooser.show_only_files = False  # Allow creating a new file
output_pickle_chooser.filter_pattern = ['*.pkl']
output_pickle_chooser.default_filename = 'psd_results.pkl'

# 4. Compute PSD Button
compute_psd_button = widgets.Button(
    description='Compute PSD',
    button_style='success',
    tooltip='Compute PSD on loaded data and save results'
)

# 5. Progress Bar
psd_progress = widgets.IntProgress(
    value=0,
    min=0,
    max=100,  # Will be updated based on total steps
    step=1,
    description='Progress:',
    bar_style='info',
    orientation='horizontal',
    layout=widgets.Layout(width='800px')
)

# 6. Output Area for Logs
psd_output_area = widgets.Output()

# 7. Channel Selection for Plotting (Modified to allow multiple selections)
plot_channel_widget = widgets.SelectMultiple(
    options=[],  # Will be populated after PSD computation
    description='Select Channels:',
    layout=widgets.Layout(width='400px', height='100px'),
    tooltip='Select one or more channels to plot PSD'
)

# 8. Plotting Mode Selection (New Widget)
plot_mode_widget = widgets.ToggleButtons(
    options=[
        ('Mean & Positive Std', 'mean_std'),
        ('Individual Traces & Mean', 'individual_mean')
    ],
    description='Plot Mode:',
    value='mean_std',
    button_style='info',
    tooltips=[
        'Display the mean PSD with positive standard deviation shading',
        'Show individual PSD traces in light gray and the mean PSD in blue'
    ],
    layout=widgets.Layout(width='400px')
)

# 9. X and Y Axis Range Widgets (New Widgets)
# X-axis (Frequency) Range
x_min_widget = widgets.FloatText(
    value=0.0,
    description='X Min (Hz):',
    layout=widgets.Layout(width='150px'),
    tooltip='Set the minimum frequency to display on the x-axis'
)

x_max_widget = widgets.FloatText(
    value=45.0,
    description='X Max (Hz):',
    layout=widgets.Layout(width='150px'),
    tooltip='Set the maximum frequency to display on the x-axis'
)

# Y-axis (PSD) Range
y_min_widget = widgets.FloatText(
    value=0.0,
    description='Y Min (V²/Hz):',
    layout=widgets.Layout(width='150px'),
    tooltip='Set the minimum PSD value to display on the y-axis'
)

y_max_widget = widgets.FloatText(
    value=1.0,  # Placeholder value; will be updated based on data
    description='Y Max (V²/Hz):',
    layout=widgets.Layout(width='150px'),
    tooltip='Set the maximum PSD value to display on the y-axis'
)

# 10. Plot PSD Button
plot_psd_button = widgets.Button(
    description='Plot PSD',
    button_style='info',
    tooltip='Plot PSD for the selected channel(s)'
)

# Step 4: Define the Callback Function for PSD Computation
def on_compute_psd_clicked(b):
    global psd_results  # Define psd_results as a global variable
    psd_results = {}
    with psd_output_area:
        clear_output()
        
        # 1. Validate and Load the Filtered .fif File
        input_fif_path = input_fif_chooser.selected
        if not input_fif_path:
            print("ERROR: No input .fif file selected. Please choose a filtered .fif file.")
            return
        if not os.path.isfile(input_fif_path):
            print(f"ERROR: The input .fif file does not exist: {input_fif_path}")
            return
        if not (input_fif_path.endswith('.fif') or input_fif_path.endswith('.fif.gz')):
            print("ERROR: Input file must have a .fif or .fif.gz extension.")
            return
        
        try:
            print(f"Loading Epochs from '{input_fif_path}'...")
            epochs_loaded = mne.read_epochs(input_fif_path, preload=True)
            print(f"Successfully loaded Epochs from '{input_fif_path}'.")
            print(f"Number of epochs: {len(epochs_loaded)}")
            print(f"Number of channels: {len(epochs_loaded.ch_names)}")
            print(f"Epoch duration: {epochs_loaded.tmin} to {epochs_loaded.tmax} seconds\n")
        except Exception as e:
            print(f"ERROR: Failed to load .fif file: {e}")
            return
        
        # 2. Reconstruct Epochs Object if isinstance is False
        if not isinstance(epochs_loaded, mne.Epochs):
            try:
                data_array = epochs_loaded.get_data()
                info = epochs_loaded.info
                epochs = mne.EpochsArray(data_array, info)
                print("Reconstructed Epochs object from EpochsFIF.")
                print(f"Type of reconstructed epochs: {type(epochs)}")
                print(f"isinstance(epochs, mne.Epochs): {isinstance(epochs, mne.Epochs)}\n")
            except Exception as e:
                print(f"ERROR: Failed to reconstruct Epochs object: {e}")
                return
        else:
            epochs = epochs_loaded
        
        # 3. PSD Computation Parameters
        window_length_s = window_length_widget.value
        overlap_percent = overlap_widget.value
        
        if window_length_s <= 0:
            print("ERROR: Window length must be positive.")
            return
        if not (0 <= overlap_percent < 100):
            print("ERROR: Overlap percentage must be between 0 and 100.")
            return
        
        # Calculate parameters for scipy.signal.welch
        sfreq = epochs.info['sfreq']  # Sampling frequency
        nperseg = int(window_length_s * sfreq)
        noverlap = int(nperseg * (overlap_percent / 100.0))
        
        # Validate nperseg against epoch length
        n_times = epochs.get_data().shape[2]
        if nperseg > n_times:
            print(f"ERROR: nperseg ({nperseg}) is greater than the number of time points per epoch ({n_times}).")
            return
        
        # Precompute Hamming window
        hamming_window = signal.hamming(nperseg, sym=False)
        
        # Define frequency range
        fmin = 0.0  # Minimum frequency of interest
        fmax = 45.0  # Maximum frequency of interest
        
        print(f"PSD Computation Parameters:")
        print(f"  - Window Length: {window_length_s} seconds ({nperseg} samples)")
        print(f"  - Overlap: {overlap_percent}% ({noverlap} samples)")
        print(f"  - Frequency Range: {fmin} Hz to {fmax} Hz\n")
        
        # 4. Validate Output Pickle File Selection
        output_pickle_path = output_pickle_chooser.selected
        if not output_pickle_path:
            print("ERROR: No output pickle file selected. Please choose an output file.")
            return
        output_dir = os.path.dirname(output_pickle_path)
        if output_dir and not os.path.isdir(output_dir):
            print(f"ERROR: The output directory does not exist: {output_dir}")
            return
        if not output_pickle_path.endswith('.pkl'):
            print("ERROR: Output file must have a .pkl extension.")
            return
        
        # 5. Initialize Progress Bar
        psd_progress.value = 0
        psd_progress.max = len(epochs.ch_names)  # Number of channels
        psd_progress.bar_style = 'info'
        display(psd_progress)
        
        # 6. Sequentially Compute PSDs for Each Channel
        print("Starting PSD computation using 'welch' method...\n")
        for ch_idx, ch_name in enumerate(tqdm(epochs.ch_names, desc="Processing Channels")):
            try:
                # Extract data for the current channel across all epochs
                ch_data = epochs.get_data(picks=ch_name).squeeze(axis=1)  # Shape: (n_epochs, n_times)
                
                # Validate ch_data shape
                if ch_data.ndim != 2:
                    print(f"WARNING: Channel '{ch_name}' data has unexpected shape {ch_data.shape}. Skipping.")
                    psd_results[ch_name] = {'psd': None, 'freqs': None}
                    psd_progress.value += 1
                    continue
                
                # Compute PSD for all epochs at once using vectorized scipy.signal.welch
                # Returns (freqs, psd) where psd is (n_epochs, n_freqs)
                freqs, psd = signal.welch(
                    ch_data,
                    fs=sfreq,
                    window=hamming_window,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    scaling='density',
                    axis=1
                )
                
                # Mask frequencies to 0-45 Hz
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                freqs_filtered = freqs[freq_mask]
                psd_filtered = psd[:, freq_mask]
                
                # Validate that freqs_filtered is not empty
                if freqs_filtered.size == 0:
                    print(f"WARNING: No frequencies found within {fmin}-{fmax} Hz for channel '{ch_name}'. Skipping.")
                    psd_results[ch_name] = {'psd': None, 'freqs': None}
                    psd_progress.value += 1
                    continue
                
                # Ensure that psd_filtered has valid data
                if np.isnan(psd_filtered).all():
                    print(f"WARNING: PSD data for channel '{ch_name}' contains all NaNs. Skipping.")
                    psd_results[ch_name] = {'psd': None, 'freqs': None}
                    psd_progress.value += 1
                    continue
                
                # Store the PSD and frequency data
                psd_results[ch_name] = {'psd': psd_filtered, 'freqs': freqs_filtered}
                
            except Exception as e:
                print(f"ERROR: Failed to compute PSD for channel '{ch_name}': {e}")
                psd_results[ch_name] = {'psd': None, 'freqs': None}
            finally:
                # Update Progress Bar
                psd_progress.value += 1
        
        # 7. Finalize Progress Bar
        psd_progress.bar_style = 'success'
        print("\nPSD computation completed successfully.\n")
        
        # 8. Populate the channel selection widget for plotting
        # Only include channels with valid PSD data
        valid_channels = [ch for ch, data in psd_results.items() if data['psd'] is not None]
        if not valid_channels:
            print("ERROR: No valid PSD data computed for any channel.")
            return
        plot_channel_widget.options = valid_channels
        
        # 9. Save the PSD Results to Pickle
        try:
            with open(output_pickle_path, 'wb') as f:
                pickle.dump(psd_results, f)
            print(f"PSD results saved successfully to '{output_pickle_path}'.")
            print(f"Number of channels with valid PSD data: {len(valid_channels)} out of {len(epochs.ch_names)}")
        except Exception as e:
            print(f"ERROR: Failed to save PSD results to pickle: {e}")
            return
        
        # 10. Reset X and Y Axis Range Widgets to Default Values
        # Optionally, you can set Y Max based on the maximum PSD value across all channels
        try:
            max_psd = max(
                np.nanmax(data['psd']) for data in psd_results.values() if data['psd'] is not None
            )
            y_max_widget.value = max_psd * 1.1  # Add 10% headroom
        except:
            y_max_widget.value = 1.0  # Fallback value

# Step 5: Define the Plotting Callback Function (Modified)
def on_plot_psd_clicked(b):
    with psd_output_area:
        # Clear previous outputs and plots
        plt.close('all')
        clear_output()
        
        if not psd_results:
            print("ERROR: No PSD results available. Please compute PSD first.")
            return
        
        selected_channels = plot_channel_widget.value
        if not selected_channels:
            print("ERROR: No channels selected. Please select at least one channel to plot.")
            return
        
        plot_mode = plot_mode_widget.value  # Get the selected plot mode
        
        # Retrieve user-specified axis limits
        x_min = x_min_widget.value
        x_max = x_max_widget.value
        y_min = y_min_widget.value
        y_max = y_max_widget.value
        
        # Validate axis limits
        if x_min >= x_max:
            print("ERROR: X Min must be less than X Max.")
            return
        if y_min >= y_max:
            print("ERROR: Y Min must be less than Y Max.")
            return
        
        # Iterate over selected channels and plot
        for channel in selected_channels:
            psd_data = psd_results.get(channel, {})
            if not psd_data or psd_data['psd'] is None:
                print(f"ERROR: No PSD data found for channel '{channel}'. Skipping.")
                continue
            
            psd = psd_data['psd']  # Shape: (n_epochs, n_freqs)
            freqs = psd_data['freqs']  # Shape: (n_freqs,)
            
            if psd.size == 0:
                print(f"ERROR: PSD data for channel '{channel}' is empty.")
                continue
            
            # Compute average PSD and standard deviation across epochs
            psd_mean = np.nanmean(psd, axis=0)
            psd_std = np.nanstd(psd, axis=0)
            
            # Validate that freqs and psd_mean have matching shapes
            if psd_mean.shape != freqs.shape:
                print("ERROR: Mismatch between PSD mean and frequency shapes.")
                continue
            
            plt.figure(figsize=(10, 6))
            
            if plot_mode == 'mean_std':
                # Option 1: Mean PSD with positive standard deviation shading
                plt.plot(freqs, psd_mean, color='blue', label=f'{channel} Mean PSD')
                # Only add positive std shading
                plt.fill_between(freqs, psd_mean, psd_mean + psd_std, alpha=0.3, color='blue', label='Std Dev (+)')
            elif plot_mode == 'individual_mean':
                # Option 2: Individual traces in light gray and mean PSD in blue
                for epoch_idx in range(psd.shape[0]):
                    plt.plot(freqs, psd[epoch_idx], color='lightgray', linewidth=0.5)
                plt.plot(freqs, psd_mean, color='blue', label=f'{channel} Mean PSD')
            else:
                print("ERROR: Unknown plot mode selected.")
                continue
            
            plt.title(f'Power Spectral Density for {channel}')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power Spectral Density (V²/Hz)')
            plt.legend(loc='upper right')
            
            # Apply user-specified axis limits
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            
            plt.tight_layout()
            plt.show()

# Step 6: Attach the Callbacks
compute_psd_button.on_click(on_compute_psd_clicked)
plot_psd_button.on_click(on_plot_psd_clicked)

# Step 7: Layout and Display the GUI

# Arrange PSD parameters horizontally
psd_params_box = widgets.HBox([
    window_length_widget,
    overlap_widget
])

# Arrange X-axis widgets horizontally
x_axis_box = widgets.HBox([
    x_min_widget,
    x_max_widget
])

# Arrange Y-axis widgets horizontally
y_axis_box = widgets.HBox([
    y_min_widget,
    y_max_widget
])

# Combine all widgets into a vertical box with modifications
psd_ui = widgets.VBox([
    widgets.HTML("<h2>Power Spectral Density (PSD) Computation</h2>"),
    
    # Step 1: Load Filtered Epochs
    widgets.Label("### Step 1: Load Filtered Epochs (.fif Format)"),
    widgets.HBox([
        widgets.Label("Input .fif File:"),
        input_fif_chooser
    ]),
    
    # Step 2: Set PSD Parameters
    widgets.Label("### Step 2: Set PSD Parameters"),
    psd_params_box,
    
    # Step 3: Select Output Pickle File
    widgets.Label("### Step 3: Select Output Pickle File"),
    widgets.HBox([
        widgets.Label("Output Pickle File:"),
        output_pickle_chooser
    ]),
    
    # Step 4: Compute PSD Button
    compute_psd_button,
    
    # Step 5: Progress Bar
    psd_progress,
    
    # Step 6: Output Area
    psd_output_area,
    
    # Step 7: Plotting Controls
    widgets.Label("### Step 4: Plot PSD"),
    widgets.HBox([
        widgets.Label("Select Channels:"),
        plot_channel_widget
    ]),
    widgets.HBox([
        widgets.Label("Plot Mode:"),
        plot_mode_widget
    ]),
    widgets.HTML("<b>Customize Plot Axes:</b>"),
    widgets.HBox([
        widgets.Label("X-Axis Range:"),
        x_axis_box
    ]),
    widgets.HBox([
        widgets.Label("Y-Axis Range:"),
        y_axis_box
    ]),
    plot_psd_button
])

# Display the GUI
display(psd_ui)