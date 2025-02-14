import os
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

import ipywidgets as widgets
from ipywidgets import (
    IntRangeSlider,
    SelectMultiple,
    Button,
    ToggleButtons,
    VBox,
    HBox,
    Label,
    Output,
    Text,
    FloatText,
    Layout
)
from IPython.display import display, clear_output

# Suppress potential matplotlib warnings
import warnings
warnings.filterwarnings("ignore")

##############################################################################
# 1) HELPER FUNCTIONS
##############################################################################

def parse_settings_xml(xml_path):
    """
    Parse an Open Ephys settings.xml file to extract:
      - Sampling rate
      - Number of channels
      - Enabled channel indices

    Returns a dict, e.g.:
      {
        "sampling_rate": 10000.0,
        "n_channels": 16,
        "enabled_indices": [16, 17, 18, ..., 63]
      }
    """
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"File not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    metadata = {
        "sampling_rate": None,
        "n_channels": None,
        "enabled_indices": []
    }

    # 1) Find the <PROCESSOR> for the Record Node
    record_node = None
    for proc in root.findall(".//PROCESSOR"):
        if proc.get("name") == "Record Node":
            record_node = proc
            break

    if record_node is not None:
        # Find <STREAM> under the 'Record Node'
        stream_tag = record_node.find("./STREAM")
        if stream_tag is not None:
            sr_str = stream_tag.get("sample_rate")
            ch_count_str = stream_tag.get("channel_count")
            if sr_str is not None:
                metadata["sampling_rate"] = float(sr_str)
            if ch_count_str is not None:
                metadata["n_channels"] = int(ch_count_str)

    # 2) Find the <PROCESSOR> named "Channel Map" to see which channels were enabled
    channel_map_node = None
    for proc in root.findall(".//PROCESSOR"):
        if proc.get("name") == "Channel Map":
            channel_map_node = proc
            break

    if channel_map_node is not None:
        # Inside <PROCESSOR>, go to <CUSTOM_PARAMETERS>/<STREAM>
        custom_params = channel_map_node.find("./CUSTOM_PARAMETERS/STREAM")
        if custom_params is not None:
            # Each <CH index="X" enabled="Y"/>
            for ch_tag in custom_params.findall("CH"):
                idx_str = ch_tag.get("index")
                enabled_str = ch_tag.get("enabled")
                if idx_str is not None and enabled_str == "1":
                    idx = int(idx_str)
                    metadata["enabled_indices"].append(idx)

    return metadata

def load_dat_file(dat_path, n_channels, dtype=np.int16):
    """
    Loads a .dat file using memmap for efficient access.

    Parameters:
    - dat_path: Path to the .dat file.
    - n_channels: Number of channels in the data.
    - dtype: Data type of the samples.

    Returns:
    - memmap object of shape (n_samples, n_channels)
    """
    file_size = os.path.getsize(dat_path)
    total_samples = file_size // np.dtype(dtype).itemsize
    n_samples = total_samples // n_channels

    if total_samples % n_channels != 0:
        print("Warning: The total number of samples is not perfectly divisible by the number of channels.")

    data = np.memmap(dat_path, dtype=dtype, mode='r', shape=(n_samples, n_channels))
    return data

def get_enabled_channels(metadata):
    """
    Given parsed metadata, return list of enabled channel labels.

    Parameters:
    - metadata: dict with "enabled_indices"

    Returns:
    - List of strings like ["Ch17", "Ch18", ...]
    """
    return [f"Ch{idx+1}" for idx in metadata["enabled_indices"]]

def extract_time_range(n_samples, sfreq):
    """
    Given number of samples and sampling frequency, return total time in seconds.

    Parameters:
    - n_samples: Total number of samples.
    - sfreq: Sampling frequency.

    Returns:
    - Total time in seconds.
    """
    return n_samples / sfreq

##############################################################################
# 2) WIDGETS INITIALIZATION
##############################################################################

# Output widgets for logs and messages
log_output = Output()

# Text boxes for file paths
dat_file_text = Text(
    value='',
    placeholder='Enter full path to your .dat file here',
    description='.dat File Path:',
    layout=Layout(width='800px')
)

xml_file_text = Text(
    value='',
    placeholder='Enter full path to settings.xml here (optional)',
    description='settings.xml Path:',
    layout=Layout(width='800px')
)

# Button to parse settings.xml
parse_xml_button = Button(
    description='Parse settings.xml',
    button_style='info',
    tooltip='Parse the selected settings.xml to auto-select channels'
)

# Channel selection widget
channel_select = SelectMultiple(
    options=[],  # To be populated after parsing XML or manually
    value=[],
    description='Channels:',
    style={'description_width': 'initial'},
    layout=Layout(width='800px', height='200px')
)

# Toggle for Auto Load vs Manual Load
load_mode_toggle = ToggleButtons(
    options=['Auto Load', 'Manual Load'],
    value='Auto Load',
    description='Load Mode:',
    button_style='',
    tooltips=[
        'Automatically load channels from settings.xml',
        'Manually select channels'
    ],
    layout=Layout(width='300px')
)

# Time range selection sliders
time_range_slider = IntRangeSlider(
    value=[0, 600],  # Default 0 to 600 seconds (60 minutes)
    min=0,
    max=3600,  # Assuming maximum 60 minutes
    step=1,
    description='Time Range (s):',
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d',
    layout=Layout(width='800px')
)

# Plot settings: x_min, x_max, y_min, y_max
plot_settings_box = VBox([
    HBox([
        FloatText(
            value=0.0,
            description='X Min (s):',
            style={'description_width': 'initial'},
            layout=Layout(width='200px')
        ),
        FloatText(
            value=600.0,
            description='X Max (s):',
            style={'description_width': 'initial'},
            layout=Layout(width='200px')
        ),
        FloatText(
            value=None,
            description='Y Min:',
            style={'description_width': 'initial'},
            layout=Layout(width='200px')
        ),
        FloatText(
            value=None,
            description='Y Max:',
            style={'description_width': 'initial'},
            layout=Layout(width='200px')
        )
    ]),
    Label("Leave Y Min and Y Max empty to auto-scale the plot.")
])

# Button to update the plot
update_plot_button = Button(
    description='Update Plot',
    button_style='primary',
    tooltip='Plot the selected channels and time range'
)

# Output area for plot
plot_output = Output()

##############################################################################
# 3) CALLBACK FUNCTIONS
##############################################################################

def on_parse_xml_clicked(b):
    """
    Callback for parsing settings.xml and auto-selecting channels.
    """
    with log_output:
        clear_output()
        xml_path = xml_file_text.value.strip()
        if not xml_path:
            print("No settings.xml file path provided.")
            return
        if not os.path.isfile(xml_path):
            print(f"settings.xml file not found at: {xml_path}")
            return
        try:
            metadata = parse_settings_xml(xml_path)
            if metadata["enabled_indices"]:
                enabled_channels = get_enabled_channels(metadata)
                channel_select.options = enabled_channels
                if load_mode_toggle.value == 'Auto Load':
                    channel_select.value = enabled_channels
                    channel_select.disabled = True
                    print(f"Auto Load: {len(enabled_channels)} channels loaded from settings.xml.")
                else:
                    channel_select.disabled = False
                    print(f"Manual Load: {len(enabled_channels)} channels available for selection.")
            else:
                print("No enabled channels found in settings.xml.")
        except Exception as e:
            print(f"Error parsing settings.xml: {e}")

parse_xml_button.on_click(on_parse_xml_clicked)

def on_load_mode_change(change):
    """
    Callback for load mode toggle.
    """
    with log_output:
        clear_output()
        mode = change['new']
        if mode == 'Auto Load':
            # Attempt to auto-load channels if metadata is available
            xml_path = xml_file_text.value.strip()
            if xml_path and os.path.isfile(xml_path):
                try:
                    metadata = parse_settings_xml(xml_path)
                    if metadata["enabled_indices"]:
                        enabled_channels = get_enabled_channels(metadata)
                        channel_select.options = enabled_channels
                        channel_select.value = enabled_channels
                        channel_select.disabled = True
                        print(f"Auto Load: {len(enabled_channels)} channels loaded from settings.xml.")
                    else:
                        print("No enabled channels found in settings.xml.")
                except Exception as e:
                    print(f"Error parsing settings.xml: {e}")
            else:
                print("No settings.xml found. Please provide a valid settings.xml path or switch to Manual Load.")
        else:
            # Manual Load: Enable channel selection
            # If settings.xml was previously parsed, keep the enabled channels
            xml_path = xml_file_text.value.strip()
            if xml_path and os.path.isfile(xml_path):
                try:
                    metadata = parse_settings_xml(xml_path)
                    if metadata["enabled_indices"]:
                        enabled_channels = get_enabled_channels(metadata)
                        channel_select.options = enabled_channels
                        channel_select.value = []
                        channel_select.disabled = False
                        print(f"Manual Load: {len(enabled_channels)} channels available for selection.")
                    else:
                        print("No enabled channels found in settings.xml.")
                except Exception as e:
                    print(f"Error parsing settings.xml: {e}")
            else:
                # If no XML, assume 64 channels
                default_channels = [f"Ch{i+1}" for i in range(64)]
                channel_select.options = default_channels
                channel_select.value = []
                channel_select.disabled = False
                print("Manual Load: No settings.xml provided. Assuming 64 channels.")

load_mode_toggle.observe(on_load_mode_change, names='value')

def on_update_plot_clicked(b):
    """
    Callback to update the Matplotlib plot based on selected channels and time range.
    """
    with log_output:
        clear_output()
        dat_path = dat_file_text.value.strip()
        if not dat_path:
            print("No .dat file path provided.")
            return
        if not os.path.isfile(dat_path):
            print(f".dat file not found at: {dat_path}")
            return

        # Parse settings.xml if available to get metadata
        xml_path = xml_file_text.value.strip()
        if xml_path and os.path.isfile(xml_path) and load_mode_toggle.value == 'Auto Load':
            try:
                metadata = parse_settings_xml(xml_path)
                sfreq = metadata["sampling_rate"] if metadata["sampling_rate"] else 10000.0
                n_channels = metadata["n_channels"] if metadata["n_channels"] else 16
                enabled_indices = metadata["enabled_indices"]
            except Exception as e:
                print(f"Error parsing settings.xml: {e}")
                sfreq = 10000.0
                n_channels = 16
        else:
            # If not using settings.xml, assume 64 channels
            sfreq = 10000.0
            n_channels = 64

        # Load data using memmap
        try:
            data = load_dat_file(dat_path, n_channels, dtype=np.int16)
            total_time = extract_time_range(data.shape[0], sfreq)
            print(f"Data loaded: {data.shape[0]} samples, {n_channels} channels, {total_time/60:.2f} minutes.")
        except Exception as e:
            print(f"Error loading .dat file: {e}")
            return

        # Get selected channels
        selected_channels = list(channel_select.value)
        if not selected_channels:
            print("No channels selected for visualization.")
            return

        # Map channel labels to data indices
        # Assuming channel_select.options are ordered as per data columns
        channel_indices = [channel_select.options.index(ch) for ch in selected_channels]

        # Get time range from slider
        start_time, end_time = time_range_slider.value
        if start_time >= end_time:
            print("Invalid time range: Start time must be less than end time.")
            return
        if end_time > total_time:
            print(f"End time exceeds total data duration ({total_time:.2f}s). Adjusting to max.")
            end_time = int(total_time)
            time_range_slider.value = [start_time, end_time]

        # Convert time to sample indices
        start_idx = int(start_time * sfreq)
        end_idx = int(end_time * sfreq)
        start_idx = max(0, start_idx)
        end_idx = min(end_idx, data.shape[0])

        # Extract the data slice
        data_slice = data[start_idx:end_idx, channel_indices]

        # Create time axis
        time_axis = np.linspace(start_time, end_time, end_idx - start_idx)

        # Retrieve plot settings
        x_min = plot_settings_box.children[0].children[0].value
        x_max = plot_settings_box.children[0].children[1].value
        y_min = plot_settings_box.children[0].children[2].value
        y_max = plot_settings_box.children[0].children[3].value

        # Plot using Matplotlib
        plot_output.clear_output()
        with plot_output:
            plt.figure(figsize=(15, 7))
            for idx, ch_label in zip(channel_indices, selected_channels):
                plt.plot(time_axis, data_slice[:, selected_channels.index(ch_label)], label=ch_label)
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.title(f"Time-Series Data from {start_time}s to {end_time}s")
            plt.legend(loc='upper right', ncol=4, fontsize='small')
            plt.tight_layout()

            # Apply plot settings if provided
            if x_min is not None:
                plt.xlim(left=x_min)
            if x_max is not None:
                plt.xlim(right=x_max)
            if y_min is not None:
                plt.ylim(bottom=y_min)
            if y_max is not None:
                plt.ylim(top=y_max)

            plt.show()

        print(f"Plot updated: Channels {selected_channels} from {start_time}s to {end_time}s.")
        print(f"Plot Settings - X Min: {x_min}, X Max: {x_max}, Y Min: {y_min}, Y Max: {y_max}")

update_plot_button.on_click(on_update_plot_clicked)

##############################################################################
# 4) UI COMPONENTS ASSEMBLY
##############################################################################

# Layout for file paths and parse button
file_load_box = VBox([
    Label("STEP 1) Enter the path to your Open Ephys .dat file:"),
    dat_file_text,
    Label("STEP 2) (Optional) Enter the path to your settings.xml file:"),
    xml_file_text,
    parse_xml_button
])

# Layout for channel selection
channel_selection_box = VBox([
    Label("STEP 3) Select Channel Load Mode:"),
    HBox([load_mode_toggle]),
    Label("STEP 4) Select Channels to Visualize:"),
    channel_select
])

# Layout for time range selection
time_selection_box = VBox([
    Label("STEP 5) Select Time Range for Visualization:"),
    time_range_slider
])

# Layout for plot settings
plot_settings_ui = VBox([
    Label("STEP 6) Customize Plot Settings:"),
    plot_settings_box
])

# Layout for update plot button
plot_button_box = VBox([
    update_plot_button
])

# Combine all components into the main UI
main_ui = VBox([
    file_load_box,
    channel_selection_box,
    time_selection_box,
    plot_settings_ui,
    plot_button_box,
    Label("Visualization:"),
    plot_output,
    Label("Logs and Messages:"),
    log_output
])

display(main_ui)

##############################################################################
# 5) INITIALIZE CHANNEL SELECT OPTIONS IF XML NOT PROVIDED
##############################################################################

def initialize_channel_select():
    """
    Initializes the channel selection options based on whether settings.xml is provided.
    If settings.xml is not provided, assume 64 channels.
    """
    xml_path = xml_file_text.value.strip()
    if xml_path and os.path.isfile(xml_path) and load_mode_toggle.value == 'Auto Load':
        try:
            metadata = parse_settings_xml(xml_path)
            if metadata["enabled_indices"]:
                enabled_channels = get_enabled_channels(metadata)
                channel_select.options = enabled_channels
                channel_select.value = enabled_channels
                channel_select.disabled = True
                with log_output:
                    print(f"Auto Load: {len(enabled_channels)} channels loaded from settings.xml.")
            else:
                channel_select.options = []
                with log_output:
                    print("No enabled channels found in settings.xml.")
        except Exception as e:
            channel_select.options = []
            with log_output:
                print(f"Error parsing settings.xml: {e}")
    else:
        # If no XML, assume 64 channels
        default_channels = [f"Ch{i+1}" for i in range(64)]
        channel_select.options = default_channels
        channel_select.value = []
        channel_select.disabled = False
        with log_output:
            print("Manual Load: No settings.xml provided. Assuming 64 channels.")

# Initialize channel_select when the cell is run
initialize_channel_select()