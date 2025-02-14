"""
open_ephys_settings_gui.py

A simple Python module that provides:
1) A function to parse Open Ephys "settings.xml"
2) A GUI (using ipywidgets) to select a file path and display the parsed info.

Usage in a Jupyter notebook:
----------------------------
import open_ephys_settings_gui
open_ephys_settings_gui.run_gui()

Requires:
---------
 - ipywidgets
 - xml.etree.ElementTree
 - Jupyter environment (for ipywidgets)
"""

import os
import xml.etree.ElementTree as ET
import ipywidgets as widgets
from IPython.display import display, clear_output

def parse_settings_xml(xml_path):
    """
    Parse an Open Ephys settings.xml file to extract:
      - The sample rate from the 'Record Node' processor's <STREAM> block
      - The number of channels from the 'Record Node'
      - Which channel indices were enabled by the 'Channel Map' processor.

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


def run_gui():
    """
    Displays a simple ipywidgets GUI in Jupyter that:
     1) Lets you enter a path to settings.xml
     2) Parses the file on button click
     3) Shows the results (sample rate, channel count, enabled indices)
    """
    # Widget 1: Text box for the XML path
    xml_path_box = widgets.Text(
        value='',
        placeholder='Enter path to settings.xml',
        description='XML Path:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='600px')
    )

    # Widget 2: Button to parse
    parse_button = widgets.Button(
        description='Parse settings.xml',
        button_style='',
        tooltip='Parse the given XML file'
    )

    # Widget 3: Output area to display results
    output_area = widgets.Output()

    def on_parse_button_click(b):
        with output_area:
            clear_output()
            xml_path = xml_path_box.value.strip()
            if not xml_path:
                print("Please provide a path to settings.xml.")
                return
            try:
                md = parse_settings_xml(xml_path)
                print("=== Parsed Open Ephys Metadata ===")
                print(f"Sampling rate (Hz):  {md['sampling_rate']}")
                print(f"Number of channels:  {md['n_channels']}")
                print(f"Enabled channel IDs: {md['enabled_indices']}")
            except Exception as e:
                print(f"Error: {e}")

    parse_button.on_click(on_parse_button_click)

    # Display the widgets (as a vertical layout)
    display(widgets.VBox([
        widgets.HBox([xml_path_box, parse_button]),
        output_area
    ]))


# If someone runs this script directly (e.g. "python open_ephys_settings_gui.py"),
# we can attempt to start the GUI. In a terminal, though, ipywidgets won't display.
# It's primarily for Jupyter usage. But we'll include the call for convenience:
if __name__ == '__main__':
    print("This module is mainly intended for Jupyter usage (ipywidgets).")
    print("To use the GUI, import in a Jupyter notebook and call run_gui().")
