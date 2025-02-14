import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from fooof import FOOOFGroup
from scipy.spatial.distance import cdist
from tqdm.notebook import tqdm
import pickle  # For saving and loading fg_dict
from io import BytesIO
from ipyfilechooser import FileChooser  # Import FileChooser

# Global variable for storing fg_dict
fg_dict = None

# Function to process Excel files and compute the PSD data dictionary
def process_excel(file_paths):
    psd_data_dict_all = {}

    for file_path in file_paths:
        try:
            excel_data = pd.ExcelFile(file_path)
            psd_data_dict = {}
            freq_data = None

            for sheet_name in excel_data.sheet_names:
                sheet_df = pd.read_excel(file_path, sheet_name=sheet_name)
                freq_column = sheet_df.iloc[:, 0]
                if freq_data is None:
                    freq_data = freq_column.values

                psd_columns = sheet_df.iloc[:, 1:]
                averaged_psd = psd_columns.mean(axis=1)
                psd_data_dict[sheet_name] = averaged_psd.values

            psd_data_dict["freq"] = freq_data
            psd_data_dict_all[file_path] = psd_data_dict
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    return psd_data_dict_all

# Function to run FOOOF analysis
def run_fooof_analysis(psd_data_dict_all, freq_range, amp_threshold, r2_threshold, max_peaks, fitting_mode, peak_width_limits):
    global fg_dict
    fg_dict = {}

    for file_path, psd_data_dict in tqdm(psd_data_dict_all.items(), desc="Fitting PSDs with FOOOF"):
        freq = psd_data_dict["freq"]
        file_suffix = os.path.basename(file_path).replace(".xlsx", "")  # Use file name as suffix

        for sheet_name, psd in psd_data_dict.items():
            if sheet_name == "freq":
                continue

            fg = FOOOFGroup(
                peak_width_limits=peak_width_limits,
                max_n_peaks=max_peaks,
                min_peak_height=amp_threshold,
                verbose=False,
                aperiodic_mode=fitting_mode,
            )

            fg.fit(freq, psd[np.newaxis, :], freq_range)
            fg.drop(fg.get_params('r_squared') < r2_threshold)

            # Add the FOOOFGroup to the dictionary with a unique key
            dict_key = f"{sheet_name}_{file_suffix}"
            fg_dict[dict_key] = fg

    return fg_dict

# Save fg_dict to a file
def save_fooof_group(fg_dict, filename="fooof_groups.pkl"):
    if fg_dict is None:
        print("Error: No FOOOFGroup dictionary to save. Run the analysis first.")
        return
    with open(filename, "wb") as f:
        pickle.dump(fg_dict, f)
    print(f"FOOOFGroup dictionary saved to {filename}")

# Load fg_dict from a file
def load_fooof_group(filename="fooof_groups.pkl"):
    global fg_dict
    try:
        with open(filename, "rb") as f:
            fg_dict = pickle.load(f)
        print(f"FOOOFGroup dictionary loaded from {filename}")
    except FileNotFoundError:
        print(f"Error: File {filename} not found. Run the analysis and save the results first.")
    return fg_dict

# GUI Elements
file_picker = FileChooser(
    title="Select Excel Files",
    select_multiple=True,
    file_filter='*.xlsx',
    style={'description_width': 'initial'},
)

freq_range_min = widgets.FloatText(
    description="Freq Min:",
    value=4.0,
    layout=widgets.Layout(width="30%"),
)
freq_range_max = widgets.FloatText(
    description="Freq Max:",
    value=45.0,
    layout=widgets.Layout(width="30%"),
)
amp_threshold = widgets.FloatText(
    description="Amplitude Threshold:",
    value=0.2,
    layout=widgets.Layout(width="50%"),
)
r2_threshold = widgets.FloatText(
    description="R² Threshold:",
    value=0.5,
    layout=widgets.Layout(width="50%"),
)
max_peaks = widgets.IntText(
    description="Max Peaks:",
    value=2,
    layout=widgets.Layout(width="50%"),
)
fitting_mode = widgets.Dropdown(
    description="Fitting Mode:",
    options=["knee", "fixed"],
    value="knee",
    layout=widgets.Layout(width="50%"),
)
peak_width_min = widgets.FloatText(
    description="Peak Width Min:",
    value=2.0,
    layout=widgets.Layout(width="50%"),
)
peak_width_max = widgets.FloatText(
    description="Peak Width Max:",
    value=10.0,
    layout=widgets.Layout(width="50%"),
)
run_button = widgets.Button(
    description="Process and Analyze",
    button_style="success",
    icon="check",
)
output = widgets.Output()

selected_files = []

# File selection handler
def on_file_selection(change):
    global selected_files
    selected_files = file_picker.selected
    with output:
        output.clear_output()
        if selected_files:
            print(f"Selected files ({len(selected_files)}):")
            for f in selected_files:
                print(f)
        else:
            print("No files selected.")

file_picker.register_callback(on_file_selection)

# Button click handler
def on_run_button_click(b):
    global fg_dict
    with output:
        output.clear_output()
        if not selected_files:
            print("Error: Please select Excel files to process.")
            return

        print(f"Processing files: {selected_files}")
        psd_data_dict_all = process_excel(selected_files)

        if psd_data_dict_all:
            freq_range = [freq_range_min.value, freq_range_max.value]
            peak_width_limits = [peak_width_min.value, peak_width_max.value]

            print("Running FOOOF analysis...")
            fg_dict = run_fooof_analysis(
                psd_data_dict_all,
                freq_range=freq_range,
                amp_threshold=amp_threshold.value,
                r2_threshold=r2_threshold.value,
                max_peaks=max_peaks.value,
                fitting_mode=fitting_mode.value,
                peak_width_limits=peak_width_limits,
            )

            # Save the fg_dict for subsequent steps
            save_fooof_group(fg_dict)

            print("\nFOOOF analysis completed. Results saved for subsequent steps.")

            # Troubleshooting: Verify `fg_dict` content
            print("Troubleshooting: Checking fg_dict keys...")
            print(list(fg_dict.keys()))
        else:
            print("Processing failed. Please check the files and try again.")

run_button.on_click(on_run_button_click)

# Display the GUI
display(
    widgets.VBox(
        [
            file_picker,
            widgets.HBox([freq_range_min, freq_range_max]),
            amp_threshold,
            r2_threshold,
            max_peaks,
            fitting_mode,
            widgets.HBox([peak_width_min, peak_width_max]),
            run_button,
            output,
        ]
    )
)
