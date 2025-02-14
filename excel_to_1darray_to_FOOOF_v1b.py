import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
from fooof import FOOOFGroup
from scipy.spatial.distance import cdist
from tqdm.notebook import tqdm  # Progress bar for Jupyter
import glob

# Function to process multiple Excel files and compute the PSD data dictionary
def process_excels(file_paths):
    psd_data_dict = {}

    for file_path in file_paths:
        try:
            excel_data = pd.ExcelFile(file_path)
            file_name = os.path.basename(file_path).split('.')[0]

            for sheet_name in excel_data.sheet_names:
                sheet_df = pd.read_excel(file_path, sheet_name=sheet_name)
                freq_column = sheet_df.iloc[:, 0]

                psd_columns = sheet_df.iloc[:, 1:]
                averaged_psd = psd_columns.mean(axis=1)
                psd_data_dict[f"{file_name}_{sheet_name}"] = {
                    "freq": freq_column.values,
                    "psd": averaged_psd.values
                }
        except Exception as e:
            print(f"An error occurred while processing the Excel file '{file_path}': {e}")

    return psd_data_dict

# Function to run FOOOF analysis
def run_fooof_analysis(psd_data_dict, freq_range, amp_threshold, r2_threshold, max_peaks, fitting_mode):
    fg_dict = {}

    for name, data in tqdm(psd_data_dict.items(), desc="Fitting PSDs with FOOOF"):
        freq = data["freq"]
        psd = data["psd"]

        fg = FOOOFGroup(
            peak_width_limits=[freq_range[0], freq_range[1]],
            max_n_peaks=max_peaks,
            min_peak_height=amp_threshold,
            verbose=False,
            aperiodic_mode=fitting_mode,
        )

        fg.fit(freq, psd[np.newaxis, :], freq_range)
        fg.drop(fg.get_params('r_squared') < r2_threshold)

        fg_dict[name] = fg
    return fg_dict

# Function to analyze quality by plotting closest spectra to the mean
def plot_closest_to_mean(fg, name, save_format, output_dir):
    models = [fg.get_fooof(ind=i) for i in range(len(fg))]
    spectra = [model.power_spectrum for model in models]
    spectra = np.array(spectra)

    if spectra.ndim != 2:
        raise ValueError(f"Unexpected spectra shape: {spectra.shape}. Expected a 2D array.")

    mean_spectrum = np.mean(spectra, axis=0)
    distances = cdist(spectra, mean_spectrum[None, :], metric='euclidean').flatten()
    closest_indices = np.argsort(distances)[:10]

    for index in closest_indices:
        fm = fg.get_fooof(ind=index, regenerate=True)
        title = f"{name} - Closest Example {index}"
        fm.plot(title=title, plot_peaks='shade')

        if save_format:
            plt.savefig(os.path.join(output_dir, f"{name}_example_{index}.{save_format}"))
        plt.show()
        plt.clf()

# GUI Elements
file_picker = widgets.Text(
    description="Excel Folder:",
    placeholder="Enter the folder path containing Excel files",
    layout=widgets.Layout(width="70%"),
)

freq_range_min = widgets.FloatText(description="Freq Min:", value=4.0, layout=widgets.Layout(width="30%"))
freq_range_max = widgets.FloatText(description="Freq Max:", value=10.0, layout=widgets.Layout(width="30%"))
amp_threshold = widgets.FloatText(description="Amplitude Threshold:", value=0.2, layout=widgets.Layout(width="50%"))
r2_threshold = widgets.FloatText(description="R² Threshold:", value=0.5, layout=widgets.Layout(width="50%"))
max_peaks = widgets.IntText(description="Max Peaks:", value=2, layout=widgets.Layout(width="50%"))
fitting_mode = widgets.Dropdown(description="Fitting Mode:", options=["knee", "fixed"], value="knee", layout=widgets.Layout(width="50%"))
output_format = widgets.Dropdown(description="Save Format:", options=[None, "svg", "png", "jpeg", "ppt"], value=None, layout=widgets.Layout(width="50%"))
output_folder = widgets.Text(
    description="Output Folder:",
    placeholder="Enter the folder path to save outputs",
    layout=widgets.Layout(width="70%"),
)
run_button = widgets.Button(description="Process and Analyze", button_style="success", icon="check")
output = widgets.Output()

# Button click handler
def on_run_button_click(b):
    with output:
        output.clear_output()
        folder_path = file_picker.value
        save_format = output_format.value
        output_dir = output_folder.value

        if not os.path.isdir(folder_path):
            print("Error: Please provide a valid folder path.")
            return

        if save_format and not os.path.isdir(output_dir):
            print("Error: Please provide a valid output folder path.")
            return

        print(f"Processing files in folder: {folder_path}")
        file_paths = glob.glob(os.path.join(folder_path, "*.xlsx"))
        psd_data_dict = process_excels(file_paths)

        if psd_data_dict:
            print("PSD data processed successfully. Running FOOOF analysis...")
            freq_range = [freq_range_min.value, freq_range_max.value]
            fg_dict = run_fooof_analysis(
                psd_data_dict,
                freq_range=freq_range,
                amp_threshold=amp_threshold.value,
                r2_threshold=r2_threshold.value,
                max_peaks=max_peaks.value,
                fitting_mode=fitting_mode.value,
            )

            print("\nAnalyzing quality of FOOOF fits...")
            for name, fg in fg_dict.items():
                try:
                    print(f"\nPlotting 10 examples closest to mean for: {name}")
                    plot_closest_to_mean(fg, name, save_format, output_dir)
                except ValueError as e:
                    print(f"Error processing {name}: {e}")
            print("Analysis and plotting completed.")
        else:
            print("Processing failed. Please check the folder and try again.")

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
            widgets.HBox([output_format, output_folder]),
            run_button,
            output,
        ]
    )
)