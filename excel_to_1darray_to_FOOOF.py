import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
from fooof import FOOOFGroup
from scipy.spatial.distance import cdist
from tqdm.notebook import tqdm  # Progress bar for Jupyter

# Function to process the Excel file and compute the PSD data dictionary
def process_excel(file_path):
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
        return psd_data_dict
    except Exception as e:
        print(f"An error occurred while processing the Excel file: {e}")
        return None

# Function to run FOOOF analysis
def run_fooof_analysis(psd_data_dict, freq_range, amp_threshold, r2_threshold, max_peaks, fitting_mode):
    fg_dict = {}
    freq = psd_data_dict["freq"]

    for sheet_name, psd in tqdm(psd_data_dict.items(), desc="Fitting PSDs with FOOOF"):
        if sheet_name == "freq":
            continue

        fg = FOOOFGroup(
            peak_width_limits=[2, 10],
            max_n_peaks=max_peaks,
            min_peak_height=amp_threshold,
            verbose=False,
            aperiodic_mode=fitting_mode,
        )

        fg.fit(freq, psd[np.newaxis, :], freq_range)
        fg.drop(fg.get_params('r_squared') < r2_threshold)

        fg_dict[sheet_name] = fg
    return fg_dict

# Function to analyze quality by plotting closest spectra to the mean
def plot_closest_to_mean(fg, sheet_name):
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
        title = f"{sheet_name} - Closest Example {index}"
        fm.plot(title=title, plot_peaks='shade')
        plt.show()
        plt.clf()

# GUI Elements
file_picker = widgets.Text(
    description="Excel Path:",
    placeholder="Enter the path to the Excel file",
    layout=widgets.Layout(width="70%"),
)

freq_range_min = widgets.FloatText(description="Freq Min:", value=4.0, layout=widgets.Layout(width="30%"))
freq_range_max = widgets.FloatText(description="Freq Max:", value=45.0, layout=widgets.Layout(width="30%"))
amp_threshold = widgets.FloatText(description="Amplitude Threshold:", value=0.2, layout=widgets.Layout(width="50%"))
r2_threshold = widgets.FloatText(description="R² Threshold:", value=0.5, layout=widgets.Layout(width="50%"))
max_peaks = widgets.IntText(description="Max Peaks:", value=2, layout=widgets.Layout(width="50%"))
fitting_mode = widgets.Dropdown(description="Fitting Mode:", options=["knee", "fixed"], value="knee", layout=widgets.Layout(width="50%"))
run_button = widgets.Button(description="Process and Analyze", button_style="success", icon="check")
output = widgets.Output()

# Button click handler
def on_run_button_click(b):
    with output:
        output.clear_output()
        file_path = file_picker.value

        if not file_path.lower().endswith(".xlsx"):
            print("Error: Please provide a valid Excel file with a '.xlsx' extension.")
            return

        print(f"Processing file: {file_path}")
        psd_data_dict = process_excel(file_path)

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
            for sheet_name, fg in fg_dict.items():
                try:
                    print(f"\nPlotting 10 examples closest to mean for sheet: {sheet_name}")
                    plot_closest_to_mean(fg, sheet_name)
                except ValueError as e:
                    print(f"Error processing {sheet_name}: {e}")
            print("Analysis and plotting completed.")
        else:
            print("Processing failed. Please check the file and try again.")

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
            run_button,
            output,
        ]
    )
)
