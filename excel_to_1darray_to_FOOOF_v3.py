import pandas as pd
import numpy as np
import os
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
from fooof import FOOOFGroup
from tqdm.notebook import tqdm  # Progress bar for Jupyter

# Function to process the Excel file and compute the PSD data dictionary
def process_excel(file_path, use_means=False):
    try:
        excel_data = pd.ExcelFile(file_path)
        psd_data_dict = {}
        freq_data = None

        if use_means and len(excel_data.sheet_names) == 2:
            # If the file has two sheets for means
            for sheet_name in excel_data.sheet_names:
                sheet_df = pd.read_excel(file_path, sheet_name=sheet_name)
                freq_column = sheet_df.iloc[:, 0]
                if freq_data is None:
                    freq_data = freq_column.values
                psd_columns = sheet_df.iloc[:, 1:]
                psd_data_dict[sheet_name] = psd_columns
        else:
            # Standard processing for individual PSDs
            for sheet_name in excel_data.sheet_names:
                sheet_df = pd.read_excel(file_path, sheet_name=sheet_name)
                freq_column = sheet_df.iloc[:, 0]
                if freq_data is None:
                    freq_data = freq_column.values

                psd_columns = sheet_df.iloc[:, 1:]
                psd_data_dict[sheet_name] = psd_columns

        psd_data_dict["freq"] = freq_data
        return psd_data_dict
    except Exception as e:
        print(f"An error occurred while processing the Excel file: {e}")
        return None

# Function to run FOOOF analysis
def run_fooof_analysis(psd_data_dict, freq_range, peak_width_limits, amp_threshold, r2_threshold, max_peaks, fitting_mode):
    fg_dict = {}
    freq = psd_data_dict["freq"]

    for sheet_name, psd in tqdm(psd_data_dict.items(), desc="Fitting PSDs with FOOOF"):
        if sheet_name == "freq":
            continue

        fg = FOOOFGroup(
            peak_width_limits=peak_width_limits,
            max_n_peaks=max_peaks,
            min_peak_height=amp_threshold,
            verbose=False,
            aperiodic_mode=fitting_mode,
        )

        if isinstance(psd, pd.DataFrame):
            # Multiple columns (e.g., blocks or eyes)
            for col_name in psd.columns:
                fg.fit(freq, psd[col_name].values[np.newaxis, :], freq_range)
        else:
            # Single trace
            fg.fit(freq, psd.values[np.newaxis, :], freq_range)

        fg.drop(fg.get_params('r_squared') < r2_threshold)
        fg_dict[sheet_name] = fg
    return fg_dict

# Function to plot and optionally export fitted plots
def plot_and_export_fits(fg_dict, export_format=None, output_dir=None):
    """
    Plot and optionally export the fitted PSDs from FOOOF analysis.
    
    Parameters:
        fg_dict (dict): Dictionary of FOOOFGroup objects.
        export_format (str): Format to export plots (e.g., 'png', 'jpeg', 'svg').
        output_dir (str): Directory to save exported plots.
    """
    for sheet_name, fg in fg_dict.items():
        print(f"Plotting FOOOF results for {sheet_name}...")
        for ind in range(len(fg)):
            # Retrieve individual FOOOFResult for the specific index
            fm = fg.get_fooof(ind=ind)

            # Create a plot for the individual result using matplotlib
            fig, ax = plt.subplots()
            fm.plot(ax=ax, plot_peaks="shade")
            plt.title(f"{sheet_name} - Trace {ind}")

            # Export the plot if an export format is specified
            if export_format and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                file_path = os.path.join(output_dir, f"{sheet_name}_trace_{ind}.{export_format}")
                fig.savefig(file_path, format=export_format, bbox_inches="tight")
                print(f"Exported plot to {file_path}")

            plt.show()
            plt.close(fig)  # Explicitly close the figure to free memory

# GUI Elements
file_picker = widgets.Text(
    description="Excel Path:",
    placeholder="Enter the path to the Excel file",
    layout=widgets.Layout(width="70%")
)

freq_range_min = widgets.FloatText(description="Freq Min:", value=4.0, layout=widgets.Layout(width="30%"))
freq_range_max = widgets.FloatText(description="Freq Max:", value=45.0, layout=widgets.Layout(width="30%"))
peak_width_min = widgets.FloatText(description="Peak Min (Hz):", value=2.0, layout=widgets.Layout(width="50%"))
peak_width_max = widgets.FloatText(description="Peak Max (Hz):", value=10.0, layout=widgets.Layout(width="50%"))
amp_threshold = widgets.FloatText(description="Amplitude Threshold:", value=0.2, layout=widgets.Layout(width="50%"))
r2_threshold = widgets.FloatText(description="R² Threshold:", value=0.5, layout=widgets.Layout(width="50%"))
max_peaks = widgets.IntText(description="Max Peaks:", value=2, layout=widgets.Layout(width="50%"))
fitting_mode = widgets.Dropdown(description="Fitting Mode:", options=["knee", "fixed"], value="knee", layout=widgets.Layout(width="50%"))
use_means = widgets.Checkbox(description="Use Means (2 Sheets)", value=False, layout=widgets.Layout(width="50%"))
export_format = widgets.Dropdown(description="Export Format:", options=[None, "png", "jpeg", "svg"], value=None, layout=widgets.Layout(width="50%"))
output_dir = widgets.Text(
    description="Output Dir:",
    placeholder="Enter output directory for plots",
    layout=widgets.Layout(width="70%")
)
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
        psd_data_dict = process_excel(file_path, use_means=use_means.value)

        if psd_data_dict:
            print("PSD data processed successfully. Running FOOOF analysis...")
            freq_range = [freq_range_min.value, freq_range_max.value]
            peak_width_limits = [peak_width_min.value, peak_width_max.value]
            fg_dict = run_fooof_analysis(
                psd_data_dict,
                freq_range=freq_range,
                peak_width_limits=peak_width_limits,
                amp_threshold=amp_threshold.value,
                r2_threshold=r2_threshold.value,
                max_peaks=max_peaks.value,
                fitting_mode=fitting_mode.value,
            )

            print("Plotting and exporting results...")
            plot_and_export_fits(
                fg_dict,
                export_format=export_format.value,
                output_dir=output_dir.value,
            )
        else:
            print("Processing failed. Please check the file and try again.")

run_button.on_click(on_run_button_click)

# Display the GUI
display(
    widgets.VBox([
        file_picker,
        widgets.HBox([freq_range_min, freq_range_max]),
        widgets.HBox([peak_width_min, peak_width_max]),
        amp_threshold,
        r2_threshold,
        max_peaks,
        fitting_mode,
        use_means,
        export_format,
        output_dir,
        run_button,
        output
    ])
)
