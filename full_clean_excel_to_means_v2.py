import pandas as pd
import ipywidgets as widgets
from IPython.display import display
import os

# Function to process the Excel file and create the output sheets
def process_excel(file_path, output_path, blocks=6, channels_per_eye=8, separate_sheets=False):
    try:
        excel_data = pd.ExcelFile(file_path)

        frequency_col = "Frequency (Hz)"
        sheet1_data = []
        sheet2_data = {frequency_col: None, "eye1": None, "eye2": None}

        # Dictionaries to hold data for separate sheets if needed
        separate_block_sheets = {}
        separate_eye_sheets = {}

        for block in range(1, blocks + 1):
            eye1_psds = []
            eye2_psds = []

            for ch in range(1, 17):
                sheet_name = f"block{block}_Ch{ch}"
                sheet_df = excel_data.parse(sheet_name)

                # Extract frequency and PSD values
                frequency = sheet_df.iloc[:, 0]
                psd_values = sheet_df.iloc[:, 1:]

                # Assign frequency to sheet2 data (only need it once)
                if sheet2_data[frequency_col] is None:
                    sheet2_data[frequency_col] = frequency

                # Separate data by eye
                if ch <= channels_per_eye:
                    eye1_psds.append(psd_values)
                else:
                    eye2_psds.append(psd_values)

            # Average the PSDs for eye1 and eye2 in this block
            avg_eye1 = pd.concat(eye1_psds, axis=1).mean(axis=1)
            avg_eye2 = pd.concat(eye2_psds, axis=1).mean(axis=1)

            # Append to sheet1 data
            sheet1_data.append([f"block{block}_eye1", frequency, avg_eye1])
            sheet1_data.append([f"block{block}_eye2", frequency, avg_eye2])

            # Collect data for separate sheets if needed
            if separate_sheets:
                separate_block_sheets[f"block{block}_eye1"] = pd.DataFrame({frequency_col: frequency, "PSD": avg_eye1})
                separate_block_sheets[f"block{block}_eye2"] = pd.DataFrame({frequency_col: frequency, "PSD": avg_eye2})

            # Aggregate the PSDs for sheet2 (across all blocks)
            if sheet2_data["eye1"] is None:
                sheet2_data["eye1"] = avg_eye1
                sheet2_data["eye2"] = avg_eye2
            else:
                sheet2_data["eye1"] += avg_eye1
                sheet2_data["eye2"] += avg_eye2

        # Average sheet2 data across all blocks
        sheet2_data["eye1"] /= blocks
        sheet2_data["eye2"] /= blocks

        if separate_sheets:
            separate_eye_sheets["eye1"] = pd.DataFrame({frequency_col: sheet2_data[frequency_col], "PSD": sheet2_data["eye1"]})
            separate_eye_sheets["eye2"] = pd.DataFrame({frequency_col: sheet2_data[frequency_col], "PSD": sheet2_data["eye2"]})

        # Create dataframes for output sheets
        sheet1_df = pd.DataFrame(columns=[frequency_col] + [f"block{block}_eye1" for block in range(1, blocks + 1)] +
                                          [f"block{block}_eye2" for block in range(1, blocks + 1)])

        # Combine frequency and PSDs for sheet1
        for i, (_, frequency, psd) in enumerate(sheet1_data):
            sheet1_df.iloc[:, 0] = frequency.values
            sheet1_df.iloc[:, i + 1] = psd.values

        sheet2_df = pd.DataFrame(sheet2_data)

        # Save to a new Excel file
        with pd.ExcelWriter(output_path) as writer:
            if separate_sheets:
                for sheet_name, df in separate_block_sheets.items():
                    df.to_excel(writer, index=False, sheet_name=sheet_name)
                for sheet_name, df in separate_eye_sheets.items():
                    df.to_excel(writer, index=False, sheet_name=sheet_name)
            else:
                sheet1_df.to_excel(writer, index=False, sheet_name="Block_Averages")
                sheet2_df.to_excel(writer, index=False, sheet_name="Aggregate")

        print(f"Processing complete. Data saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# GUI elements
file_picker = widgets.Text(
    description="Input File:",
    placeholder="Enter the path to the input Excel file",
    layout=widgets.Layout(width="80%")
)

output_picker = widgets.Text(
    description="Output File:",
    placeholder="Enter the path to the output Excel file",
    layout=widgets.Layout(width="80%")
)

blocks_widget = widgets.IntText(
    description="Blocks:",
    value=6,
    layout=widgets.Layout(width="50%")
)

channels_per_eye_widget = widgets.IntText(
    description="Channels/Eye:",
    value=8,
    layout=widgets.Layout(width="50%")
)

separate_sheets_widget = widgets.Checkbox(
    description="Separate Sheets",
    value=False,
    layout=widgets.Layout(width="50%")
)

run_button = widgets.Button(
    description="Process Data",
    button_style="success",
    icon="check"
)

output = widgets.Output()

# Button click handler
def on_run_button_click(b):
    with output:
        output.clear_output()
        input_path = file_picker.value
        output_path = output_picker.value
        blocks = blocks_widget.value
        channels_per_eye = channels_per_eye_widget.value
        separate_sheets = separate_sheets_widget.value

        if not os.path.isfile(input_path):
            print("Error: Input file does not exist.")
            return

        if not output_path.endswith(".xlsx"):
            print("Error: Output file must have a .xlsx extension.")
            return

        print(f"Processing file: {input_path}")
        process_excel(input_path, output_path, blocks, channels_per_eye, separate_sheets)

run_button.on_click(on_run_button_click)

# Display the GUI
display(
    widgets.VBox([
        file_picker,
        output_picker,
        widgets.HBox([blocks_widget, channels_per_eye_widget, separate_sheets_widget]),
        run_button,
        output
    ])
)
