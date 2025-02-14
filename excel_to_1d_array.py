import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display

# Function to process the Excel file and compute the PSD data dictionary
def process_excel(file_path):
    try:
        # Load the Excel file
        excel_data = pd.ExcelFile(file_path)
        
        # Initialize a dictionary to store PSD data for each sheet
        psd_data_dict = {}
        freq_data = None

        # Iterate through each sheet in the Excel file
        for sheet_name in excel_data.sheet_names:
            # Read the sheet into a DataFrame
            sheet_df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Extract frequency column
            freq_column = sheet_df.iloc[:, 0]
            if freq_data is None:
                freq_data = freq_column.values  # Save frequency data once

            # Average all PSD columns (ignoring the first column)
            psd_columns = sheet_df.iloc[:, 1:]
            averaged_psd = psd_columns.mean(axis=1)
            
            # Store the PSD data as its own key in the dictionary
            psd_data_dict[sheet_name] = averaged_psd.values

        # Add the shared frequency array to the dictionary
        psd_data_dict["freq"] = freq_data
        
        return psd_data_dict
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# GUI Elements
file_picker = widgets.Text(
    description='Excel Path:',
    placeholder='Enter the path to the Excel file',
    layout=widgets.Layout(width='70%')
)
run_button = widgets.Button(
    description='Process',
    button_style='success',
    icon='check'
)
output = widgets.Output()

# Button click handler
def on_run_button_click(b):
    with output:
        output.clear_output()
        file_path = file_picker.value
        
        if not file_path.lower().endswith('.xlsx'):
            print(f"Error: Please provide a valid Excel file with a '.xlsx' extension.")
            return
        
        print(f"Processing file: {file_path}")
        psd_data_dict = process_excel(file_path)
        
        if psd_data_dict:
            print("PSD Data Dictionary:")
            for key, value in psd_data_dict.items():
                print(f"{key}: {value[:5]} ...")  # Display first 5 elements for brevity
        else:
            print("Processing failed. Please check the file and try again.")

run_button.on_click(on_run_button_click)

# Display the GUI
display(widgets.VBox([file_picker, run_button, output]))
