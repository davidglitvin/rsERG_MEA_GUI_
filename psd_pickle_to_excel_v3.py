import ipywidgets as widgets
from IPython.display import display
import pandas as pd
import numpy as np
import os
import io

# Function to sanitize sheet names for Excel
def sanitize_sheet_name(name):
    """Sanitize the sheet name to make it valid for Excel."""
    invalid_chars = r'[]:*?/\\'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name[:31]  # Excel sheet name limit is 31 characters

# Function to process the data (dictionary) and export to Excel
def process_data(data, output_excel_path):
    # Create an Excel writer
    with pd.ExcelWriter(output_excel_path, engine="xlsxwriter") as writer:
        for key, value in data.items():
            print(f"Processing key: {key}")
            
            if isinstance(value, dict):
                # Extract frequencies and PSD data
                freqs = np.array(value.get("freqs", []))
                psd = np.array(value.get("psd", []))
                
                # Check format
                if len(freqs) == 91 and psd.ndim == 2 and psd.shape[1] == 91:
                    # Create a DataFrame
                    df = pd.DataFrame(
                        psd.T, 
                        columns=[f"PSD_Trace_{i + 1}" for i in range(psd.shape[0])]
                    )
                    df.insert(0, "Frequency (Hz)", freqs)  # Add frequency as the first column
                    
                    # Export to Excel
                    sheet_name = sanitize_sheet_name(key)
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"  Exported key {key} to sheet {sheet_name}")
                else:
                    print(f"  Skipping key {key} due to unexpected data format.")
            else:
                print(f"  Skipping non-dict key: {key}")

    print(f"Data exported successfully to {output_excel_path}")

# Create a file picker widget that accepts multiple files
pkl_file_picker = widgets.FileUpload(
    description="Upload .pkl files", 
    multiple=True,
    accept='.pkl'   # Optionally restrict to .pkl files
)

# Create a button to trigger the processing
process_button = widgets.Button(description="Process Data")

# Output widget to display messages
output = widgets.Output()

# Function to handle button click event
def on_process_button_clicked(b):
    with output:
        output.clear_output()

        if not pkl_file_picker.value:
            print("Please upload at least one .pkl file.")
            return

        # pkl_file_picker.value is a dict whose keys are file names
        # and values are dicts with 'content', 'name', etc.
        for file_name, file_info in pkl_file_picker.value.items():
            print(f"Processing file: {file_name}")

            # Extract file content as bytes
            pkl_file_content = file_info['content']

            # Directly load the pickle data from bytes (no temp file needed)
            data = pd.read_pickle(io.BytesIO(pkl_file_content))

            # Derive the output Excel filename from the .pkl filename
            base_name, _ = os.path.splitext(file_name)
            output_excel_name = base_name + ".xlsx"

            # Process and save to Excel
            process_data(data, output_excel_name)
        
        print("All files have been processed successfully!")

# Attach the button click event handler
process_button.on_click(on_process_button_clicked)

# Display the widgets
display(pkl_file_picker, process_button, output)
