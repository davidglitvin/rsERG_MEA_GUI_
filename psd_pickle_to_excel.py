import ipywidgets as widgets
from IPython.display import display
import pandas as pd
import numpy as np
import os

# Function to sanitize sheet names for Excel
def sanitize_sheet_name(name):
    """Sanitize the sheet name to make it valid for Excel."""
    invalid_chars = r'[]:*?/\\'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name[:31]  # Excel sheet name limit is 31 characters

# Function to process the data and export to Excel
def process_data(pkl_path, output_excel_path):
    # Load the pickled data
    data = pd.read_pickle(pkl_path)

    # Create an Excel writer
    with pd.ExcelWriter(output_excel_path, engine="xlsxwriter") as writer:
        for key, value in data.items():
            print(f"Processing key: {key}")
            
            if isinstance(value, dict):
                # Extract frequencies and PSD data
                freqs = np.array(value.get("freqs", []))
                psd = np.array(value.get("psd", []))
                
                if len(freqs) == 91 and psd.ndim == 2 and psd.shape[1] == 91:
                    # Create a DataFrame
                    df = pd.DataFrame(psd.T, columns=[f"PSD_Trace_{i + 1}" for i in range(psd.shape[0])])
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

# Create file picker widgets
pkl_file_picker = widgets.FileUpload(description="Upload .pkl file", multiple=False)
output_file_picker = widgets.Text(description="Output .xlsx path:", value="output.xlsx")

# Create a button to trigger the processing
process_button = widgets.Button(description="Process Data")

# Output widget to display messages
output = widgets.Output()

# Function to handle button click event
def on_process_button_clicked(b):
    with output:
        output.clear_output()
        if len(pkl_file_picker.value) == 0:
            print("Please upload a .pkl file.")
            return
        
        # Get the uploaded file content
        uploaded_file = list(pkl_file_picker.value.values())[0]  # Get the first uploaded file
        pkl_file_content = uploaded_file['content']  # Access the file content
        
        # Save the uploaded file to a temporary file
        temp_pkl_path = "temp.pkl"
        with open(temp_pkl_path, "wb") as f:
            f.write(pkl_file_content)
        
        # Process the data
        process_data(temp_pkl_path, output_file_picker.value)
        
        # Clean up the temporary file
        os.remove(temp_pkl_path)

# Attach the button click event handler
process_button.on_click(on_process_button_clicked)

# Display the widgets
display(pkl_file_picker, output_file_picker, process_button, output)