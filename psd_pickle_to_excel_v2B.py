import ipywidgets as widgets
from IPython.display import display
import pandas as pd
import numpy as np
import os
from ipyfilechooser import FileChooser

##############################################################################
# HELPER FUNCTIONS
##############################################################################

def sanitize_sheet_name(name):
    """Sanitize the sheet name to make it valid for Excel."""
    invalid_chars = r'[]:*?/\\'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name[:31]  # Excel sheet name limit is 31 characters

def process_data(pkl_path, output_excel_path):
    """
    Process a pickle file (assumed to be a dict of PSD data) and export its
    contents to an Excel file. For each key in the pickle:
      - if the value is a dict with 'freqs' and 'psd' keys,
      - and if the data meets the expected dimensions,
    a sheet is created in the Excel file.
    """
    # Load the pickled data using pandas (alternatively, you could use pickle.load)
    data = pd.read_pickle(pkl_path)

    # Create an Excel writer to write multiple sheets
    with pd.ExcelWriter(output_excel_path, engine="xlsxwriter") as writer:
        for key, value in data.items():
            print(f"Processing key: {key}")
            
            if isinstance(value, dict):
                # Extract frequencies and PSD data
                freqs = np.array(value.get("freqs", []))
                psd = np.array(value.get("psd", []))
                
                # Check that the dimensions are as expected
                if len(freqs) == 91 and psd.ndim == 2 and psd.shape[1] == 91:
                    # Create a DataFrame (each column is a PSD trace; frequency in first column)
                    df = pd.DataFrame(psd.T, columns=[f"PSD_Trace_{i + 1}" for i in range(psd.shape[0])])
                    df.insert(0, "Frequency (Hz)", freqs)
                    
                    # Use a sanitized version of the key as the sheet name.
                    sheet_name = sanitize_sheet_name(key)
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"  Exported key '{key}' to sheet '{sheet_name}'.")
                else:
                    print(f"  Skipping key '{key}' due to unexpected data format.")
            else:
                print(f"  Skipping non-dict key: {key}")

    print(f"Data exported successfully to {output_excel_path}")

##############################################################################
# WIDGETS FOR BATCH PROCESSING
##############################################################################

# FileUpload widget to allow multiple pickle files
pkl_file_picker = widgets.FileUpload(
    description="Upload .pkl files",
    accept=".pkl",
    multiple=True
)

# Directory chooser for selecting output directory (using ipyfilechooser)
output_dir_chooser = FileChooser(
    os.getcwd(),
    title='Select Output Directory for Excel Files',
    show_only_dirs=True,
    select_multiple=False
)

# Process button to trigger batch conversion
process_button = widgets.Button(
    description="Process Data",
    button_style="success"
)

# Output widget for logging messages
output = widgets.Output()

##############################################################################
# CALLBACK FUNCTION FOR BATCH PROCESSING
##############################################################################

def on_process_button_clicked(b):
    with output:
        output.clear_output()
        
        # Convert the upload value into a list of files regardless of its type.
        uploaded_files = pkl_file_picker.value
        files_list = []
        if isinstance(uploaded_files, dict):
            files_list = list(uploaded_files.values())
        elif isinstance(uploaded_files, (list, tuple)):
            files_list = list(uploaded_files)
        
        # Check if any files were uploaded
        if not files_list:
            print("Please upload one or more .pkl files.")
            return
        
        # Ensure an output directory has been selected
        if not output_dir_chooser.selected_path or not os.path.isdir(output_dir_chooser.selected_path):
            print("Please select a valid output directory.")
            return
        
        out_dir = output_dir_chooser.selected_path
        
        # Process each uploaded file
        for uploaded_file in files_list:
            if 'content' not in uploaded_file:
                print(f"Skipping file {uploaded_file.get('name', 'unknown')} (no content found).")
                continue
            
            pkl_file_name = uploaded_file.get('name', 'unknown.pkl')
            base_name = os.path.splitext(pkl_file_name)[0]
            output_excel_path = os.path.join(out_dir, f"{base_name}_processed.xlsx")
            
            print(f"\nProcessing file: {pkl_file_name}")
            print(f"Excel output will be: {output_excel_path}")
            
            # Save the uploaded file content to a temporary file
            temp_pkl_path = f"temp_{pkl_file_name}"
            try:
                with open(temp_pkl_path, "wb") as f:
                    f.write(uploaded_file['content'])
                
                # Process the temporary pickle file and export to Excel
                process_data(temp_pkl_path, output_excel_path)
            except Exception as e:
                print(f"Error processing file {pkl_file_name}: {e}")
            finally:
                # Clean up temporary file if it exists
                if os.path.exists(temp_pkl_path):
                    os.remove(temp_pkl_path)
        
        print("\nBatch processing completed.")

# Attach the callback to the process button
process_button.on_click(on_process_button_clicked)

##############################################################################
# DISPLAY THE WIDGETS
##############################################################################

display(widgets.HTML("<h2>Batch PSD to Excel Converter</h2>"))
display(pkl_file_picker)
display(output_dir_chooser)
display(process_button)
display(output)
