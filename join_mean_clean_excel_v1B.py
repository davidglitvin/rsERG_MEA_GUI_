import os
import re
import pandas as pd
from io import BytesIO
import ipywidgets as widgets
from IPython.display import display

##############################
# 1. HELPER PARSING FUNCTIONS
##############################

def parse_filename(file_name):
    """
    Parses the source file name to extract components and map them to abbreviations.
    
    Examples:
      "rd1_121_chronic_week1_10min_mean.xlsx" -> ["rd1_121", "Chr", "W1"]
      "wt_116_pilot_10min_mean.xlsx"         -> ["wt_116", "Pt"]
      "rd1_56_pilot_10min_mean.xlsx"         -> ["rd1_56", "Pt"]
    """
    basename = os.path.splitext(file_name)[0]
    parts = basename.split('_')

    subject_id = ""
    condition = ""
    week = ""

    # Extract subject_id (first two underscore-separated chunks)
    if len(parts) >= 2:
        subject_id = f"{parts[0]}_{parts[1]}"

    # Identify condition
    if "chronic" in parts:
        condition = "Chr"
    elif "pilot" in parts:
        condition = "Pt"

    # Identify week
    week_match = re.search(r'week(\d+)', basename, re.IGNORECASE)
    if week_match:
        week = f"W{week_match.group(1)}"

    # Build the final list (only include non-empty parts)
    name_components = [comp for comp in [subject_id, condition, week] if comp]
    return name_components

def parse_sheet_name(sheet_name):
    """
    Parses the sheet name to find 'block(\d+)' -> 'B#' and 'eye(\d+)' -> 'Ey#'.
    
    Examples:
      "block1_eye1" -> ["B1", "Ey1"]
      "block2_eye2" -> ["B2", "Ey2"]
    """
    lower_name = sheet_name.lower()

    # Match block(\d+)
    block_match = re.search(r'block(\d+)', lower_name)
    block_str = f"B{block_match.group(1)}" if block_match else ""

    # Match eye(\d+)
    eye_match = re.search(r'eye(\d+)', lower_name)
    eye_str = f"Ey{eye_match.group(1)}" if eye_match else ""

    # Filter out empty strings
    sheet_parts = [x for x in [block_str, eye_str] if x]
    return sheet_parts

def construct_new_sheet_name(file_parts, sheet_parts):
    """
    Constructs the new sheet name (e.g., "rd1_121_Chr_W1_B1_Ey1")
    and ensures it doesn't exceed Excel's 31-character limit.
    """
    combined = "_".join(file_parts + sheet_parts)
    return combined[:31] if len(combined) > 31 else combined

##########################
# 2. COMBINE & RENAME LOGIC
##########################

def combine_sheets(uploaded_files):
    """
    Combines sheets from multiple Excel files into a single Excel workbook
    with the custom naming logic for sheet names.
    """
    with pd.ExcelWriter("combined_output.xlsx", engine="xlsxwriter") as writer:
        for file_name, file_info in uploaded_files.items():
            # Parse the file name
            file_parts = parse_filename(file_name)
            
            # Convert the raw file content (memoryview or bytes) into a BytesIO for pandas
            file_bytes = BytesIO(file_info["content"])

            # Read the Excel file using openpyxl engine (works best with .xlsx)
            with pd.ExcelFile(file_bytes, engine='openpyxl') as xls:
                for original_sheet_name in xls.sheet_names:
                    df = xls.parse(original_sheet_name)

                    # Parse the sheet name for "blockX" and "eyeX"
                    sheet_parts = parse_sheet_name(original_sheet_name)

                    # Construct the final new sheet name
                    new_sheet_name = construct_new_sheet_name(file_parts, sheet_parts)

                    # Ensure uniqueness if there's a conflict
                    base_name = new_sheet_name
                    counter = 1
                    while new_sheet_name in writer.book.sheetnames:
                        suffix = f"_{counter}"
                        max_len = 31 - len(suffix)
                        new_sheet_name = f"{base_name[:max_len]}{suffix}"
                        counter += 1

                    # Write to the combined Excel
                    df.to_excel(writer, sheet_name=new_sheet_name, index=False)

##################
# 3. LAUNCH THE GUI
##################

def launch_gui():
    file_uploader = widgets.FileUpload(accept=".xlsx", multiple=True)
    output = widgets.Output()
    save_button = widgets.Button(description="Combine and Save", button_style="success")
    status_label = widgets.Label(value="Select Excel files to combine.")

    def on_save_button_click(_):
        # 1. Check if files are uploaded
        if not file_uploader.value:
            status_label.value = "No files selected. Please upload files."
            return
        
        try:
            # 2. Build a dictionary of uploaded files, handling various structures
            #    Sometimes file_uploader.value is a dict, sometimes it's a list/tuple.
            uploaded_files = {}
            
            # Try standard approach: file_uploader.value.items() if it's a dict
            try:
                # If this works, we have a dict from file_uploader.value
                for name, file_info in file_uploader.value.items():
                    # .content is the file's raw data (memoryview)
                    uploaded_files[name] = {"content": file_info.content}
            except AttributeError:
                # Otherwise, fallback: it might be a list or tuple
                if isinstance(file_uploader.value, (list, tuple)):
                    for entry in file_uploader.value:
                        # entry could be a dict like {'name':..., 'content':...}
                        # or a tuple like (filename, content)
                        if isinstance(entry, dict):
                            fname = entry.get('name')
                            c = entry.get('content')
                            if fname and c:
                                uploaded_files[fname] = {"content": c}
                        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                            fname, c = entry
                            uploaded_files[fname] = {"content": c}
                else:
                    # Unexpected structure
                    raise ValueError("Unsupported structure from file_uploader.value")

            # 3. Combine and save
            combine_sheets(uploaded_files)
            status_label.value = "Combined Excel file saved as 'combined_output.xlsx'."
        except Exception as e:
            status_label.value = f"An error occurred: {str(e)}"

    save_button.on_click(on_save_button_click)

    ui = widgets.VBox([status_label, file_uploader, save_button, output])
    display(ui)

# If running standalone, call launch_gui()
if __name__ == "__main__":
    launch_gui()
