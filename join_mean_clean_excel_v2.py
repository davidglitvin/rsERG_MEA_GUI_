import os
import pandas as pd
from io import BytesIO  # Import BytesIO
import ipywidgets as widgets
from ipywidgets import FileUpload, Button, Output, VBox, Label
from IPython.display import display
import re

# Function to parse filename and map components
def parse_filename(file_name):
    """
    Parses the source file name to extract components and map them to abbreviations.
    
    Examples:
        "rd1_121_chronic_week1_10min_mean" -> ["rd1_121", "Chr", "W1"]
        "wt_116_pilot_10min_mean" -> ["wt_116", "Pt"]
        "rd1_56_pilot_10min_mean" -> ["rd1_56", "Pt"]
    """
    basename = os.path.splitext(file_name)[0]
    components = basename.split('_')
    
    # Initialize parts
    subject_id = ""
    condition = ""
    week = ""
    
    # Extract subject ID (assuming it's the first two components, e.g., "rd1_121" or "wt_116")
    if len(components) >= 2:
        subject_id = f"{components[0]}_{components[1]}"
    
    # Extract condition
    if "chronic" in components:
        condition = "Chr"
    elif "pilot" in components:
        condition = "Pt"
    
    # Extract week
    week_match = re.search(r'week(\d+)', basename, re.IGNORECASE)
    if week_match:
        week = f"W{week_match.group(1)}"
    
    # Combine parts
    parts = [subject_id]
    if condition:
        parts.append(condition)
    if week:
        parts.append(week)
    
    return parts  # e.g., ["rd1_121", "Chr", "W1"]

def parse_sheet_name(sheet_name):
    """
    Parses the sheet name to replace 'block' with 'Bl' and 'eye' with 'Ey',
    and extract their numbers.
    
    Example:
        "block1_eye1" -> ["Bl1", "Ey1"]
    """
    # Replace 'block' with 'Bl' and 'eye' with 'Ey'
    modified = sheet_name.replace("block", "Bl").replace("eye", "Ey")
    
    # Extract block and eye numbers
    # Assuming sheet names are in the format "Bl1_Ey1" or similar
    parts = modified.split('_')
    bl = ""
    ey = ""
    for part in parts:
        if part.startswith("Bl"):
            bl = part
        elif part.startswith("Ey"):
            ey = part
    return [bl, ey]  # e.g., ["Bl1", "Ey1"]

def construct_new_sheet_name(file_parts, sheet_parts):
    """
    Constructs the new sheet name by combining file and sheet parts.
    
    Example:
        ["rd1_121", "Chr", "W1"] and ["Bl1", "Ey1"] -> "rd1_121_Chr_W1_Bl1_Ey1"
        ["wt_116", "Chr"] and ["Bl2", "Ey2"] -> "wt_116_Chr_Bl2_Ey2"
        ["rd1_56", "Pt"] and ["Bl3", "Ey1"] -> "rd1_56_Pt_Bl3_Ey1"
    """
    # Combine all parts with underscores
    combined = '_'.join(file_parts + sheet_parts)
    
    # Ensure the sheet name does not exceed 31 characters
    if len(combined) > 31:
        combined = combined[:31]
    
    return combined

# Function to combine Excel sheets and rename
def combine_sheets(uploaded_files):
    with pd.ExcelWriter("combined_output.xlsx", engine="xlsxwriter") as combined_data:
        for file_name, file_data in uploaded_files.items():
            # Parse the file name
            file_parts = parse_filename(file_name)
            # For example, ["rd1_121", "Chr", "W1"] or ["wt_116", "Pt"]
            
            file_content = file_data["content"]
            # Convert memoryview to BytesIO
            file_bytes = BytesIO(file_content)
            with pd.ExcelFile(file_bytes, engine='openpyxl') as xls:
                for sheet_name in xls.sheet_names:
                    df = xls.parse(sheet_name)
                    
                    # Parse the sheet name
                    sheet_parts = parse_sheet_name(sheet_name)
                    # For example, ["Bl1", "Ey1"]
                    
                    # Construct the new sheet name
                    new_sheet_name = construct_new_sheet_name(file_parts, sheet_parts)
                    
                    # Ensure unique sheet names within the workbook
                    original_sheet_name = new_sheet_name
                    counter = 1
                    while new_sheet_name in combined_data.book.sheetnames:
                        # Append a counter to make it unique
                        suffix = f"_{counter}"
                        # Ensure the total length doesn't exceed 31 characters
                        allowed_length = 31 - len(suffix)
                        new_sheet_name = f"{original_sheet_name[:allowed_length]}{suffix}"
                        counter += 1
                    
                    # Write the DataFrame to the combined workbook
                    df.to_excel(combined_data, sheet_name=new_sheet_name, index=False)

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
file_picker_button = widgets.Button(description="Select Excel Files", button_style="info", icon="folder")
output = widgets.Output()
save_button = Button(description="Combine and Save", button_style="success")
status_label = Label(value="Select Excel files to combine.")

selected_files = []

# Helper functions for renaming logic
def parse_filename(file_name):
    """
    Parses the source file name to extract components and map them to abbreviations.
    
    Examples:
        "rd1_121_chronic_week1_10min_mean" -> ["rd1_121", "Chr", "W1"]
        "wt_116_pilot_10min_mean" -> ["wt_116", "Pt"]
        "rd1_56_pilot_10min_mean" -> ["rd1_56", "Pt"]
    """
    basename = os.path.splitext(file_name)[0]
    components = basename.split('_')
    
    # Initialize parts
    subject_id = ""
    condition = ""
    week = ""
    
    # Extract subject ID (assuming it's the first two components, e.g., "rd1_121" or "wt_116")
    if len(components) >= 2:
        subject_id = f"{components[0]}_{components[1]}"
    
    # Extract condition
    if "chronic" in components:
        condition = "Chr"
    elif "pilot" in components:
        condition = "Pt"
    
    # Extract week
    week_match = re.search(r'week(\d+)', basename, re.IGNORECASE)
    if week_match:
        week = f"W{week_match.group(1)}"
    
    # Combine parts
    parts = [subject_id]
    if condition:
        parts.append(condition)
    if week:
        parts.append(week)
    
    return parts  # e.g., ["rd1_121", "Chr", "W1"]

def parse_sheet_name(sheet_name):
    """
    Parses the sheet name to replace 'block' with 'Bl' and 'eye' with 'Ey',
    and extract their numbers.
    
    Example:
        "block1_eye1" -> ["Bl1", "Ey1"]
    """
    # Replace 'block' with 'Bl' and 'eye' with 'Ey'
    modified = sheet_name.replace("block", "Bl").replace("eye", "Ey")
    
    # Extract block and eye numbers
    # Assuming sheet names are in the format "Bl1_Ey1" or similar
    parts = modified.split('_')
    bl = ""
    ey = ""
    for part in parts:
        if part.startswith("Bl"):
            bl = part
        elif part.startswith("Ey"):
            ey = part
    return [bl, ey]  # e.g., ["Bl1", "Ey1"]

def construct_new_sheet_name(file_parts, sheet_parts):
    """
    Constructs the new sheet name by combining file and sheet parts.
    
    Example:
        ["rd1_121", "Chr", "W1"] and ["Bl1", "Ey1"] -> "rd1_121_Chr_W1_Bl1_Ey1"
        ["wt_116", "Chr"] and ["Bl2", "Ey2"] -> "wt_116_Chr_Bl2_Ey2"
        ["rd1_56", "Pt"] and ["Bl3", "Ey1"] -> "rd1_56_Pt_Bl3_Ey1"
    """
    # Combine all parts with underscores
    combined = '_'.join(file_parts + sheet_parts)
    
    # Ensure the sheet name does not exceed 31 characters
    if len(combined) > 31:
        combined = combined[:31]
    
    return combined

# Function to combine Excel sheets and rename
def combine_sheets(uploaded_files):
    with pd.ExcelWriter("combined_output.xlsx", engine="xlsxwriter") as combined_data:
        for file_name, file_data in uploaded_files.items():
            # Parse the file name
            file_parts = parse_filename(file_name)
            # For example, ["rd1_121", "Chr", "W1"] or ["wt_116", "Pt"]
            
            file_content = file_data["content"]
            # Convert memoryview to BytesIO
            file_bytes = BytesIO(file_content)
            with pd.ExcelFile(file_bytes, engine='openpyxl') as xls:
                for sheet_name in xls.sheet_names:
                    df = xls.parse(sheet_name)
                    
                    # Parse the sheet name
                    sheet_parts = parse_sheet_name(sheet_name)
                    # For example, ["Bl1", "Ey1"]
                    
                    # Construct the new sheet name
                    new_sheet_name = construct_new_sheet_name(file_parts, sheet_parts)
                    
                    # Ensure unique sheet names within the workbook
                    original_sheet_name = new_sheet_name
                    counter = 1
                    while new_sheet_name in combined_data.book.sheetnames:
                        # Append a counter to make it unique
                        suffix = f"_{counter}"
                        # Ensure the total length doesn't exceed 31 characters
                        allowed_length = 31 - len(suffix)
                        new_sheet_name = f"{original_sheet_name[:allowed_length]}{suffix}"
                        counter += 1
                    
                    # Write the DataFrame to the combined workbook
                    df.to_excel(combined_data, sheet_name=new_sheet_name, index=False)

# Save fg_dict to a file (Assuming this is part of your workflow)
import pickle  # For saving and loading fg_dict

def save_fooof_group(fg_dict, filename="fooof_groups.pkl"):
    if fg_dict is None:
        print("Error: No FOOOFGroup dictionary to save. Run the analysis first.")
        return
    with open(filename, "wb") as f:
        pickle.dump(fg_dict, f)
    print(f"FOOOFGroup dictionary saved to {filename}")

# Save function
def on_save_button_click(b):
    if not file_uploader.value:
        status_label.value = "No files selected. Please upload files."
        return

    try:
        # Attempt to handle as a dictionary
        try:
            uploaded_files = {
                name: {"content": file_info['content']} for name, file_info in file_uploader.value.items()
            }
        except AttributeError:
            # If .items() doesn't exist, handle as a list of dicts
            try:
                uploaded_files = {
                    file_info['name']: {"content": file_info['content']} for file_info in file_uploader.value
                }
            except (TypeError, KeyError):
                # Handle as a list of tuples
                uploaded_files = {
                    name: {"content": content} for name, content in file_uploader.value
                }

        combine_sheets(uploaded_files)
        save_fooof_group(None)  # Assuming fg_dict is handled elsewhere
        status_label.value = "Combined Excel file saved as 'combined_output.xlsx'."
    except Exception as e:
        status_label.value = f"An error occurred: {str(e)}"

file_picker_button.on_click(on_save_button_click)

# Display the GUI
ui = VBox([status_label, file_picker_button, save_button, output])
display(ui)

