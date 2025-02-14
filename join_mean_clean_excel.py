import os
import pandas as pd
from ipywidgets import FileUpload, Button, Output, VBox, Label
from IPython.display import display

# Function to launch the GUI
def launch_gui():
    # Create widgets
    file_uploader = FileUpload(accept=".xlsx", multiple=True)
    output = Output()
    save_button = Button(description="Combine and Save", button_style="success")
    status_label = Label(value="Select Excel files to combine.")

    # Function to combine Excel sheets and rename
    def combine_sheets(uploaded_files):
        with pd.ExcelWriter("combined_output.xlsx", engine="xlsxwriter") as combined_data:
            for file_name, file_data in uploaded_files.items():
                prefix = os.path.splitext(file_name)[0]

                file_content = file_data["content"]
                with pd.ExcelFile(file_content) as xls:
                    for sheet_name in xls.sheet_names:
                        df = xls.parse(sheet_name)

                        # Replace 'block' with 'Bl' and 'eye' with 'Ey' in sheet names
                        modified_sheet_name = sheet_name.replace("block", "Bl").replace("eye", "Ey")
                        new_sheet_name = f"{modified_sheet_name}_{prefix}"

                        # Limit sheet name to Excel's 31-character limit
                        new_sheet_name = new_sheet_name[:31]

                        df.to_excel(combined_data, sheet_name=new_sheet_name, index=False)

    # Save function
    def on_save_button_click(b):
        if not file_uploader.value:
            status_label.value = "No files selected. Please upload files."
            return

        try:
            uploaded_files = {
                name: {"content": content.content} for name, content in file_uploader.value.items()
            }
            combine_sheets(uploaded_files)
            status_label.value = "Combined Excel file saved as 'combined_output.xlsx'."
        except Exception as e:
            status_label.value = f"An error occurred: {str(e)}"

    save_button.on_click(on_save_button_click)

    # Display widgets
    ui = VBox([status_label, file_uploader, save_button, output])
    display(ui)

# Main guard to prevent automatic execution on import
if __name__ == "__main__":
    launch_gui()
