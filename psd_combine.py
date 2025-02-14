import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np

class PSDCombiner:
    def __init__(self):
        self.psd_files = []
        self.combined_data = {}

    def load_psd_files(self, file_paths):
        """
        Load PSD files from the provided file paths.

        Parameters:
        - file_paths (list): List of file paths to PSD files.
        """
        self.psd_files.extend(file_paths)

    def combine_psds(self):
        """
        Combine PSD data from loaded files.
        """
        if not self.psd_files:
            raise ValueError("No PSD files loaded!")

        for file in self.psd_files:
            psd_data = pd.read_pickle(file)
            for key, value in psd_data.items():
                if key not in self.combined_data:
                    self.combined_data[key] = []
                self.combined_data[key].append(value)

        # Convert lists to numpy arrays
        for key in self.combined_data:
            self.combined_data[key] = np.array(self.combined_data[key])

    def export_combined_data(self, export_path):
        """
        Export the combined PSD data to a pickle file.

        Parameters:
        - export_path (str): Path to save the combined data.
        """
        if not self.combined_data:
            raise ValueError("No combined data to export!")

        with open(export_path, 'wb') as f:
            pd.to_pickle(self.combined_data, f)

# Example usage:
if __name__ == "__main__":
    combiner = PSDCombiner()

    # Example file paths
    file_paths = [
        "D:\\Multi-electrode-array ERG\\Rd1 First Set\\Analysis\\PSD Clean\\rd1_60_PSD_cleaned.pkl",
        "D:\\Multi-electrode-array ERG\\Rd1 First Set\\Analysis\\PSD Clean\\rd1_59_PSD_cleaned.pkl",
        "D:\\Multi-electrode-array ERG\\Rd1 First Set\\Analysis\\PSD Clean\\rd1_56_PSD_cleaned.pkl",
        "D:\\Multi-electrode-array ERG\\Chronic\\Analysis\\PSD Clean\\wt_116_chronic_week1_PSD_clean_MyPSDExport_cleaned.pkl",
        "D:\\Multi-electrode-array ERG\\Rd1 First Set\\Analysis\\PSD Clean\\rd1_58_PSD_cleaned.pkl"
    ]

    # Load files
    combiner.load_psd_files(file_paths)

    # Combine data
    combiner.combine_psds()

    # Export combined data
    combiner.export_combined_data("D:\\combined_psd_data.pkl")