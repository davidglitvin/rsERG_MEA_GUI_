# multi_psd_viewer.py

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import ipywidgets as widgets
from ipywidgets import (
    VBox,
    HBox,
    Button,
    SelectMultiple,
    Label,
    Output
)
from IPython.display import display, clear_output

try:
    from ipyfilechooser import FileChooser
except ImportError:
    print("ipyfilechooser not found. Install via: pip install ipyfilechooser")


##############################################################################
# 1) Helper Functions
##############################################################################

def load_mean_psd(pickle_path):
    """
    Loads a mean PSD pickle file.
    
    Expects the file to contain a dict with keys like 'Eye1' and 'Eye2', each
    storing an array for freqs and an array for psd.

    Example structure:
       {
         "Eye1": {
             "freqs": <1D np.array>,
             "psd": <1D or 2D np.array>
         },
         "Eye2": {
             "freqs": <same shape>,
             "psd": ...
         },
         ...
       }

    Returns:
        dict or None if load fails
    """
    if not os.path.isfile(pickle_path):
        print(f"File not found: {pickle_path}")
        return None

    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Failed to load {pickle_path}: {e}")
        return None


def plot_eye_data(ax, freqs, psd_data, title="", color="blue", label=""):
    """
    Plots PSD data on a given Axes 'ax'.
    - If psd_data is 1D, plot directly.
    - If 2D, average across axis=0, then plot.
    """
    if psd_data.ndim == 1:
        ax.plot(freqs, psd_data, color=color, label=label)
    else:
        mean_psd = np.mean(psd_data, axis=0)
        ax.plot(freqs, mean_psd, color=color, label=label)

    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (a.u.)")


##############################################################################
# 2) Main Interactive GUI
##############################################################################

def build_multi_psd_gui():
    """
    A GUI that:
      - Lets you select multiple (if ipyfilechooser supports it) PSD pickle files
      - Stores each loaded file with an "include" flag
      - Allows removing or toggling inclusion
      - Plots each "included" file side-by-side (Eye1/Eye2)
      - Combines included files' Eye1/Eye2 data into a single average
    """
    loaded_recordings = {}  # {file_path: {"data": <loaded_dict>, "include": bool}}

    # =========== WIDGETS ===========

    # A) File chooser. If user has older ipyfilechooser, select_multiple may do nothing.
    file_chooser = FileChooser(
        os.getcwd(),
        title='Select PSD Pickle File(s)',
        select_multiple=True
    )
    file_chooser.filter_pattern = ['*.pkl']
    file_chooser.show_only_files = True

    load_button = Button(
        description='Load Selected',
        button_style='info'
    )

    # B) Manage loaded recordings
    recordings_select = SelectMultiple(
        options=[],
        description='Loaded:',
        layout=widgets.Layout(width='400px', height='180px')
    )
    remove_button = Button(
        description='Remove Highlighted',
        button_style='danger'
    )
    toggle_include_button = Button(
        description='Toggle Include',
        button_style=''
    )

    # C) Plot & Combine Buttons
    plot_button = Button(
        description='Plot Included',
        button_style='success'
    )
    combine_button = Button(
        description='Combine Included',
        button_style='warning'
    )

    # D) Output areas
    load_output = Output()
    plot_output = Output()
    combine_output = Output()

    # =========== Helper Functions ===========

    def refresh_recordings_select():
        """Refresh the recordings_select widget to reflect loaded_recordings."""
        labeled_options = []
        for fpath, rec in loaded_recordings.items():
            mark = "[x]" if rec.get("include", False) else "[ ]"
            label = f"{mark} {os.path.basename(fpath)}"
            labeled_options.append(label)
        recordings_select.options = labeled_options

    def get_full_path(label):
        """
        Convert a label like "[x] myfile.pkl" back to the original full path.
        """
        short_name = label[4:].strip()  # remove '[ ] ' or '[x] '
        for fp, info in loaded_recordings.items():
            if os.path.basename(fp) == short_name:
                return fp
        return None

    # =========== CALLBACKS ===========

    # 1) LOAD
    def on_load_button_clicked(b):
        with load_output:
            clear_output()

            # If the user's ipyfilechooser has .selected_paths, use it
            selected_paths = []
            if hasattr(file_chooser, 'selected_paths'):
                selected_paths = file_chooser.selected_paths
            else:
                # fallback for older ipyfilechooser with only .selected
                if file_chooser.selected:
                    selected_paths = [file_chooser.selected]

            if not selected_paths:
                print("No files selected.")
                return

            print(f"Loading {len(selected_paths)} file(s)...")
            loaded_count = 0
            for path in selected_paths:
                if path in loaded_recordings:
                    print(f"Already loaded: {path}")
                    continue
                data = load_mean_psd(path)
                if data is not None:
                    loaded_recordings[path] = {"data": data, "include": True}
                    loaded_count += 1
                else:
                    print(f"Skipping {path}")

            print(f"Loaded {loaded_count} new file(s).")
            refresh_recordings_select()

    load_button.on_click(on_load_button_clicked)

    # 2) REMOVE
    def on_remove_button_clicked(b):
        with load_output:
            clear_output()
            highlighted = recordings_select.value
            if not highlighted:
                print("No recordings highlighted to remove.")
                return

            remove_count = 0
            for label in highlighted:
                path = get_full_path(label)
                if path in loaded_recordings:
                    del loaded_recordings[path]
                    remove_count += 1

            print(f"Removed {remove_count} recording(s).")
            refresh_recordings_select()

    remove_button.on_click(on_remove_button_clicked)

    # 3) TOGGLE INCLUDE
    def on_toggle_include_button_clicked(b):
        with load_output:
            clear_output()
            highlighted = recordings_select.value
            if not highlighted:
                print("No recordings highlighted to toggle.")
                return

            toggled_count = 0
            for label in highlighted:
                path = get_full_path(label)
                if path in loaded_recordings:
                    curr_val = loaded_recordings[path].get("include", True)
                    loaded_recordings[path]["include"] = not curr_val
                    toggled_count += 1

            print(f"Toggled {toggled_count} recording(s).")
            refresh_recordings_select()

    toggle_include_button.on_click(on_toggle_include_button_clicked)

    # 4) PLOT
    def on_plot_button_clicked(b):
        with plot_output:
            clear_output()

            included = [fp for fp, rec in loaded_recordings.items() if rec.get("include", False)]
            if not included:
                print("No recordings included to plot.")
                return

            n_items = len(included)
            fig, axes = plt.subplots(n_items, 2, figsize=(10, 3*n_items), squeeze=False)

            for row_idx, fpath in enumerate(included):
                rec_data = loaded_recordings[fpath]["data"]
                eye1 = rec_data.get("Eye1", {})
                eye2 = rec_data.get("Eye2", {})

                # Eye1
                ax_e1 = axes[row_idx, 0]
                if "freqs" in eye1 and "psd" in eye1:
                    plot_eye_data(ax_e1, eye1["freqs"], eye1["psd"],
                                  title=f"{os.path.basename(fpath)}\nEye1",
                                  color="blue")
                else:
                    ax_e1.text(0.5, 0.5, "No Eye1 data", ha='center', va='center')
                    ax_e1.set_title(os.path.basename(fpath))

                # Eye2
                ax_e2 = axes[row_idx, 1]
                if "freqs" in eye2 and "psd" in eye2:
                    plot_eye_data(ax_e2, eye2["freqs"], eye2["psd"],
                                  title=f"{os.path.basename(fpath)}\nEye2",
                                  color="red")
                else:
                    ax_e2.text(0.5, 0.5, "No Eye2 data", ha='center', va='center')
                    ax_e2.set_title(os.path.basename(fpath))

            plt.tight_layout()
            plt.show()

    plot_button.on_click(on_plot_button_clicked)

    # 5) COMBINE
    def on_combine_button_clicked(b):
        with combine_output:
            clear_output()

            included = [fp for fp, rec in loaded_recordings.items() if rec.get("include", False)]
            if not included:
                print("No recordings included to combine.")
                return

            all_eye1 = []
            all_eye2 = []
            freqs1 = None
            freqs2 = None

            for fpath in included:
                rec_data = loaded_recordings[fpath]["data"]
                eye1 = rec_data.get("Eye1", {})
                eye2 = rec_data.get("Eye2", {})

                # Eye1
                if "freqs" in eye1 and "psd" in eye1:
                    if freqs1 is None:
                        freqs1 = eye1["freqs"]
                    else:
                        if not np.allclose(freqs1, eye1["freqs"]):
                            print(f"WARNING: Eye1 freq mismatch in {fpath}. Combining anyway.")
                    psd = eye1["psd"]
                    if psd.ndim == 2:
                        psd = np.mean(psd, axis=0)
                    all_eye1.append(psd)

                # Eye2
                if "freqs" in eye2 and "psd" in eye2:
                    if freqs2 is None:
                        freqs2 = eye2["freqs"]
                    else:
                        if not np.allclose(freqs2, eye2["freqs"]):
                            print(f"WARNING: Eye2 freq mismatch in {fpath}. Combining anyway.")
                    psd = eye2["psd"]
                    if psd.ndim == 2:
                        psd = np.mean(psd, axis=0)
                    all_eye2.append(psd)

            combined_data = {}
            if all_eye1 and (freqs1 is not None):
                stacked_e1 = np.vstack(all_eye1)
                combined_data["Eye1"] = {
                    "freqs": freqs1,
                    "psd": np.mean(stacked_e1, axis=0)
                }
            if all_eye2 and (freqs2 is not None):
                stacked_e2 = np.vstack(all_eye2)
                combined_data["Eye2"] = {
                    "freqs": freqs2,
                    "psd": np.mean(stacked_e2, axis=0)
                }

            if not combined_data:
                print("No Eye1 or Eye2 data found to combine.")
                return

            print("Combined data from:")
            for fpath in included:
                print("  -", os.path.basename(fpath))

            fig, axes = plt.subplots(1, 2, figsize=(10,4))
            # Eye1
            if "Eye1" in combined_data:
                e1_freqs = combined_data["Eye1"]["freqs"]
                e1_psd = combined_data["Eye1"]["psd"]
                plot_eye_data(axes[0], e1_freqs, e1_psd, title="Combined Eye1", color="blue")
            else:
                axes[0].text(0.5, 0.5, "No Eye1 data", ha='center', va='center')
                axes[0].set_title("Combined Eye1")

            # Eye2
            if "Eye2" in combined_data:
                e2_freqs = combined_data["Eye2"]["freqs"]
                e2_psd = combined_data["Eye2"]["psd"]
                plot_eye_data(axes[1], e2_freqs, e2_psd, title="Combined Eye2", color="red")
            else:
                axes[1].text(0.5, 0.5, "No Eye2 data", ha='center', va='center')
                axes[1].set_title("Combined Eye2")

            plt.tight_layout()
            plt.show()

            global combined_psd_data
            combined_psd_data = combined_data
            print("Stored combined result in global 'combined_psd_data'.")

    combine_button.on_click(on_combine_button_clicked)

    # =========== Layout & Display ===========

    control_box = VBox([
        Label("STEP 1: Choose .pkl File(s) to Load:"),
        file_chooser,
        load_button,
        load_output,

        Label("STEP 2: Manage Loaded Recordings (Remove / Toggle Include)"),
        recordings_select,
        HBox([remove_button, toggle_include_button]),

        Label("STEP 3: Plot all 'included' files side by side (Eye1/Eye2)"),
        plot_button,
        plot_output,

        Label("STEP 4: Combine all 'included' recordings into a single average Eye1/Eye2"),
        combine_button,
        combine_output
    ])

    display(control_box)
