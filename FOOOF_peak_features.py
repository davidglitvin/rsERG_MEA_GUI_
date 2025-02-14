import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from collections import defaultdict


# Function to extract FOOOF parameters
def extract_fooof_parameters(fg_dict, freq_range=(4, 45), amplitude_threshold=0.15):
    extracted_data = defaultdict(lambda: {"amplitudes": [], "center_frequencies": [], "fwhms": []})

    for fg_name, fg in fg_dict.items():
        print(f"Extracting peak parameters for FOOOFGroup: {fg_name}")

        num_channels = len(fg.power_spectra)

        for index in range(num_channels):
            fm = fg.get_fooof(ind=index, regenerate=True)

            if fm.has_data:
                for peak in fm.peak_params_:
                    if freq_range[0] <= peak[0] <= freq_range[1] and peak[1] >= amplitude_threshold:
                        extracted_data[fg_name]["amplitudes"].append(peak[1])
                        extracted_data[fg_name]["center_frequencies"].append(peak[0])
                        extracted_data[fg_name]["fwhms"].append(peak[2])

    return extracted_data


# Function to plot bar or line graphs
def plot_fooof_data(extracted_data, plot_type, parameter, group_patterns, x_tick_labels=None):
    grouped_data = defaultdict(list)

    # Organize data into groups based on patterns
    for fg_name, params in extracted_data.items():
        for pattern in group_patterns:
            if pattern in fg_name:
                grouped_data[pattern].append(np.array(params[parameter]))

    # Compute mean and std for each group
    means = []
    stds = []
    for pattern in group_patterns:
        group_values = grouped_data.get(pattern, [])
        if group_values:
            stacked_values = np.vstack(group_values)
            means.append(stacked_values.mean(axis=0))
            stds.append(stacked_values.std(axis=0))
        else:
            means.append([])
            stds.append([])

    # Plotting
    if plot_type == "Bar":
        x = np.arange(len(group_patterns))
        means = [np.mean(m) if len(m) > 0 else 0 for m in means]
        stds = [np.mean(s) if len(s) > 0 else 0 for s in stds]

        plt.bar(x, means, yerr=stds, capsize=5)
        plt.xticks(x, group_patterns, rotation=45, ha="right")
        plt.ylabel(parameter.capitalize())
        plt.title(f"{parameter.capitalize()} (Bar Plot)")
        plt.show()

    elif plot_type == "Line":
        for i, (mean, std) in enumerate(zip(means, stds)):
            x = np.arange(1, len(mean) + 1)
            plt.errorbar(x, mean, yerr=std, label=group_patterns[i], capsize=3)

        if x_tick_labels:
            plt.xticks(np.arange(1, len(x_tick_labels) + 1), x_tick_labels)
        plt.ylabel(parameter.capitalize())
        plt.title(f"{parameter.capitalize()} (Line Plot)")
        plt.legend()
        plt.show()


# GUI
def fooof_gui(fg_dict):
    # Widgets for parameter selection
    parameter_dropdown = widgets.Dropdown(
        options=["amplitudes", "center_frequencies", "fwhms"],
        description="Parameter:",
    )
    plot_type_dropdown = widgets.Dropdown(
        options=["Bar", "Line"],
        description="Plot Type:",
    )
    block_patterns_input = widgets.Text(
        value="block1_eye1_rd1_56, block2_eye1_rd1_56, block3_eye1_rd1_56, block4_eye1_rd1_56, block5_eye1_rd1_56, block6_eye1_rd1_56",
        description="Block Patterns:",
        layout=widgets.Layout(width="95%"),
    )
    eye_patterns_input = widgets.Text(
        value="eye1_rd1_56, eye2_rd1_56",
        description="Eye Patterns:",
        layout=widgets.Layout(width="95%"),
    )
    run_button = widgets.Button(description="Run", button_style="success", icon="check")
    output = widgets.Output()

    # Button click handler
    def on_run_button_click(b):
        with output:
            output.clear_output()

            parameter = parameter_dropdown.value
            plot_type = plot_type_dropdown.value
            block_patterns = block_patterns_input.value.split(", ")
            eye_patterns = eye_patterns_input.value.split(", ")

            # Extract data from FOOOFGroup objects
            print("Extracting data...")
            extracted_data = extract_fooof_parameters(fg_dict)

            # Plot block patterns
            if block_patterns:
                print(f"Plotting for block patterns: {block_patterns}")
                plot_fooof_data(
                    extracted_data,
                    plot_type,
                    parameter,
                    block_patterns,
                    x_tick_labels=["Block1", "Block2", "Block3", "Block4", "Block5", "Block6"]
                    if plot_type == "Line"
                    else None,
                )

            # Plot eye patterns
            if eye_patterns:
                print(f"Plotting for eye patterns: {eye_patterns}")
                plot_fooof_data(extracted_data, "Bar", parameter, eye_patterns)

    run_button.on_click(on_run_button_click)

    # Display GUI
    display(
        widgets.VBox(
            [
                parameter_dropdown,
                plot_type_dropdown,
                block_patterns_input,
                eye_patterns_input,
                run_button,
                output,
            ]
        )
    )


# Example Usage
# To launch the GUI, call this function and pass your fg_dict:
fooof_gui(fg_dict)
