
# EyeAvg NaN Artifact-Cleaning and PSD Pipeline

This tutorial describes the full pipeline for processing your rsERG EyeAvg data using the following modules:

1. `manual_segment_eyeavg_click_popup_v10.py`  
2. `eyeavg_nan_viewer_popup_v1.py`  
3. `eyeavg_nan_to_BPF_fif_v1.py`  
4. `psd_compute_NaN.py`

The examples below assume that these files are all located in:

```bash
/Users/davidlitvin/Multielectrode array rsERG/Updated_MEA_Pipeline
```

If they are currently in a different folder (e.g. `/Users/.../Desktop/Artifact Dropper/`), either move them into the pipeline folder or adjust the paths accordingly.

---

## 1. Open a Jupyter **Terminal** and `cd` into your pipeline folder

1. Launch **JupyterLab** in your usual way (inside your rsERG virtual environment).
2. In JupyterLab, go to:  
   **File → New → Terminal**
3. In the new terminal, move into the folder that holds your modules:

```bash
cd "/Users/davidlitvin/Multielectrode array rsERG/Updated_MEA_Pipeline"
```

You can verify the contents with:

```bash
ls
```

You should see something like:

```text
manual_segment_eyeavg_click_popup_v10.py
eyeavg_nan_viewer_popup_v1.py
eyeavg_nan_to_BPF_fif_v1.py
psd_compute_NaN.py
...
```

---

## 2. Step 1 – Manual artifact marking with Eye averaging

**Goal:**  
Take the original 16‑channel epoched pickle, average Eye1 (ch 0–7) and Eye2 (ch 8–15), interactively mark bad segments, and save a **NaN-marked EyeAvg pickle**, e.g.:

```text
..._EyeAvg_manualNaN_v4.pkl
```

From the Jupyter terminal (still in the pipeline directory), run:

```bash
python manual_segment_eyeavg_click_popup_v10.py \
"/Users/davidlitvin/Desktop/Isoflurane/MNE Array/1_C57_220_Isoflurane0p5per_up_epoched.pkl" \
--sfreq 2000 \
--epoch-len 5 \
--drop-eye1 0 1 2
```

### What this does

- Loads the 16‑channel epoched file (e.g. shape `(240, 16, 10000)` at 2000 Hz).
- Drops Eye1 channels **0, 1, 2** before averaging, so Eye1_avg is built from channels 3–7.
- Computes **Eye1_avg** and **Eye2_avg** as two synthetic channels.
- Opens a Qt popup window that shows the EyeAvg time series:
  - Use **← / →** arrow keys to scroll in time.
  - Use the GUI controls to select Eye1 or Eye2.
  - Click and drag on the trace to mark bad segments.
  - Use the buttons (e.g. *Add segment*, *Undo last*) to manage your artifact list.
- When you click **Save / Export**, it writes a pickle file like:

```text
/Users/davidlitvin/Desktop/Isoflurane/MNE Array/
  1_C57_220_Isoflurane0p5per_up_epoched_EyeAvg_manualNaN_v4.pkl
```

That file is a **3D NumPy array**:

```text
(n_epochs, 2, n_times_epoch)
```

with NaNs marking the segments you selected as artifact.

---

## 3. Step 2 – Visual QC of EyeAvg NaNs (viewer popup)

**Goal:**  
Quickly inspect whether NaNs were inserted in the expected regions for Eye1_avg and Eye2_avg.

From the same terminal (still in `Updated_MEA_Pipeline`), run:

```bash
python eyeavg_nan_viewer_popup_v1.py \
"/Users/davidlitvin/Desktop/Isoflurane/MNE Array/1_C57_220_Isoflurane0p5per_up_epoched_EyeAvg_manualNaN_v4.pkl" \
--sfreq 2000
```

### What you see

A Qt window with:

- A plot of the EyeAvg time series over a fixed window (e.g. 20 s).
- Controls to switch between Eye 1 and Eye 2.
- A readout of the **NaN fraction** in the current window.
- NaN segments visible as:
  - **Gaps** in the black trace, and
  - **Light grey shaded spans** over the corresponding time ranges.

Use this window to confirm that your manual artifact marking makes sense before moving on to spectral analysis.

---

## 4. Step 3 – Band‑pass filter EyeAvg+NaNs to a 2‑channel FIF

Now we convert the EyeAvg+NaN pickle into a **2‑channel MNE Epochs FIF** that is band‑pass filtered but still preserves NaNs.

### 4.1. Setup Python path in a Jupyter notebook

Open a Jupyter **notebook** (not the terminal) and run:

```python
import sys
sys.path.append("/Users/davidlitvin/Multielectrode array rsERG/Updated_MEA_Pipeline")
```

This allows Python to import your custom modules from that directory.

### 4.2. Run the NaN‑aware band‑pass filter

In the same notebook, run:

```python
from eyeavg_nan_to_BPF_fif_v1 import filter_eyeavg_nan_to_fif

input_pkl = r"/Users/davidlitvin/Desktop/Isoflurane/MNE Array/1_C57_220_Isoflurane0p5per_up_epoched_EyeAvg_manualNaN_v4.pkl"
output_fif = r"/Users/davidlitvin/Desktop/Isoflurane/MNE Array/1_C57_220_Isoflurane0p5per_up_epoched_EyeAvg_manualNaN_BPF0p5-45.fif"

epochs_filt = filter_eyeavg_nan_to_fif(
    input_pkl_path=input_pkl,
    output_fif_path=output_fif,
    sfreq=2000.0,
    l_freq=0.5,      # high‑pass at 0.5 Hz
    h_freq=45.0,     # low‑pass at 45 Hz
    fir_design="firwin",
    filter_length="auto",
    overwrite=True,
    verbose=True,
)
```

### 4.3. What this function does

`filter_eyeavg_nan_to_fif`:

1. Loads the EyeAvg+NaN pickle (shape `(n_epochs, 2, n_times_epoch)`).
2. Flattens to continuous form `(2, n_epochs * n_times_epoch)`.
3. Applies a **NaN‑aware band‑pass filter** separately to Eye1 and Eye2:
   - For each channel, find contiguous finite segments.
   - Filter each finite segment with `mne.filter.filter_data`.
   - Keep NaNs at their original positions.
4. Reshapes back to `(n_epochs, 2, n_times_epoch)`.
5. Wraps the result as an `mne.EpochsArray` with channels:
   - `"Eye1_avg"`
   - `"Eye2_avg"`
6. Saves the epochs to the specified FIF file (`output_fif`).

### 4.4. Quick sanity check of the FIF

You can verify the result with:

```python
import mne
import numpy as np

epochs = mne.read_epochs(output_fif, preload=True)
data = epochs.get_data()

print(epochs)
print("Data shape:", data.shape)           # Expect (240, 2, 10000)
print("Any NaNs?", np.isnan(data).any())  # Should be True
print("sfreq:", epochs.info['sfreq'])     # Should be 2000.0
print("Channels:", epochs.ch_names)       # ['Eye1_avg', 'Eye2_avg']
```

At this point you have a **cleaned, EyeAvg, NaN‑aware, band‑pass filtered FIF** ready for spectral analysis.

---

## 5. Step 4 – NaN‑aware PSD computation and visualization (Jupyter GUI)

The final stage is to compute Welch PSD while respecting NaNs and to visualize it via a Jupyter GUI. You saved this into:

```text
/Users/davidlitvin/Multielectrode array rsERG/Updated_MEA_Pipeline/psd_compute_NaN.py
```

This module:

- Defines `welch_longest_clean_segment` (NaN‑aware Welch per epoch).
- Builds ipywidgets UI elements.
- Displays a GUI (`psd_ui`) for:
  - Selecting a filtered FIF file.
  - Setting window length and overlap.
  - Selecting an output PSD pickle.
  - Computing PSD per epoch and channel (NaN‑aware).
  - Plotting PSD as mean±std or individual+mean.

### 5.1. Import and launch the GUI in Jupyter

In a notebook cell (again making sure the path is set):

```python
import sys
sys.path.append("/Users/davidlitvin/Multielectrode array rsERG/Updated_MEA_Pipeline")

import psd_compute_NaN
```

Because `psd_compute_NaN.py` ends with `display(psd_ui)`, simply importing the module will:

- Build all widgets and callbacks.
- Render the full PSD GUI in your notebook.

If instead you added a function at the bottom like:

```python
def launch_psd_nan_gui():
    display(psd_ui)
```

then you would do:

```python
import sys
sys.path.append("/Users/davidlitvin/Multielectrode array rsERG/Updated_MEA_Pipeline")

import psd_compute_NaN
psd_compute_NaN.launch_psd_nan_gui()
```

### 5.2. Using the PSD GUI

Once the GUI appears in your notebook:

#### Step 1 – Select input filtered FIF

- Click the **input .fif** file chooser.
- Navigate to:

  ```text
  /Users/davidlitvin/Desktop/Isoflurane/MNE Array/
  ```

- Select your filtered file, e.g.:

  ```text
  1_C57_220_Isoflurane0p5per_up_epoched_EyeAvg_manualNaN_BPF0p5-45.fif
  ```

#### Step 2 – Set PSD parameters

- `Window (s)` – Welch window length in seconds (e.g. `2.0`).
- `Overlap (%)` – overlap between windows (e.g. `50` %).

Internally, the GUI will compute:

```text
nperseg = window_length_s * sfreq
noverlap = nperseg * (overlap_percent / 100)
```

and it will reject cases where `nperseg > n_times_per_epoch`.

#### Step 3 – Select output PSD pickle

- Click the **output .pkl** chooser.
- Choose a filename and location, e.g.:

  ```text
  /Users/davidlitvin/Desktop/Isoflurane/MNE Array/1_C57_220_Isoflurane0p5per_EyeAvg_PSD_nanaware.pkl
  ```

#### Step 4 – Compute PSD (NaN‑aware)

- Click **“Compute PSD (NaN‑aware)”**.

What happens under the hood:

- For each channel (e.g. `Eye1_avg`, `Eye2_avg`):
  - For each epoch:
    - Extract the **longest contiguous finite segment** (i.e. ignoring NaNs).
    - If that segment has at least `nperseg` samples, compute Welch PSD on that segment.
    - Otherwise, assign NaNs for that epoch’s PSD row.
- All epoch PSDs are stacked into an array:
  - Shape `(n_epochs, n_freqs)` per channel.
- A dictionary is built:

  ```python
  psd_results = {
      "Eye1_avg": {"psd": psd_array_eye1, "freqs": freqs},
      "Eye2_avg": {"psd": psd_array_eye2, "freqs": freqs},
      # etc. for any other channels present
  }
  ```

- `psd_results` is saved to the chosen `.pkl` file.
- The GUI populates the **channel selection** widget with any channels that have valid PSD data.

#### Step 5 – Plot PSD

- In **Select Channels**, choose `Eye1_avg`, `Eye2_avg`, or both.
- Choose a **Plot mode**:
  - **Mean & +Std** – plot the mean PSD with a shaded +1 std band.
  - **Individual + Mean** – plot each epoch’s PSD in grey and the mean in blue.
- Set axis ranges:
  - `X Min (Hz)` / `X Max (Hz)` – e.g. `0`–`45` Hz.
  - `Y Min` / `Y Max` – adjust to taste; the tool tries to auto‑set `Y Max` from the data.
- Click **“Plot PSD”**.

The resulting plots will:

- Respect the NaN masks (since PSDs are computed only on clean segments).
- Provide per‑channel, per‑epoch views of spectral content in your cleaned, EyeAvg data.

---

## 6. Summary of the full pipeline

1. **Manual artifact marking & Eye averaging** (Qt popup, terminal):

   ```bash
   cd "/Users/davidlitvin/Multielectrode array rsERG/Updated_MEA_Pipeline"

   python manual_segment_eyeavg_click_popup_v10.py \
   "/Users/davidlitvin/Desktop/Isoflurane/MNE Array/1_C57_220_Isoflurane0p5per_up_epoched.pkl" \
   --sfreq 2000 --epoch-len 5 --drop-eye1 0 1 2
   ```

   → Produces `..._EyeAvg_manualNaN_v4.pkl`

2. **Visual QC of NaNs** (Qt popup, terminal):

   ```bash
   python eyeavg_nan_viewer_popup_v1.py \
   "/Users/davidlitvin/Desktop/Isoflurane/MNE Array/1_C57_220_Isoflurane0p5per_up_epoched_EyeAvg_manualNaN_v4.pkl" \
   --sfreq 2000
   ```

3. **NaN‑aware band‑pass filtering to FIF** (notebook):

   ```python
   import sys
   sys.path.append("/Users/davidlitvin/Multielectrode array rsERG/Updated_MEA_Pipeline")

   from eyeavg_nan_to_BPF_fif_v1 import filter_eyeavg_nan_to_fif

   input_pkl = r"/Users/davidlitvin/Desktop/Isoflurane/MNE Array/1_C57_220_Isoflurane0p5per_up_epoched_EyeAvg_manualNaN_v4.pkl"
   output_fif = r"/Users/davidlitvin/Desktop/Isoflurane/MNE Array/1_C57_220_Isoflurane0p5per_up_epoched_EyeAvg_manualNaN_BPF0p5-45.fif"

   epochs_filt = filter_eyeavg_nan_to_fif(
       input_pkl_path=input_pkl,
       output_fif_path=output_fif,
       sfreq=2000.0,
       l_freq=0.5,
       h_freq=45.0,
       fir_design="firwin",
       filter_length="auto",
       overwrite=True,
       verbose=True,
   )
   ```

4. **NaN‑aware PSD computation & plotting** (notebook):

   ```python
   import sys
   sys.path.append("/Users/davidlitvin/Multielectrode array rsERG/Updated_MEA_Pipeline")

   import psd_compute_NaN
   # or, if defined:
   # psd_compute_NaN.launch_psd_nan_gui()
   ```

This Markdown file can be dropped directly into a GitHub repo as `EyeAvg_NaN_pipeline_tutorial.md` (or any name you prefer).
