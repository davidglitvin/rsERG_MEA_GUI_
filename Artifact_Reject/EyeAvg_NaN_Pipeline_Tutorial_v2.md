
# Eye-Averaged NaN Cleaning & PSD Pipeline (Updated Tutorial)

This tutorial describes the full **EyeAvg + NaN** pipeline using the modules we’ve developed.  
It now includes the new NaN-aware PSD cleaning and channel-mean tools:

- `psd_clean_v2d3_NaN.py`
- `psd_channel_means_v1b_NaN.py`

The intended workflow is:

1. **Manual artifact marking & NaN insertion** (standalone GUI)
2. **Quick NaN sanity check** (standalone GUI)
3. **Band-pass filter the EyeAvg+NaN data → 2‑ch FIF**
4. **NaN-aware PSD computation and interactive cleaning from FIF**
5. **Final Eye1/Eye2 channel means & export**

You can also still use `psd_compute_NaN.py` as a simpler PSD viewer if you want.

---

## 0. Environment & file locations

All modules live in:

```bash
/Users/davidlitvin/Multielectrode array rsERG/Updated_MEA_Pipeline
```

The original 16‑channel epoched pickle lives, for example, here:

```bash
/Users/davidlitvin/Desktop/Isoflurane/MNE Array/1_C57_220_Isoflurane0p5per_up_epoched.pkl
```

### 0.1. Starting a JupyterLab terminal

In JupyterLab:

1. Open JupyterLab in your browser.
2. Click **File → New Launcher → Terminal**.
3. In the terminal, `cd` into your pipeline folder:

```bash
cd "/Users/davidlitvin/Multielectrode array rsERG/Updated_MEA_Pipeline"
```

You’ll use this terminal to run the standalone GUIs (Qt pop‑ups).

### 0.2. Making the pipeline modules importable in notebooks

In any Jupyter notebook where you call the Python functions, add:

```python
import sys
sys.path.append("/Users/davidlitvin/Multielectrode array rsERG/Updated_MEA_Pipeline")
```

This lets you `import` the custom modules directly.

---

## 1. Manual artifact marking & EyeAvg NaN export

**Module:** `manual_segment_eyeavg_click_popup_v10.py`  
**Input:** 16‑channel epoched `.pkl` (shape `(n_epochs, 16, n_times)`)  
**Output:** Eye-averaged, NaN‑marked `.pkl` (2 channels: Eye1_avg, Eye2_avg)

From the Jupyter **terminal** (not a notebook cell), run for example:

```bash
python manual_segment_eyeavg_click_popup_v10.py \
"/Users/davidlitvin/Desktop/Isoflurane/MNE Array/1_C57_220_Isoflurane0p5per_up_epoched.pkl" \
--sfreq 2000 \
--epoch-len 5 \
--drop-eye1 0 1 2
```

Key points:

- `--sfreq 2000` → sampling rate (Hz).
- `--epoch-len 5` → epoch duration in seconds (5 s → 10,000 samples at 2 kHz).
- `--drop-eye1 0 1 2` → drop channels 0,1,2 from Eye1 before averaging.
- The module:
  - Averages ch 0–7 → **Eye1_avg**.
  - Averages ch 8–15 → **Eye2_avg**.
  - Lets you click **start** and **end** of bad segments per eye and add them.
  - Replaces those segments with **NaNs** in the corresponding EyeAvg channel.
  - Saves an EyeAvg+NaN pickle, e.g.:  
    `1_C57_220_Isoflurane0p5per_up_epoched_EyeAvg_manualNaN_v4.pkl`

You should see messages in the terminal confirming shape `(n_epochs, 2, n_times)` and export path.

---

## 2. Quick NaN viewer (sanity check of manual segments)

**Module:** `eyeavg_nan_viewer_popup_v1.py`  
**Input:** EyeAvg+NaN pickle  
**Output:** Visualization only (no export)

From the Jupyter **terminal**:

```bash
python eyeavg_nan_viewer_popup_v1.py \
"/Users/davidlitvin/Desktop/Isoflurane/MNE Array/1_C57_220_Isoflurane0p5per_up_epoched_EyeAvg_manualNaN_v4.pkl" \
--sfreq 2000 \
--epoch-len 5
```

This pops up a window that lets you:

- Scroll through time.
- Switch between Eye1 and Eye2.
- Confirm that the artifact segments you marked are indeed NaN (gaps / flat missing portions).

If everything looks OK, proceed to filtering.

---

## 3. Band-pass filter EyeAvg+NaN to 2‑channel FIF

**Module:** `eyeavg_nan_to_BPF_fif_v1.py`  
**Input:** EyeAvg+NaN pickle (`(n_epochs, 2, n_times)`)  
**Output:** Filtered Epochs FIF (2 channels) containing NaNs

In a **Jupyter notebook**, with the pipeline folder on `sys.path`:

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
    l_freq=0.5,      # lower cutoff
    h_freq=45.0,     # upper cutoff
    fir_design="firwin",
    filter_length="auto",
    overwrite=True,
    verbose=True,
)
```

Notes:

- The function:
  - Loads the EyeAvg+NaN pickle.
  - Respects NaNs by filtering **only** finite stretches between NaNs.
  - Reassembles filtered data with NaNs preserved.
  - Creates a 2‑channel `mne.Epochs` object and saves it as a FIF file.
- The output file is something like:  
  `..._EyeAvg_manualNaN_BPF0p5-45.fif`  
  (MNE will warn about naming conventions if it doesn’t end with `-epo.fif`, but the file is valid).

This filtered FIF is the starting point for NaN‑aware PSD analysis and cleaning.

---

## 4. NaN-aware PSD from filtered FIF (psd_compute_NaN)

**Module:** `psd_compute_NaN.py`  
**Input:** Filtered 2‑ch FIF with NaNs  
**Output:** PSD pickle (`psd_results`) per channel (optional / simple path)

This GUI is a simpler PSD viewer/exporter; it **also** does NaN‑aware Welch by looking at the longest finite segment per epoch.

In a Jupyter notebook:

```python
import sys
sys.path.append("/Users/davidlitvin/Multielectrode array rsERG/Updated_MEA_Pipeline")

import psd_compute_NaN  # this should define `psd_ui`

from IPython.display import display
display(psd_compute_NaN.psd_ui)
```

Typical usage:

1. **Select filtered FIF**  
   Use the file chooser to point to something like:  
   `/Users/davidlitvin/Desktop/Isoflurane/MNE Array/1_C57_220_Isoflurane0p5per_up_epoched_EyeAvg_manualNaN_BPF0p5-45.fif`
2. Set **window length** (s), **overlap (%)**.
3. Choose an **output .pkl** name.
4. Click **Compute PSD (NaN‑aware)**.
5. Use the **Plot** controls to inspect PSDs per channel (Eye1_avg, Eye2_avg).

This step is optional now that you have the more powerful cleaning GUI (`psd_clean_v2d3_NaN`), but it remains useful as a quick, focused PSD computation tool.

---

## 5. NaN-aware PSD cleaning from FIF (psd_clean_v2d3_NaN)

**Module:** `psd_clean_v2d3_NaN.py`  
**Input:** Filtered 2‑ch FIF with NaNs (`...EyeAvg_manualNaN_BPF0p5-45.fif`)  
**Output:**  
- Figures showing PSDs with kept/excluded epochs highlighted.  
- **Cleaned PSD pickle**, to feed into the channel‑means GUI.

In a Jupyter notebook:

```python
import sys
sys.path.append("/Users/davidlitvin/Multielectrode array rsERG/Updated_MEA_Pipeline")

import psd_clean_v2d3_NaN
psd_clean_v2d3_NaN.launch_psd_clean_nan_gui()
```

### 5.1 Row 1 — Load & compute NaN‑aware PSD

- Use the file chooser to select your filtered Epochs FIF:  
  `..._EyeAvg_manualNaN_BPF0p5-45.fif`
- Decide whether to:
  - **Segment into 6 blocks of 120 epochs each**, or
  - Use **all epochs** as a single block (`Segment into 6 blocks? = No`).
- Set **Welch window length (s)** and **Overlap (%)**.
- Click **Load & Compute PSD**.
  - The module uses NaN‑aware Welch (longest finite segment ≥ `nperseg` per epoch).
  - It builds a PSD dict: key → `{freqs, psd}`, where keys are:
    - `Eye1_avg`, `Eye2_avg`, or
    - `block1:Eye1_avg`, `block2:Eye1_avg`, etc.

### 5.2 Row 2 — Display & exclusion settings

- **Channels/Blocks**: select which keys you want to see (e.g. `Eye1_avg`, `Eye2_avg` or blocks).
- **What to show**:
  - Kept traces (light gray),
  - Excluded traces (red),
  - Original mean (blue),
  - New mean (green, after exclusion).
- **Exclusion parameters**:
  - Low band (e.g. 1–3 Hz) & threshold × mean.
  - Test bands (e.g. 7–21 Hz in 2 Hz steps) & threshold × mean.
  - `#bands over thresh to exclude`: how many test bands must exceed threshold to drop an epoch.
- Click **Plot PSDs** to generate subplots for each selected key.

### 5.3 Row 3 — Export

- **Export figures**: use the figure chooser:
  - `my_psd_plots.png` → saves each figure as `my_psd_plots_fig1.png`, etc.
  - `my_psd_plots.pptx` → packs all current figures into a PPTX (if `python-pptx` available).
- **Export cleaned PSD (.pkl)**:
  - Choose an output path like:
    - `/Users/davidlitvin/Desktop/Isoflurane/MNE Array/psd_cleaned_EyeAvg_nan.pkl`
  - Click **Export cleaned PSD**.
  - This cleaned PSD pickle will be used in the next step.

The cleaned PSD dict has the same key structure, but with **only the kept epochs** retained in `psd` for each key.

---

## 6. Eye1/Eye2 channel means (psd_channel_means_v1b_NaN)

**Module:** `psd_channel_means_v1b_NaN.py`  
**Input:** Cleaned PSD `.pkl` from `psd_clean_v2d3_NaN`  
**Output:**  
- Screening plots of per‑channel means (Eye1/Eye2).  
- Final Eye1/Eye2 **mean‑of‑means** PSD curves.  
- Exportable data (`.pkl` / `.xlsx`) and figures (`.png` / `.pptx`, etc.).

In a Jupyter notebook:

```python
import sys
sys.path.append("/Users/davidlitvin/Multielectrode array rsERG/Updated_MEA_Pipeline")

import psd_channel_means_v1b_NaN
psd_channel_means_v1b_NaN.launch_psd_channel_means_nan_gui()
```

### 6.1 Load cleaned PSD

- Click **Load PSD Pickle**.
- Choose the cleaned PSD `.pkl` created in Step 5, e.g.:  
  `/Users/davidlitvin/Desktop/Isoflurane/MNE Array/psd_cleaned_EyeAvg_nan.pkl`
- Available keys (e.g. `Eye1_avg`, `Eye2_avg`, `block1:Eye1_avg`, ...) populate the **Channels / Keys** box.

The module automatically classifies keys into **Eye1** and **Eye2** based on names like `Eye1_avg`, `Eye2_avg` or `blockX:Eye1_avg`, `blockX:Eye2_avg`.

### 6.2 Plot channel means and screen per Eye group

- Choose which keys to include in the initial screening (default = all).
- Adjust:
  - Whether to show Eye1/Eye2.
  - Whether to plot original vs new means (pre/post PSD exclusion).
  - Low‑band / test‑band thresholds (for an additional PSD‑level exclusion pass).
  - Axis limits and fonts.
- Click **Plot channel means**.
  - You’ll see Eye1/Eye2 screening figures with per‑channel curves in different colors.

### 6.3 Select final channels & compute Eye1/Eye2 group means

- Two `SelectMultiple` widgets appear:
  - **Eye1 channels**: choose the Eye1 keys to include in the final Eye1 group mean.
  - **Eye2 channels**: choose the Eye2 keys to include in the final Eye2 group mean.
- Click **Plot final Eye1/Eye2 group means**.
  - First figure: per‑channel **new** means per eye + black Eye1/Eye2 group mean.
  - Second figure: **“Final Eye1/Eye2 means of means”** (just two curves, Eye1 vs Eye2).

Internally it stores:

- `Eye1_MeanOfMeans`, `Eye1_freqs`
- `Eye2_MeanOfMeans`, `Eye2_freqs`

in a `final_data_dict` that can be exported.

### 6.4 Export figures and final data

Two export sections:

1. **All figures** (screening + final):
   - Use the first export chooser:
     - `group_psd.png` → `group_psd_fig1.png`, `group_psd_fig2.png`, ...
     - `group_psd.pptx` → a PPTX with all the figures (requires `python-pptx`).

2. **Final Eye1/Eye2 means (data & plot)**:
   - Use the second export chooser:
     - `.pkl` → pickled `final_data_dict`.
     - `.xlsx` → Excel table with freqs and Eye1/Eye2 mean curves.
     - `.png`/`.svg` → the “means‑only” figure.
     - `.pptx` → the “means‑only” figure in a PPTX slide.

---

## 7. Summary of the full pipeline

1. **Manual artifact marking & EyeAvg creation**  
   `manual_segment_eyeavg_click_popup_v10.py` → `...EyeAvg_manualNaN_vX.pkl`

2. **Visual NaN sanity check (optional)**  
   `eyeavg_nan_viewer_popup_v1.py`

3. **Band‑pass filter to FIF (2‑ch, NaN‑aware)**  
   `eyeavg_nan_to_BPF_fif_v1.py` → `...EyeAvg_manualNaN_BPF0p5-45.fif`

4. **NaN‑aware PSD cleaning from FIF**  
   `psd_clean_v2d3_NaN.py` → cleaned PSD `.pkl` per key

5. **Eye1/Eye2 group means & export**  
   `psd_channel_means_v1b_NaN.py` → final Eye1/Eye2 mean‑of‑means curves + plots

6. (**Optional**) Simple PSD compute from FIF  
   `psd_compute_NaN.py` if you want a lighter PSD GUI in between.

You can drop this `.md` file directly into your GitHub repo (e.g. as `EyeAvg_NaN_Pipeline_Tutorial.md`) so collaborators and reviewers can follow the same steps.
