
# rsERG + EEG MEA Analysis Pipeline

This document walks through the full end‑to‑end pipeline for analysing multichannel rsERG + EEG data, from raw Open Ephys recordings to PSD summaries and aperiodic FOOOF analyses.

The tutorial assumes the following core modules are available in your Python path:

- `file_load_v4_autoXML_chanmap_EEG.py`
- `manual_segment_eyeavg_eeg_click_popup_v16.py`
- `pickle_to_MNE_to_BPF_v6_nanPreserve_0_100HzDefault.py`
- `psd_clean_v8_import_psdCompute_varSfreqSafe_nanWelchLongestSeg_0_100Hz_fix1.py`
- `psd_channel_means_v3b_NaN_eyeEEGExcel_3chHeuristic.py` (imported as `psd_means`)
- `excel_to_1darray_to_FOOOF_v1A11_agg_log_log.py`
- `psd_pkl_epochs_to_FOOOF_aperiodic_v4_binnedSeries_peaksExcel.py` (imported as `fooof_ap`)

---

## 0. Environment and general notes

### Python environment

- Python 3.x (tested in a conda / venv environment).
- Core packages: `numpy`, `scipy`, `pandas`, `matplotlib`, `mne`, `ipywidgets`, `ipyfilechooser`, `tqdm`, `fooof`, `python-pptx`, `PyQt5`.
- All GUI‑type modules (`file_load`, `pickle_to_MNE_to_BPF`, `psd_clean`, `psd_channel_means`, `psd_pkl_epochs_to_FOOOF_aperiodic`) are designed to run inside **Jupyter** notebooks, except the manual artifact GUI which is a standalone Qt app.

In a notebook, make sure the pipeline directory is on your `sys.path`, for example:

```python
import sys
sys.path.append("/Users/davidlitvin/Multielectrode array rsERG/Updated_MEA_Pipeline")
```

or adapt as needed.

### Typical data flow (high level)

1. **Open Ephys continuous.dat → epoched Eye1/Eye2/EEG `.pkl`**  
   `file_load_v4_autoXML_chanmap_EEG.py`

2. **Manual artifact rejection and NaN marking on Eye1/Eye2/EEG average traces**  
   `manual_segment_eyeavg_eeg_click_popup_v16.py`

3. **NaN‑aware band‑pass filtering + conversion to MNE Epochs, export `.fif`**  
   `pickle_to_MNE_to_BPF_v6_nanPreserve_0_100HzDefault.py`

4. **NaN‑aware PSD computation (Welch‑longest‑segment) from `.fif` → cleaned PSD `.pkl`**  
   `psd_clean_v8_import_psdCompute_varSfreqSafe_nanWelchLongestSeg_0_100Hz_fix1.py`

5. **Channel screening + Eye1/Eye2/EEG means of means and Excel export**  
   `psd_channel_means_v3b_NaN_eyeEEGExcel_3chHeuristic.py`

6a. **(Optional) Aggregate FOOOF analysis from Excel PSD files**  
   `excel_to_1darray_to_FOOOF_v1A11_agg_log_log.py`

6b. **Instantaneous (epoch‑wise) FOOOF aperiodic analysis directly from PSD `.pkl`**  
   `psd_pkl_epochs_to_FOOOF_aperiodic_v4_binnedSeries_peaksExcel.py`

---

## 1. From Open Ephys to Eye/EEG epoched `.pkl`

**Module:** `file_load_v4_autoXML_chanmap_EEG.py`  

**Goal:**  
- Automatically find the `settings.xml` in a `Record Node ...` folder.  
- Infer number of channels and sampling rate from the XML.  
- Locate the corresponding `continuous.dat`.  
- Map channels to Eye1 (1–8), Eye2 (9–16), and optional EEG channels (≥17).  
- Epoch the continuous recording into 5‑second epochs and save to a pickle.

### Typical folder layout

Example structure (Open Ephys):

```text
/Path/To/RecordingRoot/
└── Record Node 102/
    ├── settings.xml
    └── experiment1/
        └── recording1/
            └── continuous/
                └── Acquisition_Board-100.acquisition_board/
                    └── continuous.dat
```

### How to run (Jupyter)

In a notebook:

```python
import file_load_v4_autoXML_chanmap_EEG as fl

# This should build and display the ipywidgets GUI
fl.build_file_load_gui()
```

If you prefer running as a script from a terminal:

```bash
cd "/path/to/Updated_MEA_Pipeline"
python file_load_v4_autoXML_chanmap_EEG.py
```

### Core steps in the GUI

1. **Select root directory** containing the `Record Node ...` folder(s).  
2. The module will automatically:
   - Traverse into each `Record Node ...` folder.
   - Parse `settings.xml` for sampling rate and number of channels.
   - Find the matching `continuous.dat` file.
3. **Channel mapping:** choose how many EEG channels (0–8).  
   - Eye1 = first 8 channels.  
   - Eye2 = next 8 channels.  
   - EEG = remaining selected channels (if present).
4. **Epoching:** continuous data are reshaped into 5‑second epochs (using the inferred sampling rate).  
5. **Output:** a `.pkl` file with either:
   - A 3D array `(n_epochs, n_channels, n_times)`; or  
   - A dict of 10‑min blocks: `{"block1": arr, "block2": arr, ...}`.

Keep track of the output path; it is the input for the manual artifact GUI.

---

## 2. Manual artifact segmentation on Eye1/Eye2/EEG

**Module:** `manual_segment_eyeavg_eeg_click_popup_v16.py`  

This is a PyQt5 GUI that lets you scroll through the averaged Eye1, Eye2, and EEG signals and mark artifact segments. It then writes a new `.pkl` with those segments replaced by NaNs (on all underlying channels, not just the averages).

### How to run (terminal)

From a terminal (recommended):

```bash
cd "/Users/davidlitvin/Multielectrode array rsERG/Updated_MEA_Pipeline"

python manual_segment_eyeavg_eeg_click_popup_v16.py     "/path/to/epoched_output_from_file_load.pkl"     --sfreq 2000     --epoch-len 5
```

Adapt `--sfreq` and `--epoch-len` to match your recording (e.g. 2000 Hz, 5s).

### Behavior

- Input formats:
  - 3D array `(n_epochs, n_channels, n_times)`
  - dict of such arrays (`{"block1": ..., "block2": ...}`)
  - 2D array `(n_channels, n_times)`
- Displayed traces:
  - `Eye1_avg = mean(Ch1..Ch8)`
  - `Eye2_avg = mean(Ch9..Ch16)`
  - `EEG_avg = mean(all channels beyond 16)`, or a single EEG channel if only one exists.
- Controls:
  - **Window (s):** horizontal zoom.
  - **Amp zoom:** vertical scaling (large values allowed).
  - **Amp center:** Auto / Zero / Manual.
- You can add, edit, and delete segments per trace.  
- When you save, the module writes a new `.pkl` where all marked segments are set to NaN across the corresponding epochs and channels.

Use this cleaned, NaN‑marked `.pkl` as the input to the MNE/BPF step.

---

## 3. NaN‑preserving band‑pass filter and export to `.fif`

**Module:** `pickle_to_MNE_to_BPF_v6_nanPreserve_0_100HzDefault.py`  

**Goal:**  
- Load the epoched NaN‑marked `.pkl`.  
- Convert to an MNE `EpochsArray`.  
- Apply a NaN‑aware band‑pass filter (default 0–100 Hz).  
- Preserve NaNs throughout filtering.  
- Save the result as a `.fif` file for downstream PSD analysis.

### How to run (Jupyter)

```python
import pickle_to_MNE_to_BPF_v6_nanPreserve_0_100HzDefault as bpf

bpf.build_pickle_to_mne_bpf_gui()
```

### Core steps in the GUI

1. **Load epoched `.pkl`:**
   - Select the NaN‑marked epoched pickle created by the manual segmentation step.
   - The module infers sampling rate where possible; otherwise you set it explicitly.
2. **Filter settings:**
   - Default: **0–100 Hz**, FIR (`fir_design="firwin"`) with NaN‑aware handling.
3. **NaN policy:**
   - Choose **“Preserve NaNs (no interpolation)”** to keep the “holes” from artifacts intact.
4. **Export:**
   - Module writes a filtered MNE `Epochs` object to `.fif` and prints the output path.

The resulting filtered `.fif` is the input to the PSD clean GUI.

---

## 4. PSD computation with NaN‑aware Welch (0–100 Hz)

**Module:** `psd_clean_v8_import_psdCompute_varSfreqSafe_nanWelchLongestSeg_0_100Hz_fix1.py`  

**Goal:**  
Compute epoch‑wise PSDs in a NaN‑aware way using a “longest clean segment” variant of Welch, and export a cleaned PSD `.pkl` suitable for both channel‑means and instantaneous FOOOF analyses.

### How to run (Jupyter)

```python
import psd_clean_v8_import_psdCompute_varSfreqSafe_nanWelchLongestSeg_0_100Hz_fix1 as psd_clean

psd_clean.build_exportable_plot_psd_gui()
```

### Core steps in the GUI

1. **Load filtered `.fif`:**
   - Choose the `.fif` produced by the BPF module.
2. **Segment into blocks (optional):**
   - Option to split the recording into 6 roughly equal blocks (e.g. 6 × 10 min).
   - Otherwise, treat all epochs as a single block.
3. **Welch parameters:**
   - Window length (seconds).  
   - Overlap (%).  
   - Frequency range is clamped to **0–100 Hz** in this version.
4. **NaN‑aware logic:**
   - For each epoch and channel, the code finds the **longest contiguous segment of finite samples** and runs Welch on that segment only. If there is not enough clean data, it fills that epoch’s PSD with NaNs.
5. **Outputs:**
   - A dictionary‑like PSD structure saved to `.pkl`, either directly or nested under `["psd_results"]`. Each key is a channel/block name:
     ```python
     {
       "Ch1": {"freqs": 1D array, "psd": 2D array (n_epochs, n_freqs)},
       ...
     }
     ```
   - Optional figures for screening PSDs by block and/or channel.

This cleaned PSD `.pkl` is used by both the channel‑means GUI and the FOOOF aperiodic GUI.

---

## 5. Eye1/Eye2/EEG channel means and Excel export

**Module:** `psd_channel_means_v3b_NaN_eyeEEGExcel_3chHeuristic.py`  
**Imported as:**

```python
import psd_channel_means_v3b_NaN_eyeEEGExcel_3chHeuristic as psd_means
```

### How to run (Jupyter)

```python
psd_means.launch_psd_channel_means_nan_gui()
# or, equivalently (depending on the version):
# psd_means.build_exportable_plot_psd_gui()
```

### Core capabilities

1. **Load PSD `.pkl`:**
   - Select the cleaned PSD pickle from the previous step.
   - Channel keys are listed (e.g. `Ch1`, `Ch2`, or `Eye1_avg`, etc.).

2. **Channel classification:**
   - Channels are automatically classified as **Eye1**, **Eye2**, or **EEG** based on:
     - Names containing `"eye1"`, `"eye2"`, or `"eeg"`; and/or  
     - Numeric mapping: `Ch1–8 → Eye1`, `Ch9–16 → Eye2`, `Ch≥17 → EEG`.

3. **Screening plots:**
   - For each eye (and EEG when present), you can plot:
     - Original mean PSD per channel (before exclusions).  
     - New mean PSD after exclusion based on band‑limited heuristics.

4. **Heuristic exclusion / selection:**
   - Low‑frequency (e.g. 1–3 Hz) and test bands (e.g. 7–21 Hz in 2‑Hz bins) with thresholds can be used to flag channels.  
   - GUI presents per‑eye lists of channels so you can select which ones to include in the final group means.

5. **Final “means of means” plot:**
   - Plots Eye1, Eye2, and EEG group means on the same axes.  
   - Axis limits and font sizes are configurable.

6. **Export options:**
   - Export figures (PNG, SVG, PPTX).  
   - Export **Excel** containing the frequency axis and means of means:

     ```text
     Col 1: Freq_Hz
     Col 2: Eye1_MeanOfMeans
     Col 3: Eye2_MeanOfMeans
     Col 4: EEG_MeanOfMeans
     ```

These Excel outputs are useful for group‑level summaries and for feeding into the aggregate FOOOF module.

---

## 6a. Aggregate FOOOF from Excel PSDs (optional)

**Module:** `excel_to_1darray_to_FOOOF_v1A11_agg_log_log.py`  

This module is designed to take Excel PSD files (e.g. from `psd_channel_means` or other sources), extract 1D PSD curves, and run FOOOF fits over chosen frequency ranges. It is primarily for **aggregate** rather than **epoch‑wise** analyses.

Typical usage pattern in a notebook (simplified):

```python
import excel_to_1darray_to_FOOOF_v1A11_agg_log_log as fooof_excel

# Check the module header for the exact entry‑point function.
# Depending on the version, you will run something like:
fooof_excel.build_excel_to_fooof_gui()
```

In this GUI you typically:

1. Load one or more Excel files containing PSD columns.  
2. Specify which columns correspond to which conditions or eyes.  
3. Set FOOOF parameters (freq range, peak width limits, aperiodic mode, R² threshold, etc.).  
4. Run FOOOF, review fits, and export parameter tables and plots.

If you are primarily interested in **instantaneous** aperiodic dynamics, use the PSD‑pkl‑based FOOOF module described next.

---

## 6b. Instantaneous aperiodic FOOOF analysis from PSD `.pkl`

**Module:** `psd_pkl_epochs_to_FOOOF_aperiodic_v4_binnedSeries_peaksExcel.py`  
**Imported as:**

```python
import psd_pkl_epochs_to_FOOOF_aperiodic_v4_binnedSeries_peaksExcel as fooof_ap
```

### How to run (Jupyter)

```python
fooof_ap.build_psd_pkl_fooof_aperiodic_gui()
```

### Inputs

- Cleaned PSD `.pkl` produced by `psd_clean_v8_import_psdCompute_varSfreqSafe_nanWelchLongestSeg_0_100Hz_fix1.py`.  
- Structure:
  ```python
  {
      "Ch1": {"freqs": 1D array, "psd": 2D array (n_epochs, n_freqs)},
      ...
  }
  ```
  or nested under `"psd_results"`.

### Core features

1. **Channel/key selection:**
   - Choose which channels/blocks to analyse.

2. **FOOOF parameters:**
   - Frequency range (min / max).  
   - Aperiodic mode (`"fixed"` or `"knee"`).  
   - Peak width limits.  
   - Maximum number of peaks.  
   - Minimum peak amplitude.  
   - R² threshold for accepting fits.

3. **Epoch‑wise FOOOF fitting:**
   - For each selected channel and each epoch:
     - Runs a FOOOF fit on the PSD curve.  
     - Stores aperiodic exponent and offset (and optionally knee).  
     - Applies R² threshold; sub‑threshold fits are set to NaN.
   - Epochs containing NaNs in the PSD are skipped but **flagged** so you can see where data were unusable.

4. **Interactive inspection:**
   - Slider for epoch index: moving the slider automatically updates the per‑epoch FOOOF plot.  
   - Changing any FOOOF parameter widget triggers re‑fitting for the currently inspected channel and updates the plot.  
   - Each epoch’s FOOOF plot title includes exponent, offset, and R².

5. **Aperiodic time‑series plots:**
   - Plots exponent vs epoch index and offset vs epoch index.  
   - Options to **bin** epochs (e.g. average every 2, 4, 6, 12 epochs corresponding to 10, 20, 30, 60 s blocks).  
   - Epochs skipped due to NaNs or poor fits are marked in the plots (e.g. colored placeholders) so you retain a sense of the full temporal structure.

6. **Excel export:**
   - Exports, for each epoch, at least:
     - Aperiodic exponent.  
     - Aperiodic offset.  
     - R² of the fit.  
     - Peak parameters (center frequency, amplitude, bandwidth, etc., depending on your settings).
   - This allows later statistical analysis of instantaneous aperiodic dynamics.

---

## 7. Suggested end‑to‑end workflow summary

1. **Prepare data:** acquire Open Ephys recordings with rsERG + EEG, ensure consistent channel mapping.  
2. **Run file loader:** `file_load_v4_autoXML_chanmap_EEG` → epoched Eye/EEG `.pkl`.  
3. **Manual cleaning:** `manual_segment_eyeavg_eeg_click_popup_v16` → NaN‑marked `.pkl`.  
4. **Filter & export:** `pickle_to_MNE_to_BPF_v6_nanPreserve_0_100HzDefault` → filtered `.fif`.  
5. **Compute PSDs:** `psd_clean_v8_import_psdCompute_varSfreqSafe_nanWelchLongestSeg_0_100Hz_fix1` → cleaned PSD `.pkl`.  
6. **Channel screening & summary:** `psd_channel_means_v3b_NaN_eyeEEGExcel_3chHeuristic` → plots + Excel of Eye1/Eye2/EEG means.  
7a. **Aggregate FOOOF (optional):** `excel_to_1darray_to_FOOOF_v1A11_agg_log_log` on exported Excel files.  
7b. **Instantaneous aperiodic FOOOF:** `psd_pkl_epochs_to_FOOOF_aperiodic_v4_binnedSeries_peaksExcel` → epoch‑wise exponent/offset time‑series and detailed Excel export.

This README can be placed at the root of your repository as `README.md` or inside a dedicated `docs/` folder (e.g. `docs/rsERG_MEA_pipeline_tutorial.md`) and linked from the main README.
