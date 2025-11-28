"""
eyeavg_nan_to_BPF_fif_v1.py

Band-pass filter an Eye-average NaN-cleaned pickle and save as a 2-channel FIF.

Assumes input pickle contains a 3D NumPy array with shape:
    (n_epochs, 2, n_times_epoch)
produced by manual_segment_eyeavg_click_popup_v10.py, with NaNs marking artifacts.

Steps:
- Load the 3D array.
- Flatten to continuous data: (2, n_times_total).
- Apply a NaN-aware band-pass filter separately to Eye1 and Eye2:
    * For each channel, find contiguous segments of finite samples.
    * Filter each finite segment with mne.filter.filter_data.
    * Leave NaNs in place.
- Reshape back to (n_epochs, 2, n_times_epoch).
- Save as an MNE Epochs FIF file with 2 channels:
      "Eye1_avg", "Eye2_avg".
"""

import os
import pickle
from typing import Optional

import numpy as np
import mne


def _nan_aware_filter_2d(
    data: np.ndarray,
    sfreq: float,
    l_freq: Optional[float],
    h_freq: Optional[float],
    fir_design: str = "firwin",
    filter_length: str = "auto",
    verbose: bool = False,
) -> np.ndarray:
    """
    NaN-aware filtering for 2D data (n_channels, n_times).

    For each channel:
      - Find contiguous runs of finite samples.
      - Apply mne.filter.filter_data to each run.
      - Keep NaNs at their original positions.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data (float, may contain NaNs).
    sfreq : float
        Sampling frequency in Hz.
    l_freq, h_freq : float or None
        Low / high cutoff for band-pass (MNE-style).
    fir_design : {'firwin', 'firwin2'}
        FIR design method.
    filter_length : 'auto' or int
        Filter length passed to mne.filter.filter_data.
    verbose : bool
        If True, print some info.

    Returns
    -------
    data_filt : ndarray, shape (n_channels, n_times)
        Filtered data with NaNs preserved.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array (n_channels, n_times), got {data.shape}")

    n_channels, n_times = data.shape
    data_filt = data.copy()

    for ch in range(n_channels):
        x = data[ch]
        isnan = np.isnan(x)
        if isnan.all():
            if verbose:
                print(f"Channel {ch}: all NaN, skipping.")
            continue

        finite_idx = np.where(~isnan)[0]
        if finite_idx.size == 0:
            continue

        # find contiguous runs of finite samples
        starts = [finite_idx[0]]
        ends = []
        for i in range(1, len(finite_idx)):
            if finite_idx[i] != finite_idx[i - 1] + 1:
                ends.append(finite_idx[i - 1])
                starts.append(finite_idx[i])
        ends.append(finite_idx[-1])

        if verbose:
            print(f"Channel {ch}: {len(starts)} finite segment(s).")

        for s, e in zip(starts, ends):
            # segment is [s, e] inclusive
            seg = x[s:e + 1]
            if seg.size < 4:
                # very short segments are not worth filtering; just copy them
                if verbose:
                    print(f"  segment {s}-{e}: too short ({seg.size} samples), skipping filter.")
                continue

            seg_2d = seg[np.newaxis, :]  # shape (1, n_times_seg)

            seg_filt = mne.filter.filter_data(
                seg_2d,
                sfreq=sfreq,
                l_freq=l_freq,
                h_freq=h_freq,
                fir_design=fir_design,
                filter_length=filter_length,
                verbose="error" if not verbose else True,
            )[0]  # back to 1D

            data_filt[ch, s:e + 1] = seg_filt

        # re-apply NaNs explicitly
        data_filt[ch, isnan] = np.nan

    return data_filt


def filter_eyeavg_nan_to_fif(
    input_pkl_path: str,
    output_fif_path: str,
    sfreq: float = 2000.0,
    l_freq: Optional[float] = 0.5,
    h_freq: Optional[float] = 45.0,
    fir_design: str = "firwin",
    filter_length: str = "auto",
    overwrite: bool = True,
    verbose: bool = True,
):
    """
    Load an EyeAvg NaN-cleaned pickle, filter it, and save as a 2-channel FIF.

    Parameters
    ----------
    input_pkl_path : str
        Path to the EyeAvg_manualNaN.pkl file (3D array, (n_epochs, 2, n_times_epoch)).
    output_fif_path : str
        Path where the filtered FIF file should be written.
    sfreq : float
        Sampling frequency in Hz (e.g., 2000.0).
    l_freq : float or None
        Low cutoff frequency in Hz (for band-pass / high-pass). Use None to disable.
    h_freq : float or None
        High cutoff frequency in Hz (for band-pass / low-pass). Use None to disable.
    fir_design : str
        FIR design passed to mne.filter.filter_data (e.g., 'firwin').
    filter_length : str or int
        Filter length, 'auto' or an integer number of taps.
    overwrite : bool
        If True, allow overwriting existing FIF.
    verbose : bool
        If True, print status messages.

    Returns
    -------
    epochs_filt : mne.Epochs
        The filtered Epochs object (2 channels: Eye1_avg, Eye2_avg).
    """
    # -------------------------------
    # Load pickle
    # -------------------------------
    if verbose:
        print(f"Loading EyeAvg NaN pickle: {input_pkl_path}")

    with open(input_pkl_path, "rb") as f:
        arr = pickle.load(f)

    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (n_epochs, 2, n_times), got {arr.shape}")
    n_epochs, n_ch, n_times_epoch = arr.shape
    if n_ch != 2:
        raise ValueError(f"Expected 2 channels (Eye1, Eye2), got {n_ch}")

    if verbose:
        print(f"  Data shape: {arr.shape} (n_epochs, n_channels, n_times_epoch)")
        print(f"  sfreq = {sfreq} Hz, epoch_len = {n_times_epoch / sfreq:.3f} s")

    # -------------------------------
    # Flatten to continuous (2, n_total_times)
    # -------------------------------
    data_cont = arr.transpose(1, 0, 2).reshape(2, n_epochs * n_times_epoch)

    # -------------------------------
    # NaN-aware filtering
    # -------------------------------
    if verbose:
        print("Filtering (NaN-aware)...")
        print(f"  Band-pass: l_freq={l_freq}, h_freq={h_freq}, "
              f"fir_design={fir_design}, filter_length={filter_length}")

    data_filt = _nan_aware_filter_2d(
        data_cont,
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        fir_design=fir_design,
        filter_length=filter_length,
        verbose=verbose,
    )

    # -------------------------------
    # Reshape back to 3D
    # -------------------------------
    data_filt_3d = data_filt.reshape(2, n_epochs, n_times_epoch).transpose(1, 0, 2)

    if verbose:
        print(f"Filtered data reshaped back to {data_filt_3d.shape} (n_epochs, 2, n_times_epoch)")

    # -------------------------------
    # Create MNE Epochs and save
    # -------------------------------
    ch_names = ["Eye1_avg", "Eye2_avg"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    epochs_filt = mne.EpochsArray(data_filt_3d, info, verbose=False)

    out_dir = os.path.dirname(output_fif_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if verbose:
        print(f"Saving filtered FIF to: {output_fif_path}")

    epochs_filt.save(output_fif_path, overwrite=overwrite)

    if verbose:
        print("Done.")

    return epochs_filt
