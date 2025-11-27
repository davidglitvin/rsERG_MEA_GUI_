"""artifact_excision_nanmark_5s_v1.py

Variant of the artifact excision pipeline that *preserves* temporal structure
by replacing artifact-contaminated samples with NaNs instead of excising them.

This module reuses the loading and artifact-detection logic from
`artifact_excision_resegment_5s_v1.py`, but changes the cleaning step:

    - Original: remove (excise) all samples where any artifact criterion (i–iv)
      is True, reconnect the remaining data, then resegment into 5 s epochs.

    - This module: keep the continuous data intact, but set all samples where
      any artifact criterion (i–iv) is True to NaN. Then we cut the data into
      consecutive 5 s epochs (dropping only an incomplete tail, if any) and
      export as an MNE Epochs FIF file.

This is useful when you want to preserve the original time base (e.g., for
synchrony across modalities or to avoid distorting temporal statistics),
while still being able to mask out contaminated segments in downstream
analyses.

IMPORTANT
---------
- Filtering cannot be applied *across* NaNs. Therefore this pipeline applies
  the band-pass filter on the continuous data first (without NaNs), then
  replaces artifact samples with NaNs *after* filtering. This preserves
  the temporal structure but does not prevent filter ringing around artifacts.
- If you need perfectly local filtering that does not “see” artifacts,
  consider the excision-based pipeline instead.

Usage
-----
    from artifact_excision_nanmark_5s_v1 import nanmark_and_export

    epochs_nan = nanmark_and_export(
        input_pkl_path=r"...your_file.pkl",
        output_fif_path=r"...optional_output.fif",
        sfreq=2000.0,
        epoch_len_s=5.0,
        amp_z_hard=7.0,
        deriv_z_hard=7.0,
        drift_z_thresh=7.0,
        intercept_z_thresh=7.0,
        min_dur_ms_amp=10.0,
        min_dur_ms_deriv=8.0,
        l_freq=0.5,
        h_freq=100.0,
    )

    # epochs_nan.get_data() will have shape (n_epochs, n_channels, n_times)
    # and contain NaNs at artifact time points.
"""

import os
from typing import Optional

import numpy as np
import mne

from artifact_excision_resegment_5s_v1 import (  # type: ignore
    load_pickle_as_continuous,
    build_artifact_mask_i_to_iv,
)


def _unique_output_path(
    base_dir: str,
    base_name: str,
    suffix: str = "_nan5s_filtered.fif",
) -> str:
    """Create a unique FIF output path in base_dir using base_name + suffix."""
    root = os.path.splitext(base_name)[0]
    candidate = os.path.join(base_dir, root + suffix)
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(base_dir, f"{root}_nan5s_filtered_{counter}.fif")
        counter += 1
    return candidate


def _nanmark_and_epoch(
    data_cont: np.ndarray,
    artifact_mask: np.ndarray,
    epoch_len_samples: int,
) -> np.ndarray:
    """Apply NaNs at artifact samples and cut into consecutive 5 s epochs.

    Parameters
    ----------
    data_cont : ndarray, shape (n_channels, n_times)
        Filtered continuous data (no NaNs yet).
    artifact_mask : ndarray, shape (n_times,)
        Boolean mask where True indicates an artifact sample (any criterion).
    epoch_len_samples : int
        Epoch length in samples, e.g. 5 s * sfreq.

    Returns
    -------
    epochs_data : ndarray, shape (n_epochs, n_channels, epoch_len_samples)
        Data segmented into consecutive, non-overlapping 5 s epochs with NaNs
        at artifact positions. Any incomplete tail shorter than one full epoch
        is discarded.
    """
    if data_cont.shape[1] != artifact_mask.shape[0]:
        raise ValueError(
            "artifact_mask length must match data_cont time dimension "
            f"(got data_cont.shape[1]={data_cont.shape[1]}, "
            f"artifact_mask.shape[0]={artifact_mask.shape[0]})."
        )

    n_channels, n_times = data_cont.shape
    if epoch_len_samples <= 0:
        raise ValueError("epoch_len_samples must be positive.")

    # Apply NaNs
    data_nan = data_cont.copy()
    data_nan[:, artifact_mask] = np.nan

    # How many full epochs fit?
    n_epochs = n_times // epoch_len_samples
    if n_epochs == 0:
        raise RuntimeError(
            "Not enough samples to form a single full epoch of length "
            f"{epoch_len_samples} samples. Total samples: {n_times}."
        )

    trimmed_len = n_epochs * epoch_len_samples
    data_trim = data_nan[:, :trimmed_len]

    # Reshape into (n_epochs, n_channels, epoch_len_samples)
    epochs_data = data_trim.reshape(
        n_channels, n_epochs, epoch_len_samples
    ).transpose(1, 0, 2)

    return epochs_data


def nanmark_and_export(
    input_pkl_path: str,
    output_fif_path: Optional[str] = None,
    sfreq: float = 2000.0,
    epoch_len_s: float = 5.0,
    amp_z_hard: float = 8.0,
    deriv_z_hard: float = 8.0,
    drift_z_thresh: float = 8.0,
    intercept_z_thresh: float = 8.0,
    min_dur_ms_amp: float = 10.0,
    min_dur_ms_deriv: float = 8.0,
    l_freq: Optional[float] = 0.5,
    h_freq: Optional[float] = 100.0,
    fir_design: str = "firwin",
    filter_length: str = "auto",
    verbose: bool = True,
) -> mne.Epochs:
    """Load, detect artifacts (i–iv), filter, NaN-mark, epoch, and export to FIF.

    Steps
    -----
    1. Load pickle and reconstruct continuous data (n_channels, n_times).
    2. Detect artifact mask (union of criteria i–iv) and component masks.
    3. (Optionally) band-pass filter the continuous data.
    4. Replace artifact samples with NaNs.
    5. Cut the continuous NaN-marked data into consecutive 5 s epochs.
    6. Wrap as MNE EpochsArray and save as FIF.

    Parameters
    ----------
    input_pkl_path : str
        Path to the input pickle file.
    output_fif_path : str | None
        Output FIF path. If None, created in the same directory as input with
        suffix "_nan5s_filtered.fif" and auto-increment if needed.
    sfreq : float
        Sampling frequency.
    epoch_len_s : float
        Length of epochs for resegmentation, in seconds (default 5 s).
    amp_z_hard : float
        Hard z-threshold for amplitude excursions (criterion i).
    deriv_z_hard : float
        Hard z-threshold for derivative excursions (criterion ii).
    drift_z_thresh : float
        Robust z-threshold for drift power (criterion iii).
    intercept_z_thresh : float
        Robust z-threshold for 1/f intercept (criterion iv).
    min_dur_ms_amp : float
        Minimum duration in ms for amplitude excursions.
    min_dur_ms_deriv : float
        Minimum duration in ms for derivative excursions.
    l_freq : float | None
        Lower cutoff for band-pass filter (Hz). If None, no high-pass.
    h_freq : float | None
        Upper cutoff for band-pass filter (Hz). If None, no low-pass.
    fir_design : str
        FIR design method for MNE filter (e.g., "firwin").
    filter_length : str | int
        Filter length, e.g. "auto" or an integer number of taps.
    verbose : bool
        If True, prints progress and summary.

    Returns
    -------
    epochs_nan : mne.Epochs
        The NaN-marked and (optionally) band-pass filtered epochs.
    """
    if verbose:
        print(f"[nanmark_and_export] Loading pickle: {input_pkl_path}")

    data_cont, epoch_len_samples, ch_names = load_pickle_as_continuous(
        input_pkl_path,
        sfreq=sfreq,
        epoch_len_s=epoch_len_s,
    )
    # Ensure floating type for filtering
    data_cont = data_cont.astype(np.float64, copy=False)

    if verbose:
        n_channels, n_times = data_cont.shape
        print(f"  Continuous data shape: (n_channels={n_channels}, n_times={n_times})")
        print(f"  Total duration: {n_times / sfreq:.2f} s")
        print(f"  Epoch length target: {epoch_len_samples} samples "
              f"({epoch_len_samples / sfreq:.2f} s)")

    # Build artifact mask (i–iv)
    artifact_mask, component_masks = build_artifact_mask_i_to_iv(
        data_cont,
        sfreq=sfreq,
        epoch_len_samples=epoch_len_samples,
        amp_z_hard=amp_z_hard,
        deriv_z_hard=deriv_z_hard,
        drift_z_thresh=drift_z_thresh,
        intercept_z_thresh=intercept_z_thresh,
        min_dur_ms_amp=min_dur_ms_amp,
        min_dur_ms_deriv=min_dur_ms_deriv,
    )

    n_times = data_cont.shape[1]
    n_artifact = int(artifact_mask.sum())
    n_good = int((~artifact_mask).sum())

    if verbose:
        def _sec(count): return count / sfreq
        print("Artifact counts per criterion (samples, not mutually exclusive):")
        for key in ["amp", "deriv", "drift", "intercept"]:
            c = int(component_masks[key].sum())
            print(f"  {key:9s}: {c:7d} samples (~{_sec(c):7.2f} s)")
        print(f"Combined (union): {n_artifact} samples (~{_sec(n_artifact):.2f} s)")
        print(f"Remaining clean (non-artifact):  {n_good} samples (~{_sec(n_good):.2f} s)")
        print(f"Fraction marked as artifact: {n_artifact / n_times:.2%}")

    # Filter continuous data first (no NaNs yet)
    data_filt = data_cont.copy()
    if l_freq is not None or h_freq is not None:
        if verbose:
            print(f"Applying band-pass filter before NaN marking: "
                  f"{l_freq}-{h_freq} Hz (fir_design={fir_design}, "
                  f"filter_length={filter_length})")
        data_filt = mne.filter.filter_data(
            data_filt,
            sfreq=sfreq,
            l_freq=l_freq,
            h_freq=h_freq,
            fir_design=fir_design,
            filter_length=filter_length,
            verbose=verbose,
        )

    # Apply NaNs at artifact positions and epoch
    if verbose:
        print("Replacing artifact samples with NaNs (preserving temporal structure) "
              "and cutting into consecutive 5 s epochs.")
    epochs_data = _nanmark_and_epoch(
        data_filt,
        artifact_mask,
        epoch_len_samples=epoch_len_samples,
    )

    n_epochs_new, n_channels, n_samples_epoch = epochs_data.shape
    if verbose:
        print(f"After NaN marking + 5 s segmentation: {n_epochs_new} epochs of "
              f"{n_samples_epoch} samples each (~{n_samples_epoch / sfreq:.2f} s).")

    # Build MNE Epochs (NaNs allowed)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    epochs_nan = mne.EpochsArray(epochs_data, info, verbose=False)

    # Build output path if needed
    if output_fif_path is None:
        base_dir = os.path.dirname(os.path.abspath(input_pkl_path))
        base_name = os.path.basename(input_pkl_path)
        output_fif_path = _unique_output_path(base_dir, base_name)

    if verbose:
        print(f"Saving NaN-marked FIF to: {output_fif_path}")

    epochs_nan.save(output_fif_path, overwrite=True)
    if verbose:
        print("Done.")

    return epochs_nan


if __name__ == "__main__":
    # Example script entry point: edit INPUT_PKL and run
    INPUT_PKL = r"/path/to/your_file.pkl"  # <-- EDIT THIS
    if not os.path.isfile(INPUT_PKL):
        raise SystemExit(
            "Please edit artifact_excision_nanmark_5s_v1.py and set INPUT_PKL "
            "to an existing pickle file before running as a script."
        )

    nanmark_and_export(INPUT_PKL)
