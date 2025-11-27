
"""artifact_excision_resegment_5s_v1.py

Utility to:
1. Load a pickle file containing rsERG-like data.
2. Treat it as a continuous multichannel time series.
3. Detect artifacts of types (i)–(iv):
   (i)  large amplitude |z| excursions
   (ii) sharp derivative |z| excursions
   (iii) high low-frequency drift power (0.5–1 Hz)
   (iv) broadband 1/f power surges (1/f intercept at 20 Hz)
4. Excise those artifact segments from the continuous data.
5. Reconnect remaining samples and resegment into 5 s epochs.
6. Band-pass filter and export as a cleaned MNE FIF file.

Assumptions
----------
- Sampling rate is known (default 2000 Hz).
- If the pickle contains a 3D array of shape (n_epochs, n_channels, n_times),
  then epochs are assumed to be back-to-back pieces of a continuous recording,
  and we reconstruct a continuous array by concatenating them in time.
- If the pickle contains a dict of such arrays, we concatenate across blocks.
- If the pickle contains a 2D array (n_channels, n_times), we treat it as
  already-continuous.

You can either import and call `clean_and_export(...)` from a Jupyter notebook,
or run this file as a script and edit the INPUT_PKL path at the bottom.
"""

import os
import pickle
from typing import Tuple, Optional, Dict

import numpy as np
import mne
from mne.time_frequency import psd_array_welch


# =============================================================================
# Basic helpers
# =============================================================================

def load_pickle_as_continuous(
    file_path: str,
    sfreq: float = 2000.0,
    epoch_len_s: float = 5.0,
) -> Tuple[np.ndarray, int, list]:
    """Load a pickle file and return a continuous array (n_channels, n_times).

    The function supports:
      - dict of arrays: concatenates along epoch axis
      - 3D array: (n_epochs, n_channels, n_times)
      - 2D array: (n_channels, n_times)

    Returns
    -------
    data_cont : ndarray, shape (n_channels, n_times)
        Continuous data.
    epoch_len_samples : int
        Epoch length in samples used for later 5 s re-epoching.
    ch_names : list of str
        Channel names (Ch1, Ch2, ...).
    """
    with open(file_path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        arrays = []
        for key in sorted(obj.keys()):
            arr = np.asarray(obj[key])
            if arr.ndim != 3:
                raise ValueError(
                    f"Dict entry '{key}' has shape {arr.shape}, expected 3D "
                    "(n_epochs, n_channels, n_times)."
                )
            arrays.append(arr)
        if not arrays:
            raise ValueError("Dictionary in pickle is empty.")
        data = np.concatenate(arrays, axis=0)  # (n_epochs_total, n_channels, n_times)

    else:
        data = np.asarray(obj)

    if data.ndim == 3:
        # data: (n_epochs, n_channels, n_times)
        n_epochs, n_channels, n_times = data.shape
        # Concatenate epochs into continuous time
        data_cont = data.transpose(1, 0, 2).reshape(n_channels, n_epochs * n_times)
        epoch_len_samples = n_times

    elif data.ndim == 2:
        # data: (n_channels, n_times)
        n_channels, n_times = data.shape
        data_cont = data
        epoch_len_samples = int(round(epoch_len_s * sfreq))
        if epoch_len_samples <= 0:
            raise ValueError("epoch_len_s must be positive.")

    else:
        raise ValueError(
            f"Unsupported data shape {data.shape}. "
            "Expected 2D (n_channels, n_times) or 3D (n_epochs, n_channels, n_times) "
            "or dict of such arrays."
        )

    ch_names = [f"Ch{i+1}" for i in range(data_cont.shape[0])]
    return data_cont, epoch_len_samples, ch_names


def zscore_per_channel_2d(data_2d: np.ndarray) -> np.ndarray:
    """Z-score each channel across time.

    Parameters
    ----------
    data_2d : ndarray, shape (n_channels, n_times)

    Returns
    -------
    z : ndarray, shape (n_channels, n_times)
    """
    data_2d = np.asarray(data_2d, dtype=float)
    mean = data_2d.mean(axis=1, keepdims=True)
    std = data_2d.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    z = (data_2d - mean) / std
    return z


def robust_zscore(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute robust z-score using median and MAD.

    z = (x - median) / (1.4826 * MAD)

    Parameters
    ----------
    x : ndarray
    axis : int
        Axis along which to compute z (typically windows/epochs).

    Returns
    -------
    z : ndarray
        Robust z-scores, same shape as x.
    """
    x = np.asarray(x, dtype=float)
    median = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - median), axis=axis, keepdims=True)
    mad_scaled = mad * 1.4826
    mad_scaled[mad_scaled == 0] = 1.0
    z = (x - median) / mad_scaled
    return z


def _compute_band_power_db(
    freqs: np.ndarray,
    psd: np.ndarray,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Compute mean PSD power in a band and convert to dB.

    Parameters
    ----------
    freqs : ndarray, shape (n_freqs,)
    psd : ndarray, shape (n_epochs, n_channels, n_freqs)
    fmin : float
    fmax : float

    Returns
    -------
    power_db : ndarray, shape (n_epochs, n_channels)
    """
    freqs = np.asarray(freqs)
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band_mask):
        raise RuntimeError(f"No frequency bins found in band {fmin}-{fmax} Hz.")

    band_psd = psd[..., band_mask]  # (n_epochs, n_channels, n_band_freqs)
    power = band_psd.mean(axis=-1)
    power_db = 10 * np.log10(power + 1e-20)
    return power_db


def _compute_aperiodic_20hz(
    freqs: np.ndarray,
    psd: np.ndarray,
    fmin: float = 2.0,
    fmax: float = 40.0,
    f_eval: float = 20.0,
) -> np.ndarray:
    """Fit simple 1/f line in log-log space and evaluate at f_eval.

    Parameters
    ----------
    freqs : ndarray, shape (n_freqs,)
    psd : ndarray, shape (n_epochs, n_channels, n_freqs)
    fmin : float
    fmax : float
    f_eval : float

    Returns
    -------
    intercept_vals : ndarray, shape (n_epochs, n_channels)
        Log10 power at f_eval from the fitted 1/f line.
    """
    freqs = np.asarray(freqs)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        raise RuntimeError(f"No frequency bins found in fit band {fmin}-{fmax} Hz.")

    log_f = np.log10(freqs[mask])
    log_psd = np.log10(psd[..., mask] + 1e-20)  # (n_epochs, n_channels, n_fit_freqs)

    n_epochs, n_channels, n_fit = log_psd.shape
    f_eval_log = np.log10(f_eval)

    intercept_vals = np.zeros((n_epochs, n_channels), dtype=float)

    for e in range(n_epochs):
        for ch in range(n_channels):
            y = log_psd[e, ch, :]
            if not np.all(np.isfinite(y)):
                intercept_vals[e, ch] = np.nan
                continue
            try:
                slope, intercept = np.polyfit(log_f, y, 1)
                intercept_vals[e, ch] = slope * f_eval_log + intercept
            except Exception:
                intercept_vals[e, ch] = np.nan

    return intercept_vals


# =============================================================================
# Artifact detection: i) & ii) on continuous data
# =============================================================================

def detect_amp_artifacts(
    z_amp: np.ndarray,
    z_thresh: float,
    min_samples: int,
) -> np.ndarray:
    """Detect large-amplitude artifacts (criterion i) on continuous data.

    Parameters
    ----------
    z_amp : ndarray, shape (n_channels, n_times)
        Amplitude z-scores per channel.
    z_thresh : float
        Hard z-threshold, e.g. 8.
    min_samples : int
        Minimum length (in samples) of a contiguous run above threshold.

    Returns
    -------
    artifact_mask : ndarray, shape (n_times,)
        Boolean mask, True where the time point is in an artifact run.
    """
    n_channels, n_times = z_amp.shape
    artifact_mask = np.zeros(n_times, dtype=bool)

    if z_thresh <= 0:
        return artifact_mask

    for ch in range(n_channels):
        row = np.abs(z_amp[ch])
        if row.max() < z_thresh:
            continue

        run_len = 0
        for i, is_high in enumerate(row >= z_thresh):
            if is_high:
                run_len += 1
            else:
                if run_len >= min_samples:
                    start = i - run_len
                    end = i
                    artifact_mask[start:end] = True
                run_len = 0

        # tail run
        if run_len >= min_samples:
            start = n_times - run_len
            end = n_times
            artifact_mask[start:end] = True

    return artifact_mask


def detect_deriv_artifacts(
    z_deriv: np.ndarray,
    z_thresh: float,
    min_samples: int,
    n_times: int,
) -> np.ndarray:
    """Detect sharp derivative artifacts (criterion ii) on continuous data.

    Parameters
    ----------
    z_deriv : ndarray, shape (n_channels, n_times-1)
        Z-scored first-difference per channel.
    z_thresh : float
        Hard z-threshold, e.g. 8.
    min_samples : int
        Minimum length (in samples of the derivative vector) of a contiguous run.
    n_times : int
        Number of original time points in the amplitude signal.

    Returns
    -------
    artifact_mask : ndarray, shape (n_times,)
        Boolean mask, True where the time point is in an artifact run.
    """
    n_channels, n_dt = z_deriv.shape
    if n_dt != n_times - 1:
        raise ValueError(
            f"z_deriv has length {n_dt} along time, expected n_times-1={n_times-1}."
        )

    mask_dt = np.zeros(n_dt, dtype=bool)

    if z_thresh <= 0:
        return np.zeros(n_times, dtype=bool)

    for ch in range(n_channels):
        row = np.abs(z_deriv[ch])
        if row.max() < z_thresh:
            continue

        run_len = 0
        for i, is_high in enumerate(row >= z_thresh):
            if is_high:
                run_len += 1
            else:
                if run_len >= min_samples:
                    start = i - run_len
                    end = i
                    mask_dt[start:end] = True
                run_len = 0

        # tail run
        if run_len >= min_samples:
            start = n_dt - run_len
            end = n_dt
            mask_dt[start:end] = True

    # Map derivative mask to time-point mask
    artifact_mask = np.zeros(n_times, dtype=bool)
    idx = np.where(mask_dt)[0]
    if idx.size > 0:
        idx_plus = np.clip(idx + 1, 0, n_times - 1)
        artifact_mask[idx] = True
        artifact_mask[idx_plus] = True

    return artifact_mask


# =============================================================================
# Artifact detection: iii) & iv) using 5 s windows
# =============================================================================

def detect_spectral_artifacts(
    data_cont: np.ndarray,
    sfreq: float,
    epoch_len_samples: int,
    drift_z_thresh: float,
    intercept_z_thresh: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect spectral artifacts (criteria iii & iv) using 5 s windows.

    The continuous data is segmented into non-overlapping windows of length
    `epoch_len_samples`. For each window and channel, we compute:
      - drift power (0.5–1 Hz) -> robust z across windows
      - 1/f intercept at 20 Hz (fit 2–40 Hz) -> robust z across windows

    Parameters
    ----------
    data_cont : ndarray, shape (n_channels, n_times)
    sfreq : float
    epoch_len_samples : int
    drift_z_thresh : float
    intercept_z_thresh : float

    Returns
    -------
    mask_drift : ndarray, shape (n_times,)
        True in windows where drift power z exceeds threshold (criterion iii).
    mask_intercept : ndarray, shape (n_times,)
        True in windows where 1/f intercept z exceeds threshold (criterion iv).
    """
    n_channels, n_times = data_cont.shape

    if n_times < epoch_len_samples:
        # Not enough data for one full 5 s window; no spectral artifact tagging
        return np.zeros(n_times, dtype=bool), np.zeros(n_times, dtype=bool)

    n_win = n_times // epoch_len_samples
    if n_win == 0:
        return np.zeros(n_times, dtype=bool), np.zeros(n_times, dtype=bool)

    trimmed_len = n_win * epoch_len_samples
    data_trim = data_cont[:, :trimmed_len]
    # shape -> (n_windows, n_channels, epoch_len_samples)
    data_win = data_trim.reshape(n_channels, n_win, epoch_len_samples).transpose(1, 0, 2)

    # PSD for each window and channel
    n_fft = min(epoch_len_samples, 2048)
    n_overlap = n_fft // 2

    psd, freqs = psd_array_welch(
        data_win,
        sfreq=sfreq,
        fmin=0.1,
        fmax=80.0,
        n_fft=n_fft,
        n_overlap=n_overlap,
        average="mean",
        verbose=False,
    )

    # (iii) drift power 0.5–1 Hz
    drift_power_db = _compute_band_power_db(freqs, psd, fmin=0.5, fmax=1.0)
    drift_z = robust_zscore(drift_power_db, axis=0)  # across windows

    # (iv) 1/f intercept at 20 Hz (2–40 Hz fit)
    intercept_vals = _compute_aperiodic_20hz(freqs, psd, fmin=2.0, fmax=40.0, f_eval=20.0)
    intercept_z = robust_zscore(intercept_vals, axis=0)

    bad_win_drift = (drift_z >= drift_z_thresh).any(axis=1)
    bad_win_intercept = (intercept_z >= intercept_z_thresh).any(axis=1)

    mask_drift = np.zeros(n_times, dtype=bool)
    mask_intercept = np.zeros(n_times, dtype=bool)

    for w in range(n_win):
        start = w * epoch_len_samples
        end = start + epoch_len_samples
        if bad_win_drift[w]:
            mask_drift[start:end] = True
        if bad_win_intercept[w]:
            mask_intercept[start:end] = True

    # Any leftover tail (n_times - trimmed_len) is not tagged by spectral criteria
    return mask_drift, mask_intercept


# =============================================================================
# High-level artifact mask builder (i–iv) and excision/re-epoching
# =============================================================================

def build_artifact_mask_i_to_iv(
    data_cont: np.ndarray,
    sfreq: float,
    epoch_len_samples: int,
    amp_z_hard: float = 8.0,
    deriv_z_hard: float = 8.0,
    drift_z_thresh: float = 8.0,
    intercept_z_thresh: float = 8.0,
    min_dur_ms_amp: float = 10.0,
    min_dur_ms_deriv: float = 8.0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Build a combined artifact mask for criteria (i)–(iv).

    Parameters
    ----------
    data_cont : ndarray, shape (n_channels, n_times)
    sfreq : float
    epoch_len_samples : int
        Used for the 5 s windows in criteria (iii) and (iv).
    amp_z_hard : float
        Hard threshold for amplitude |z| excursions.
    deriv_z_hard : float
        Hard threshold for derivative |z| excursions.
    drift_z_thresh : float
        Robust z threshold for drift power.
    intercept_z_thresh : float
        Robust z threshold for 1/f intercept.
    min_dur_ms_amp : float
        Minimum duration in ms for amplitude excursions.
    min_dur_ms_deriv : float
        Minimum duration in ms for derivative excursions.

    Returns
    -------
    artifact_mask : ndarray, shape (n_times,)
        True where any criterion (i)–(iv) flags artifact.
    component_masks : dict
        Dictionary with individual masks:
        'amp', 'deriv', 'drift', 'intercept'.
    """
    n_channels, n_times = data_cont.shape

    # i) amplitude z
    z_amp = zscore_per_channel_2d(data_cont)
    min_samples_amp = max(int(round(min_dur_ms_amp * sfreq / 1000.0)), 1)
    mask_amp = detect_amp_artifacts(z_amp, amp_z_hard, min_samples_amp)

    # ii) derivative z
    deriv = np.diff(data_cont, axis=1)
    z_deriv = zscore_per_channel_2d(deriv)
    min_samples_deriv = max(int(round(min_dur_ms_deriv * sfreq / 1000.0)), 1)
    mask_deriv = detect_deriv_artifacts(z_deriv, deriv_z_hard, min_samples_deriv, n_times)

    # iii & iv) spectral
    mask_drift, mask_intercept = detect_spectral_artifacts(
        data_cont,
        sfreq=sfreq,
        epoch_len_samples=epoch_len_samples,
        drift_z_thresh=drift_z_thresh,
        intercept_z_thresh=intercept_z_thresh,
    )

    artifact_mask = mask_amp | mask_deriv | mask_drift | mask_intercept
    component_masks = {
        "amp": mask_amp,
        "deriv": mask_deriv,
        "drift": mask_drift,
        "intercept": mask_intercept,
    }
    return artifact_mask, component_masks


def excise_artifacts_and_resegment(
    data_cont: np.ndarray,
    artifact_mask: np.ndarray,
    epoch_len_samples: int,
) -> np.ndarray:
    """Excise artifact segments, reconnect, and resegment into 5 s epochs.

    Parameters
    ----------
    data_cont : ndarray, shape (n_channels, n_times)
    artifact_mask : ndarray, shape (n_times,)
    epoch_len_samples : int
        Target epoch length in samples, e.g. 5 s * 2000 Hz = 10000.

    Returns
    -------
    epochs_data : ndarray, shape (n_epochs_new, n_channels, epoch_len_samples)

    Notes
    -----
    - All time points where `artifact_mask` is True are removed.
    - Remaining samples are concatenated and then cut into consecutive epochs
      of length `epoch_len_samples`. Any tail shorter than one full epoch is
      discarded.
    """
    if data_cont.shape[1] != artifact_mask.shape[0]:
        raise ValueError("artifact_mask length must match data_cont time dimension.")

    n_channels, n_times = data_cont.shape
    good_mask = ~artifact_mask
    data_good = data_cont[:, good_mask]
    n_good = data_good.shape[1]

    if n_good < epoch_len_samples:
        raise RuntimeError(
            "Not enough clean data to form a single epoch of length "
            f"{epoch_len_samples} samples. Clean samples: {n_good}."
        )

    n_epochs_new = n_good // epoch_len_samples
    trimmed_len = n_epochs_new * epoch_len_samples
    data_good_trim = data_good[:, :trimmed_len]

    # reshape into (n_epochs, n_channels, epoch_len_samples)
    epochs_data = data_good_trim.reshape(
        n_channels, n_epochs_new, epoch_len_samples
    ).transpose(1, 0, 2)

    return epochs_data


# =============================================================================
# High-level function: clean & export
# =============================================================================

def _unique_output_path(
    base_dir: str,
    base_name: str,
    suffix: str = "_clean5s_filtered.fif",
) -> str:
    """Create a unique FIF output path in base_dir using base_name + suffix."""
    root = os.path.splitext(base_name)[0]
    candidate = os.path.join(base_dir, root + suffix)
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(base_dir, f"{root}_clean5s_filtered_{counter}.fif")
        counter += 1
    return candidate


def clean_and_export(
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
    l_freq: float = 0.5,
    h_freq: float = 100.0,
    fir_design: str = "firwin",
    filter_length: str = "auto",
    verbose: bool = True,
) -> mne.Epochs:
    """Full pipeline: load, detect artifacts (i–iv), excise, re-epoch, filter, save.

    Parameters
    ----------
    input_pkl_path : str
        Path to the input pickle file.
    output_fif_path : str | None
        Output FIF path. If None, created in same directory as input with
        suffix "_clean5s_filtered.fif" and auto-increment if needed.
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
    l_freq : float
        Lower cutoff for band-pass filter (Hz).
    h_freq : float
        Upper cutoff for band-pass filter (Hz).
    fir_design : str
        FIR design method for MNE filter (e.g., "firwin").
    filter_length : str | int
        Filter length, e.g. "auto" or an integer number of taps.
    verbose : bool
        If True, prints progress and summary.

    Returns
    -------
    epochs_filtered : mne.Epochs
        The cleaned and band-pass filtered epochs.
    """
    if verbose:
        print(f"Loading pickle: {input_pkl_path}")

    data_cont, epoch_len_samples, ch_names = load_pickle_as_continuous(
        input_pkl_path,
        sfreq=sfreq,
        epoch_len_s=epoch_len_s,
    )

    if verbose:
        n_channels, n_times = data_cont.shape
        print(f"Continuous data shape: (n_channels={n_channels}, n_times={n_times})")
        print(f"  Total duration: {n_times / sfreq:.2f} s")
        print(f"  Epoch length target: {epoch_len_samples} samples "
              f"({epoch_len_samples / sfreq:.2f} s)")

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
        print(f"Remaining clean:  {n_good} samples (~{_sec(n_good):.2f} s)")
        print(f"Fraction removed: {n_artifact / n_times:.2%}")

    if n_good < epoch_len_samples:
        raise RuntimeError(
            f"Not enough clean samples ({n_good}) to form a single epoch of "
            f"length {epoch_len_samples}. Adjust thresholds or check data."
        )

    epochs_data = excise_artifacts_and_resegment(
        data_cont,
        artifact_mask,
        epoch_len_samples=epoch_len_samples,
    )

    n_epochs_new, n_channels, n_samples_epoch = epochs_data.shape
    if verbose:
        print(f"After excision + resegmentation: {n_epochs_new} epochs of "
              f"{n_samples_epoch} samples each (~{n_samples_epoch / sfreq:.2f} s).")

    # Build MNE Epochs and filter
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    epochs = mne.EpochsArray(epochs_data, info, verbose=False)

    if verbose:
        print(f"Applying band-pass filter: {l_freq}-{h_freq} Hz "
              f"(fir_design={fir_design}, filter_length={filter_length})")

    epochs_filtered = epochs.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        fir_design=fir_design,
        filter_length=filter_length,
        verbose=verbose,
    )

    # Build output path if needed
    if output_fif_path is None:
        base_dir = os.path.dirname(os.path.abspath(input_pkl_path))
        base_name = os.path.basename(input_pkl_path)
        output_fif_path = _unique_output_path(base_dir, base_name)

    if verbose:
        print(f"Saving cleaned FIF to: {output_fif_path}")

    epochs_filtered.save(output_fif_path, overwrite=True)
    if verbose:
        print("Done.")

    return epochs_filtered


# =============================================================================
# Script entry point (edit paths here if running as a script)
# =============================================================================

if __name__ == "__main__":
    # Example usage: edit this path to point to your .pkl file and run:
    #
    #   python artifact_excision_resegment_5s_v1.py
    #
    INPUT_PKL = r"/path/to/your_file.pkl"  # <-- EDIT THIS
    if not os.path.isfile(INPUT_PKL):
        raise SystemExit(
            "Please edit artifact_excision_resegment_5s_v1.py and set INPUT_PKL "
            "to an existing pickle file before running as a script."
        )

    clean_and_export(INPUT_PKL)
