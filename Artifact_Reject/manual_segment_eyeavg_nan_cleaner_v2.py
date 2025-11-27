"""manual_segment_eyeavg_nan_cleaner_v2.py

Manual segment-based artifact marking on Eye1/Eye2 averages, with NaN replacement
and export to a new pickle file.

This version *does not* rely on mouse clicks inside the matplotlib figure.
Instead, you define segments using a slider for [start, end] in seconds, which
is robust in JupyterLab even with the default (non-interactive) matplotlib
backend.

Workflow
--------
1. Load a multichannel rsERG pickle file.
2. Optionally average Eye1 (ch 0–7) and Eye2 (ch 8–15) into two synthetic
   channels, with the ability to drop some channels from each average.
3. Scroll through the averaged time series with a movable window (Window / Center).
4. Choose Eye1_avg or Eye2_avg in the channel dropdown.
5. Use the "Segment (s)" range slider to choose [start, end] (absolute times).
6. Click "Add segment" to append that [start, end] interval (for the current
   channel) to the segments table.
7. Remove segments by selecting them in the table and clicking "Remove segment".
8. When finished, specify an output directory and filename, then click
   "Clean & export as .pkl":
   - For segments on Eye1_avg, NaNs are applied across all Eye1 channels.
   - For segments on Eye2_avg, NaNs are applied across all Eye2 channels.
   - The internal structure of the pickle (dict / 3D array / 2D array) is
     preserved, only with NaNs inserted.

Typical usage (JupyterLab)
--------------------------
    from manual_segment_eyeavg_nan_cleaner_v2 import launch_manual_segment_cleaner

    input_pkl = r"D:\Open Ephys\Isoflurane\C57 220\20251114\Analysis\MNE Array\1_C57_220_Isoflurane0p5per_up_epoched.pkl"

    launch_manual_segment_cleaner(
        input_pkl_path=input_pkl,
        sfreq=2000.0,
        epoch_len_s=5.0,
        average_eye_groups=True,
        eye1_indices=tuple(range(0, 8)),
        eye2_indices=tuple(range(8, 16)),
        drop_indices_eye1=[0, 1, 2],   # drop these from Eye1 average
        # drop_indices_eye2=[...]      # optional, analogous for Eye2
    )

You do NOT need %matplotlib widget for this version; standard inline plots work.
"""

import os
import pickle
from typing import Dict, Tuple, Sequence, Optional, List

import numpy as np
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display, clear_output

from artifact_excision_resegment_5s_v1 import (
    load_pickle_as_continuous,
)


# -------------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------------


def _compute_eye_averages(
    data_cont: np.ndarray,
    ch_names: Sequence[str],
    eye1_indices: Sequence[int],
    eye2_indices: Sequence[int],
    drop_indices_eye1: Optional[Sequence[int]] = None,
    drop_indices_eye2: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Compute Eye1/Eye2 average channels with optional dropping.

    Parameters
    ----------
    data_cont : ndarray, shape (n_channels, n_times)
        Original continuous data.
    ch_names : sequence of str
        Original channel names.
    eye1_indices : sequence of int
        Indices for Eye1 group (e.g., 0–7).
    eye2_indices : sequence of int
        Indices for Eye2 group (e.g., 8–15).
    drop_indices_eye1 : sequence of int | None
        Channel indices (in original space) to drop from Eye1 before averaging.
    drop_indices_eye2 : sequence of int | None
        Channel indices (in original space) to drop from Eye2 before averaging.

    Returns
    -------
    data_avg : ndarray, shape (2, n_times)
        data_avg[0] = average of Eye1 channels, data_avg[1] = average of Eye2 channels.
    ch_names_avg : list of str
        Names of the synthetic channels: ["Eye1_avg", "Eye2_avg"].
    """
    n_channels, n_times = data_cont.shape

    eye1 = np.array(eye1_indices, dtype=int)
    eye2 = np.array(eye2_indices, dtype=int)

    # Restrict to valid indices
    eye1 = eye1[(eye1 >= 0) & (eye1 < n_channels)]
    eye2 = eye2[(eye2 >= 0) & (eye2 < n_channels)]

    # Drop requested indices
    if drop_indices_eye1 is not None:
        drop1 = np.array(list(drop_indices_eye1), dtype=int)
        eye1 = np.array([idx for idx in eye1 if idx not in drop1])
    if drop_indices_eye2 is not None:
        drop2 = np.array(list(drop_indices_eye2), dtype=int)
        eye2 = np.array([idx for idx in eye2 if idx not in drop2])

    if eye1.size == 0:
        raise ValueError("Eye1 group has no channels left after dropping.")
    if eye2.size == 0:
        raise ValueError("Eye2 group has no channels left after dropping.")

    eye1_avg = data_cont[eye1].mean(axis=0)
    eye2_avg = data_cont[eye2].mean(axis=0)

    data_avg = np.vstack([eye1_avg, eye2_avg])
    ch_names_avg = ["Eye1_avg", "Eye2_avg"]

    print("Eye averaging configuration:")
    print(f"  Eye1 indices used: {eye1.tolist()}")
    print(f"  Eye2 indices used: {eye2.tolist()}")

    return data_avg, ch_names_avg


def _apply_nan_segments_to_pickle(
    input_pkl_path: str,
    output_pkl_path: str,
    segments: List[Dict[str, int]],
    n_times_total: int,
    epoch_len_samples: int,
    eye1_indices: Sequence[int],
    eye2_indices: Sequence[int],
) -> None:
    """Apply NaN replacement based on manually marked segments and save new pickle.

    Parameters
    ----------
    input_pkl_path : str
        Path to the original pickle file.
    output_pkl_path : str
        Path where the cleaned pickle will be written.
    segments : list of dict
        Each dict has keys: 'ch' (0 or 1), 'start', 'end' (sample indices).
        ch=0 means Eye1_avg, ch=1 means Eye2_avg.
    n_times_total : int
        Total number of samples in the continuous time base.
    epoch_len_samples : int
        Epoch length in samples in the original data.
    eye1_indices, eye2_indices : sequence of int
        Original channel indices belonging to Eye1 and Eye2 groups.
    """
    # Build masks for Eye1 and Eye2
    mask_eye1 = np.zeros(n_times_total, dtype=bool)
    mask_eye2 = np.zeros(n_times_total, dtype=bool)

    for seg in segments:
        ch = int(seg["ch"])
        s = int(seg["start"])
        e = int(seg["end"])
        s = max(0, min(n_times_total, s))
        e = max(0, min(n_times_total, e))
        if e <= s:
            continue
        if ch == 0:
            mask_eye1[s:e] = True
        elif ch == 1:
            mask_eye2[s:e] = True

    # Load original pickle
    with open(input_pkl_path, "rb") as f:
        obj = pickle.load(f)

    def _apply_to_3d_array(arr: np.ndarray) -> np.ndarray:
        """Apply NaNs to a 3D array (n_epochs, n_channels, n_times)."""
        arr = np.asarray(arr)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {arr.shape}.")
        n_epochs, n_channels, n_times = arr.shape
        if n_times != epoch_len_samples:
            raise ValueError(
                f"Array n_times={n_times} does not match epoch_len_samples={epoch_len_samples}."
            )
        if n_epochs * n_times != n_times_total:
            raise ValueError(
                f"Total samples n_epochs*n_times={n_epochs*n_times} does not "
                f"match n_times_total={n_times_total}."
            )

        # Ensure float for NaNs
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float64, copy=True)

        # Reshape masks into (n_epochs, n_times)
        mask1_2d = mask_eye1.reshape(n_epochs, n_times)
        mask2_2d = mask_eye2.reshape(n_epochs, n_times)

        # Apply to Eye1 channels
        for ch in eye1_indices:
            if 0 <= ch < n_channels:
                arr[:, ch, :][mask1_2d] = np.nan

        # Apply to Eye2 channels
        for ch in eye2_indices:
            if 0 <= ch < n_channels:
                arr[:, ch, :][mask2_2d] = np.nan

        return arr

    def _apply_to_2d_array(arr: np.ndarray) -> np.ndarray:
        """Apply NaNs to a 2D array (n_channels, n_times_total)."""
        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {arr.shape}.")
        n_channels, n_times = arr.shape
        if n_times != n_times_total:
            raise ValueError(
                f"2D array n_times={n_times} does not match n_times_total={n_times_total}."
            )

        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float64, copy=True)

        for ch in eye1_indices:
            if 0 <= ch < n_channels:
                arr[ch, mask_eye1] = np.nan
        for ch in eye2_indices:
            if 0 <= ch < n_channels:
                arr[ch, mask_eye2] = np.nan

        return arr

    # Dispatch depending on container type
    if isinstance(obj, dict):
        # Assume dict of 3D arrays
        keys_sorted = sorted(obj.keys())
        arrays = [np.asarray(obj[k]) for k in keys_sorted]
        if not arrays:
            raise ValueError("Dictionary in pickle is empty.")
        n_epochs_total = sum(a.shape[0] for a in arrays)
        n_times = arrays[0].shape[2]
        if n_times != epoch_len_samples:
            raise ValueError(
                f"Dict arrays n_times={n_times} do not match epoch_len_samples={epoch_len_samples}."
            )
        if n_epochs_total * n_times != n_times_total:
            raise ValueError(
                f"Total samples in dict ({n_epochs_total} epochs x {n_times} times) "
                f"does not match n_times_total={n_times_total}."
            )

        mask1_2d = mask_eye1.reshape(n_epochs_total, n_times)
        mask2_2d = mask_eye2.reshape(n_epochs_total, n_times)

        ep_offset = 0
        for key, arr in zip(keys_sorted, arrays):
            n_ep_k, n_ch, n_t = arr.shape
            local_mask1 = mask1_2d[ep_offset : ep_offset + n_ep_k, :]
            local_mask2 = mask2_2d[ep_offset : ep_offset + n_ep_k, :]

            if not np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float64, copy=True)

            for ch in eye1_indices:
                if 0 <= ch < n_ch:
                    arr[:, ch, :][local_mask1] = np.nan
            for ch in eye2_indices:
                if 0 <= ch < n_ch:
                    arr[:, ch, :][local_mask2] = np.nan

            obj[key] = arr
            ep_offset += n_ep_k

    elif isinstance(obj, np.ndarray):
        if obj.ndim == 3:
            obj = _apply_to_3d_array(obj)
        elif obj.ndim == 2:
            obj = _apply_to_2d_array(obj)
        else:
            raise ValueError(
                f"Unsupported ndarray shape {obj.shape}; expected 2D or 3D."
            )
    else:
        raise ValueError(
            "Unsupported pickle content type. "
            "Expected dict of arrays or numpy ndarray (2D or 3D)."
        )

    # Save cleaned object
    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
    with open(output_pkl_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Cleaned pickle saved to: {output_pkl_path}")


# -------------------------------------------------------------------------
# GUI builder (slider-based segment selection)
# -------------------------------------------------------------------------


def _build_gui(
    input_pkl_path: str,
    data_eye: np.ndarray,
    sfreq: float,
    epoch_len_samples: int,
    ch_names_eye: List[str],
    eye1_indices: Sequence[int],
    eye2_indices: Sequence[int],
) -> None:
    """Build and display the manual segment marking GUI (slider-based).

    Parameters
    ----------
    input_pkl_path : str
        Path to original pickle file (used later for saving).
    data_eye : ndarray, shape (2, n_times_total)
        Averaged Eye1/Eye2 continuous data.
    sfreq : float
        Sampling frequency.
    epoch_len_samples : int
        Epoch length in samples for the original data.
    ch_names_eye : list of str
        Names of averaged channels (e.g. ["Eye1_avg", "Eye2_avg"]).
    eye1_indices, eye2_indices : sequence of int
        Channel indices for Eye1 and Eye2 groups (to which NaNs are applied).
    """
    n_channels, n_times = data_eye.shape
    total_dur_s = n_times / sfreq
    times = np.arange(n_times) / sfreq

    # State: segments
    # segments: list of dicts { "ch": int, "start": int, "end": int }
    state = {
        "segments": [],
    }

    # Widgets: channel & window controls
    ch_options = [(f"{ch_names_eye[i]} (idx {i})", i) for i in range(n_channels)]
    ch_select = widgets.Dropdown(
        options=ch_options,
        value=0,
        description="Channel:",
        layout=widgets.Layout(width="280px"),
    )

    win_len_slider = widgets.FloatSlider(
        value=20.0,
        min=1.0,
        max=min(120.0, total_dur_s),
        step=1.0,
        description="Window (s):",
        readout_format=".1f",
        continuous_update=False,
        layout=widgets.Layout(width="400px"),
    )

    center_slider = widgets.FloatSlider(
        value=min(10.0, total_dur_s / 2.0),
        min=0.0,
        max=total_dur_s,
        step=0.5,
        description="Center (s):",
        readout_format=".1f",
        continuous_update=False,
        layout=widgets.Layout(width="400px"),
    )

    # Segment range slider (absolute time)
    segment_slider = widgets.FloatRangeSlider(
        value=[0.0, min(5.0, total_dur_s)],
        min=0.0,
        max=total_dur_s,
        step=0.01,
        description="Segment (s):",
        readout_format=".2f",
        continuous_update=False,
        layout=widgets.Layout(width="600px"),
    )

    # Segment controls
    add_segment_button = widgets.Button(
        description="Add segment", button_style="success"
    )
    remove_segment_button = widgets.Button(
        description="Remove selected segment", button_style="warning"
    )

    # Output dir / filename controls
    default_dir = os.path.dirname(input_pkl_path)
    in_base = os.path.splitext(os.path.basename(input_pkl_path))[0]
    output_dir_text = widgets.Text(
        value=default_dir,
        description="Output dir:",
        layout=widgets.Layout(width="600px"),
    )
    output_name_text = widgets.Text(
        value=f"{in_base}_nan_cleaned.pkl",
        description="File name:",
        layout=widgets.Layout(width="400px"),
    )
    clean_export_button = widgets.Button(
        description="Clean & export as .pkl", button_style="primary"
    )

    # Segments list + summary
    segments_select = widgets.Select(
        options=[], description="Segments:", rows=8, layout=widgets.Layout(width="650px")
    )
    seg_summary = widgets.HTML()

    plot_output = widgets.Output()
    log_output = widgets.Output()

    def _refresh_segments_widgets():
        """Refresh the segments list widget and summary."""
        options = []
        total_eye1 = 0.0
        total_eye2 = 0.0
        for idx, seg in enumerate(state["segments"]):
            ch = seg["ch"]
            s = seg["start"]
            e = seg["end"]
            start_s = s / sfreq
            end_s = e / sfreq
            dur_s = max(0.0, end_s - start_s)
            label = (
                f"[{idx}] {ch_names_eye[ch]}: "
                f"{start_s:.3f}–{end_s:.3f} s "
                f"(samples {s}-{e})"
            )
            options.append((label, idx))
            if ch == 0:
                total_eye1 += dur_s
            elif ch == 1:
                total_eye2 += dur_s
        segments_select.options = options

        seg_summary.value = (
            f"<b>Total segments:</b> {len(state['segments'])}<br>"
            f"Eye1_avg total marked: {total_eye1:.2f} s<br>"
            f"Eye2_avg total marked: {total_eye2:.2f} s"
        )

    def update_plot(*_):
        """Redraw the plot for current widget settings."""
        with plot_output:
            clear_output(wait=True)

            win_len = max(0.5, float(win_len_slider.value))
            center = float(center_slider.value)
            t0 = max(0.0, center - win_len / 2.0)
            t1 = min(total_dur_s, center + win_len / 2.0)
            if t1 <= t0:
                t1 = min(total_dur_s, t0 + 0.5)

            i0 = int(round(t0 * sfreq))
            i1 = int(round(t1 * sfreq))
            if i1 <= i0:
                i1 = min(i0 + 1, n_times)

            t_seg = times[i0:i1]
            ch_idx = int(ch_select.value)
            y = data_eye[ch_idx, i0:i1]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(t_seg, y, linewidth=0.8, color="black")

            # Draw existing segments for this channel
            for seg in state["segments"]:
                if seg["ch"] != ch_idx:
                    continue
                s = seg["start"]
                e = seg["end"]
                # Overlap with current window
                if e <= i0 or s >= i1:
                    continue
                s_clamp = max(s, i0)
                e_clamp = min(e, i1)
                ax.axvspan(
                    s_clamp / sfreq,
                    e_clamp / sfreq,
                    color="red",
                    alpha=0.2,
                    linewidth=0,
                )

            # Draw current slider segment for this channel
            seg_start_s, seg_end_s = segment_slider.value
            seg_start_s = max(0.0, min(total_dur_s, seg_start_s))
            seg_end_s = max(0.0, min(total_dur_s, seg_end_s))
            if seg_end_s > seg_start_s:
                s_samp = int(round(seg_start_s * sfreq))
                e_samp = int(round(seg_end_s * sfreq))
                if e_samp > i0 and s_samp < i1:
                    s_clamp = max(s_samp, i0)
                    e_clamp = min(e_samp, i1)
                    ax.axvspan(
                        s_clamp / sfreq,
                        e_clamp / sfreq,
                        color="red",
                        alpha=0.35,
                        linewidth=0,
                    )

            ax.set_xlim(t_seg[0], t_seg[-1])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (a.u.)")
            ax.set_title(
                f"{ch_names_eye[ch_idx]} | Window {t0:.2f}–{t1:.2f} s (len {win_len:.2f} s)"
            )
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

    def on_add_segment_clicked(_):
        ch_idx = int(ch_select.value)
        start_s, end_s = segment_slider.value
        start_s = float(start_s)
        end_s = float(end_s)
        if end_s <= start_s:
            with log_output:
                clear_output(wait=True)
                print("Cannot add segment: end time must be > start time.")
            return

        s = int(round(start_s * sfreq))
        e = int(round(end_s * sfreq))
        s = max(0, min(n_times, s))
        e = max(0, min(n_times, e))
        if e <= s:
            with log_output:
                clear_output(wait=True)
                print("Segment collapsed after clamping; not added.")
            return

        state["segments"].append({"ch": ch_idx, "start": s, "end": e})
        _refresh_segments_widgets()
        update_plot()

    def on_remove_segment_clicked(_):
        if not segments_select.options:
            return
        idx = segments_select.value
        if idx is None:
            return
        if idx < 0 or idx >= len(state["segments"]):
            return
        state["segments"].pop(idx)
        _refresh_segments_widgets()
        update_plot()

    def on_clean_export_clicked(_):
        out_dir = output_dir_text.value.strip()
        out_name = output_name_text.value.strip()
        if not out_name:
            out_name = "cleaned_nan.pkl"
        if not out_dir:
            out_dir = os.path.dirname(input_pkl_path)
        output_path = os.path.join(out_dir, out_name)

        with log_output:
            clear_output(wait=True)
            if not state["segments"]:
                print(
                    "No segments defined. Nothing will be changed, but a copy of the "
                    "original pickle will still be written."
                )
            else:
                print(f"Applying NaNs to {len(state['segments'])} segments...")
            print(f"Output path: {output_path}")

        _apply_nan_segments_to_pickle(
            input_pkl_path=input_pkl_path,
            output_pkl_path=output_path,
            segments=state["segments"],
            n_times_total=n_times,
            epoch_len_samples=epoch_len_samples,
            eye1_indices=eye1_indices,
            eye2_indices=eye2_indices,
        )

        with log_output:
            print("Done.")

    # Wire callbacks
    ch_select.observe(update_plot, names="value")
    win_len_slider.observe(update_plot, names="value")
    center_slider.observe(update_plot, names="value")
    segment_slider.observe(update_plot, names="value")

    add_segment_button.on_click(on_add_segment_clicked)
    remove_segment_button.on_click(on_remove_segment_clicked)
    clean_export_button.on_click(on_clean_export_clicked)

    # Initial draw
    _refresh_segments_widgets()
    update_plot()

    # Layout
    controls_row1 = widgets.HBox([ch_select])
    controls_row2 = widgets.HBox([win_len_slider, center_slider])

    seg_slider_row = widgets.HBox([segment_slider])
    seg_buttons = widgets.HBox(
        [add_segment_button, remove_segment_button]
    )

    output_path_row = widgets.HBox([output_dir_text, output_name_text])

    gui = widgets.VBox(
        [
            widgets.HTML(
                value="<h3>Manual Segment-Based NaN Cleaner (Eye1/Eye2 averages, slider-based)</h3>"
            ),
            controls_row1,
            controls_row2,
            plot_output,
            seg_slider_row,
            seg_buttons,
            segments_select,
            seg_summary,
            widgets.HTML("<b>Export cleaned recording (NaNs inserted at segments)</b>"),
            output_path_row,
            clean_export_button,
            log_output,
        ]
    )

    display(gui)


# -------------------------------------------------------------------------
# Public entry point
# -------------------------------------------------------------------------


def launch_manual_segment_cleaner(
    input_pkl_path: str,
    sfreq: float = 2000.0,
    epoch_len_s: float = 5.0,
    *,
    average_eye_groups: bool = True,
    eye1_indices: Sequence[int] = tuple(range(0, 8)),
    eye2_indices: Sequence[int] = tuple(range(8, 16)),
    drop_indices_eye1: Optional[Sequence[int]] = None,
    drop_indices_eye2: Optional[Sequence[int]] = None,
) -> None:
    """Launch the manual Eye1/Eye2 segment marking GUI (slider-based).

    Parameters
    ----------
    input_pkl_path : str
        Path to the input pickle file.
    sfreq : float
        Sampling frequency (Hz).
    epoch_len_s : float
        Epoch length (seconds) used for the original epochs.
    average_eye_groups : bool
        If True (recommended), compute Eye1/Eye2 averages from the original channels
        before displaying. Currently this is the intended mode.
    eye1_indices, eye2_indices : sequence of int
        Original channel indices belonging to Eye1 and Eye2.
    drop_indices_eye1, drop_indices_eye2 : sequence of int | None
        Channels to drop from Eye1/Eye2 averages before computing Eye1_avg/Eye2_avg.

    Notes
    -----
    - This function does not use automatic artifact criteria; it purely relies on
      manual marking of [start, end] segments using the Segment (s) range slider.
    - The NaN replacement is applied to the underlying original channels belonging
      to each Eye group, and the cleaned data is saved as a pickle with the same
      internal structure as the input.
    """
    if not os.path.isfile(input_pkl_path):
        raise FileNotFoundError(f"Input pickle file not found: {input_pkl_path}")

    # Load continuous representation
    data_cont, epoch_len_samples, ch_names = load_pickle_as_continuous(
        input_pkl_path,
        sfreq=sfreq,
        epoch_len_s=epoch_len_s,
    )

    if average_eye_groups:
        data_eye, ch_names_eye = _compute_eye_averages(
            data_cont=data_cont,
            ch_names=ch_names,
            eye1_indices=eye1_indices,
            eye2_indices=eye2_indices,
            drop_indices_eye1=drop_indices_eye1,
            drop_indices_eye2=drop_indices_eye2,
        )
    else:
        if data_cont.shape[0] < 2:
            raise ValueError(
                "Data has fewer than 2 channels; cannot show Eye1/Eye2 view."
            )
        data_eye = data_cont[:2, :]
        ch_names_eye = ["Ch1", "Ch2"]

    _build_gui(
        input_pkl_path=input_pkl_path,
        data_eye=data_eye,
        sfreq=sfreq,
        epoch_len_samples=epoch_len_samples,
        ch_names_eye=ch_names_eye,
        eye1_indices=eye1_indices,
        eye2_indices=eye2_indices,
    )
