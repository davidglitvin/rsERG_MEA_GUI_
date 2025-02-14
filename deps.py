# deps.py

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import pandas as pd
import mne
import numpy as np
import os
from mne.preprocessing import ICA
from sklearn.decomposition import PCA, FastICA
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D
from mne.datasets import sample
from mne.decoding import UnsupervisedSpatialFilter
from mne.minimum_norm import compute_source_psd, read_inverse_operator
from mne.io import read_raw_fif
from mne.datasets import sample
from mne.decoding import UnsupervisedSpatialFilter
from mne.minimum_norm import compute_source_psd, read_inverse_operator

# from mne_connectivity import spectral_connectivity_time
# from mne_connectivity import spectral_connectivity_epochs

# Import NeuroDSP functions
from neurodsp.spectral import compute_spectrum, trim_spectrum
from neurodsp.burst import detect_bursts_dual_threshold
from neurodsp.rhythm import compute_lagged_coherence

from neurodsp.plts import (
    plot_time_series, plot_power_spectra,
    plot_bursts, plot_lagged_coherence
)

# Morlet Wavelets
from neurodsp.timefrequency.wavelets import compute_wavelet_transform

# Pyts decomposition
from pyts.decomposition import SingularSpectrumAnalysis
import pywt

# TQDM
from tqdm import tqdm
from scipy.signal import welch
from scipy.signal import find_peaks
from scipy.integrate import trapz

from mpl_toolkits.mplot3d import Axes3D  # noqa
from tqdm.notebook import trange, tqdm

# FOOOF
from fooof import FOOOF, FOOOFGroup
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fg
from fooof.utils import trim_spectrum
from fooof.utils.data import subsample_spectra
from fooof.sim.gen import gen_aperiodic
from fooof.data import FOOOFSettings
from fooof.plts.templates import plot_hist
from fooof.plts.spectra import plot_spectra
from fooof.plts.periodic import plot_peak_fits, plot_peak_params
from fooof.plts.aperiodic import plot_aperiodic_params, plot_aperiodic_fits
from fooof.plts.annotate import plot_annotated_peak_search
from fooof.analysis.error import compute_pointwise_error_fm, compute_pointwise_error_fg
from fooof.utils.download import fetch_fooof_data

# More NeuroDSP
from neurodsp.sim import sim_combined
from neurodsp.filt import filter_signal
from neurodsp.plts import plot_time_series
from neurodsp.plts import plot_timefrequency
from neurodsp.utils import create_times

# bycycle
from bycycle import Bycycle, BycycleGroup
from bycycle.plts import plot_burst_detect_summary, plot_feature_categorical
from bycycle.plts import plot_feature_hist
from bycycle.cyclepoints import find_extrema, find_zerox
from bycycle.cyclepoints.zerox import find_flank_zerox
from bycycle.plts import plot_cyclepoints_array
from bycycle.utils.download import load_bycycle_data

# pptx
from pptx import Presentation
from pptx.util import Inches

# ipyfilechooser
from ipyfilechooser import FileChooser
