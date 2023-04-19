import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy


def process(emg_signal, sampling_rate=1000, filterCutoff=100):

    # Sanitize input
    emg_signal = nk.signal_sanitize(emg_signal)

    # Clean signal
    # emg_cleaned = emg_clean(emg_signal, sampling_rate=sampling_rate)
    emg_cleaned = emg_clean(emg_signal, filterCutoff=filterCutoff, sampling_rate=sampling_rate)

    # Get amplitude
    amplitude = nk.emg_amplitude(emg_cleaned)

    # Get onsets, offsets, and periods of activity
    activity_signal, info = nk.emg_activation(
        amplitude, sampling_rate=sampling_rate, threshold="default"
    )
    info["sampling_rate"] = sampling_rate  # Add sampling rate in dict info

    # Prepare output
    signals = pd.DataFrame(
        {"EMG_Raw": emg_signal, "EMG_Clean": emg_cleaned, "EMG_Amplitude": amplitude}
    )

    signals = pd.concat([signals, activity_signal], axis=1)

    return signals, info


def emg_clean(emg_signal, sampling_rate=1000, filterCutoff=100):
    emg_signal = nk.as_vector(emg_signal)

    # Missing data
    n_missing = np.sum(np.isnan(emg_signal))
    if n_missing > 0:
        warn(
            "There are " + str(n_missing) + " missing data points in your signal."
            " Filling missing values by using the forward filling method.",
            category=nk.NeuroKitWarning,
        )
        emg_signal = _emg_clean_missing(emg_signal)

    # Parameters
    order = 4
    frequency = filterCutoff
    frequency = (
        2 * np.array(frequency) / sampling_rate
    )  # Normalize frequency to Nyquist Frequency (Fs/2).

    # Filtering
    b, a = scipy.signal.butter(N=order, Wn=frequency, btype="highpass", analog=False)
    filtered = scipy.signal.filtfilt(b, a, emg_signal)

    # Baseline detrending
    clean = nk.signal_detrend(filtered, order=0)

    return clean

# =============================================================================
# Handle missing data
# =============================================================================
def _emg_clean_missing(emg_signal):

    emg_signal = pd.DataFrame.pad(pd.Series(emg_signal))

    return emg_signal