import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyedflib import EdfReader


# ----------------------------------------
# EEG Channel Mapping Utility
# ----------------------------------------

def list_eeg_channels(edf_path: str) -> pd.DataFrame:
    """
    Reads EEG channel labels from an EDF file and maps them to standard brain areas (10-20 system).
    Saves the channel metadata as JSON in the 'signals_keys' folder.

    Parameters
    ----------
    edf_path : str
        Path to the EDF file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing EEG channel metadata with columns:
        ['Index', 'Clean Label', 'Raw Label', 'Brain Area']
    """
    brain_areas = {
        'Fp': 'Prefrontal',
        'Af': 'Anterior frontal',
        'Fc': 'Fronto-central',
        'Ft': 'Fronto-temporal',
        'F':  'Frontal',
        'Cp': 'Centro-parietal',
        'C':  'Central (motor area)',
        'Tp': 'Temporo-parietal',
        'Po': 'Parieto-occipital',
        'P':  'Parietal',
        'T':  'Temporal',
        'Oz': 'Occipital',
        'O':  'Occipital (visual area)',
        'Cz': 'Central midline',
        'Iz': 'Inion (deep occipital)'
    }

    with EdfReader(edf_path) as f:
        raw_labels = f.getSignalLabels()

    channel_info = []
    for idx, raw_label in enumerate(raw_labels):
        clean_label = raw_label.strip().replace(".", "")
        # Extract prefix letters only (no digits)
        prefix = ''.join([c for c in clean_label if not c.isdigit()])
        # Lookup brain area: try first two letters, then first letter, else 'Unknown'
        area = brain_areas.get(prefix[:2], brain_areas.get(prefix[:1], 'Unknown'))

        channel_info.append({
            # 'Index': idx,
            'Clean Label': clean_label,
            'Raw Label': raw_label.strip(),
            'Brain Area': area
        })

    df = pd.DataFrame(channel_info)

    # Save metadata as JSON
    output_dir = 'signals_keys'
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'eeg_channels.json')
    with open(json_path, 'w') as json_file:
        json.dump(df.to_dict(orient='records'), json_file, indent=4)

    print(f"✅ EEG channel metadata saved to '{json_path}'")
    return df


# ----------------------------------------
# EDF Dataset Loader
# ----------------------------------------

def load_edf_dataset(folder_path: str, stop_subject: str | None = None) -> list[dict]:
    """
    Recursively loads EDF files from a directory, extracting signals, annotations, and metadata.

    Parameters
    ----------
    folder_path : str
        Path to directory containing EDF files.
    stop_subject : str or None
        Optional substring to stop loading files once encountered in filename.

    Returns
    -------
    list of dict
        Each dict contains:
            - filename: EDF filename
            - channels: list of channel labels
            - sample_rates: list of sampling frequencies
            - signals: dict of {channel_label: np.ndarray}
            - annotations: tuple of (onsets, durations, labels)
            - time: np.ndarray time vector for signals
    """
    edf_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.edf'):
                edf_files.append(os.path.join(root, file))

    if stop_subject:
        # Truncate list at first occurrence of stop_subject in filename
        for i, filepath in enumerate(edf_files):
            if stop_subject in os.path.basename(filepath):
                edf_files = edf_files[:i]
                break

    print(f"📁 Found {len(edf_files)} EDF files to load.")

    edf_data_list = []
    for filepath in tqdm(edf_files, desc="Loading EDF files"):
        try:
            with EdfReader(filepath) as f:
                n_signals = f.signals_in_file
                channels = f.getSignalLabels()
                sample_rates = f.getSampleFrequencies()
                annotations = f.readAnnotations()
                signals = [f.readSignal(i) for i in range(n_signals)]

            duration = len(signals[0]) / sample_rates[0]
            time = np.linspace(0, duration, len(signals[0]))

            edf_data_list.append({
                "filename": os.path.basename(filepath),
                "channels": channels,
                "sample_rates": sample_rates,
                "signals": {label: signals[i] for i, label in enumerate(channels)},
                "annotations": list(zip(*annotations)),
                "time": time
            })
        except Exception as e:
            print(f"⚠️ Error loading '{filepath}': {e}")

    return edf_data_list




# ----------------------------------------
# Signal Segmentation Utility
# ----------------------------------------

def segment_signal(signal: np.ndarray, window_size_samples: int, step_size_samples: int) -> list[np.ndarray]:
    """
    Splits a 1D signal into overlapping segments/windows.

    Parameters
    ----------
    signal : np.ndarray
        1D array of samples.
    window_size_samples : int
        Number of samples in each window.
    step_size_samples : int
        Step size between windows.

    Returns
    -------
    list of np.ndarray
        List of signal windows.
    """
    segments = []
    for start in range(0, len(signal) - window_size_samples + 1, step_size_samples):
        segments.append(signal[start:start + window_size_samples])
    return segments


# ----------------------------------------
# Create DataFrame from Epochs
# ----------------------------------------

def create_dataframe_from_epochs(
    epochs: list[np.ndarray],
    channels: list[str],
    subject_ids: list[int] | None = None,
    window_sec: float | None = None
) -> pd.DataFrame:
    """
    Builds a DataFrame where each row corresponds to one epoch,
    columns correspond to channels with mean signal values.

    Parameters
    ----------
    epochs : list of np.ndarray
        Each epoch contains concatenated signals from selected channels.
    channels : list of str
        Channel names in order matching the epochs.
    subject_ids : list or None
        Optional list of subject/run identifiers per epoch.
    window_sec : float or None
        Optional window duration in seconds.

    Returns
    -------
    pd.DataFrame
        DataFrame with mean feature per channel, and optional subject_id and window_sec.
    """
    n_channels = len(channels)
    n_samples_per_channel = epochs[0].shape[0] // n_channels

    data_rows = []
    for i, epoch in enumerate(epochs):
        row = {}
        for ch_idx in range(n_channels):
            start = ch_idx * n_samples_per_channel
            end = (ch_idx + 1) * n_samples_per_channel
            ch_signal = epoch[start:end]
            row[channels[ch_idx]] = np.mean(ch_signal)
        if subject_ids is not None:
            row['subject_id'] = subject_ids[i]
        if window_sec is not None:
            row['window_sec'] = window_sec
        data_rows.append(row)

    return pd.DataFrame(data_rows)


# ----------------------------------------
# Extract EEG Epochs for Imagery vs Rest Classification
# ----------------------------------------

def extract_epochs_imagery_vs_rest(
    edf_data_list: list[dict],
    selected_channels: list[str] | None = None,
    rest_label: str = "T0",
    imagery_labels: tuple[str, ...] = ("T1", "T2"),
    t_min: float = -0.5,
    t_max: float = 2.0,
    fs: int = 160,
    n_subjects: int | None = None,
    save_dataset: bool = True,
    save_dir: str = "saved_datasets",
    save_basename: str = "imagery_vs_rest_dataset"
) -> tuple[list[np.ndarray], list[int], pd.DataFrame]:
    """
    Extracts fixed-length EEG epochs around rest and imagery events from EDF data.
    Returns concatenated channel signals, binary labels, and a DataFrame of features.
    Optionally saves the dataset to disk.

    Parameters
    ----------
    edf_data_list : list of dict
        EDF data loaded from `load_edf_dataset`.
    selected_channels : list of str, optional
        Channels to extract.
    rest_label : str, optional
        Annotation label for rest state (default: "T0").
    imagery_labels : tuple of str, optional
        Labels corresponding to motor imagery tasks (default: ("T1", "T2")).
    t_min : float, optional
        Epoch start relative to event onset in seconds (default: -0.5).
    t_max : float, optional
        Epoch end relative to event onset in seconds (default: 2.0).
    fs : int, optional
        Sampling frequency in Hz (default: 160).
    n_subjects : int or None, optional
        Number of subjects for filename suffix (default: None).
    save_dataset : bool, optional
        Whether to save the dataset to disk (default: True).
    save_dir : str, optional
        Directory to save datasets (default: "saved_datasets").
    save_basename : str, optional
        Base filename for saved datasets (default: "imagery_vs_rest_dataset").

    Returns
    -------
    tuple
        X: list of np.ndarray, concatenated epochs signals
        y: list of int, labels (0=rest, 1=imagery)
        df: pd.DataFrame, mean features per channel and metadata
    """
    if selected_channels is None:
        selected_channels = ["C3", "C4", "Cz"] # Default channels

    X, y, run_ids = [], [], []
    epoch_length = int((t_max - t_min) * fs)

    for run_id, edf in enumerate(tqdm(edf_data_list, desc="Extracting epochs")):
        # Normalize channel keys for matching
        norm_map = {ch.strip(".").upper(): ch for ch in edf["channels"]}
        time = edf["time"]

        if not edf["annotations"]:
            continue

        onsets = [float(a[0]) for a in edf["annotations"]]
        labels = [str(a[2]) for a in edf["annotations"]]

        for onset, label in zip(onsets, labels):
            if label == rest_label:
                target_label = 0
            elif label in imagery_labels:
                target_label = 1
            else:
                continue  # skip unknown labels

            epoch_signals = []
            valid_epoch = True

            for ch in selected_channels:
                ch_key = norm_map.get(ch.upper())
                if ch_key is None:
                    valid_epoch = False
                    break

                signal = edf["signals"][ch_key]
                start_idx = np.searchsorted(time, onset + t_min)
                end_idx = start_idx + epoch_length

                if end_idx <= len(signal):
                    epoch = signal[start_idx:end_idx]
                    epoch_signals.append(epoch)
                else:
                    valid_epoch = False
                    break

            if valid_epoch and len(epoch_signals) == len(selected_channels):
                epoch_concat = np.concatenate(epoch_signals)
                X.append(epoch_concat)
                y.append(target_label)
                run_ids.append(run_id)

    print(f"✅ Extraction complete: {len(X)} epochs (Imagery: {sum(y)}, Rest: {len(y) - sum(y)})")

    df = create_dataframe_from_epochs(X, selected_channels, subject_ids=run_ids, window_sec=(t_max - t_min))
    df.rename(columns={'subject_id': 'run_id'}, inplace=True)
    df = df.sort_values(by='run_id').reset_index(drop=True)

    # Compute subject_id with offset after run_id 14 (subject changes every 14 runs)
    df['subject_id'] = ((df['run_id'] - 1) // 14).clip(lower=0).astype(int)

    if n_subjects is not None:
        save_basename = f"{save_basename}_{n_subjects}S"

    if save_dataset:
        os.makedirs(save_dir, exist_ok=True)

        # Save NumPy arrays
        npz_path = os.path.join(save_dir, save_basename + ".npz")
        np.savez_compressed(npz_path, X=np.array(X), y=np.array(y))
        print(f"✅ Numpy dataset saved to: {npz_path}")

        # Save DataFrame
        pkl_path = os.path.join(save_dir, save_basename + ".pkl")
        df.to_pickle(pkl_path)
        print(f"✅ DataFrame saved to: {pkl_path}")

    return X, y, df


# ----------------------------------------
# Additional Utility Functions
# ----------------------------------------

def show_channel_samples(edf_data_list: list[dict], channel: str) -> None:
    """
    Prints first 10 samples from a specified EEG channel in the first EDF file.

    Parameters
    ----------
    edf_data_list : list of dict
        EDF data list.
    channel : str
        Channel name (e.g., "C3").
    """
    edf = edf_data_list[0]
    norm_channels = {ch.strip(".").upper(): ch for ch in edf["channels"]}
    available = ", ".join(norm_channels.keys())
    print(f"\n📈 Available channels: {available}")

    target = channel.upper()
    if target in norm_channels:
        real_name = norm_channels[target]
        signal = edf["signals"][real_name]
        print(f"Samples from {real_name} [0:10]: {np.round(signal[:10], 2)}")
    else:
        print(f"⚠️ Channel '{channel}' not found in first EDF file.")


def plot_channel_signal(edf_data_list: list[dict], channel: str) -> None:
    """
    Plots EEG signal from a specified channel in the first EDF file.

    Parameters
    ----------
    edf_data_list : list of dict
        EDF data list.
    channel : str
        Channel name (e.g., "C3").
    """
    edf = edf_data_list[0]
    norm_channels = {ch.strip(".").upper(): ch for ch in edf["channels"]}
    target = channel.upper()

    if target in norm_channels:
        signal = edf["signals"][norm_channels[target]]
        plt.figure(figsize=(10, 4))
        plt.plot(edf["time"], signal)
        plt.title(f"{channel} - {edf['filename']}")
        plt.xlabel("Time [s]")
        plt.ylabel("EEG Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print(f"⚠️ Channel '{channel}' not available for plotting.")


def print_annotations(edf_data_list: list[dict]) -> None:
    """
    Prints all annotations (events) from the first EDF file.

    Parameters
    ----------
    edf_data_list : list of dict
        EDF data list.
    """
    print("\n🧠 Annotations from first EDF file:")
    edf = edf_data_list[0]

    if not edf["annotations"]:
        print("  ⚠️ No annotations found.")
        return

    onsets, durations, labels = edf["annotations"]

    if len(onsets) == 0:
        print("  ℹ️ No events present.")
        return

    for onset, duration, label in zip(onsets, durations, labels):
        print(f"  - Event '{label}' at {onset:.2f}s lasting {duration:.2f}s")
