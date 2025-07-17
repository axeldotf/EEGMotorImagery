from typing import Optional, Union, Tuple, Dict
import os
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import shap
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_shap_json(path: str) -> Dict:
    """
    Load and return the SHAP analysis JSON data from a file.

    Args:
        path: Path to the JSON file.
    Returns:
        Parsed JSON dictionary.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)



def process_data(
    json_data: Dict,
    top_n_areas: int = 5,
    top_n_prefixes: int = 15
) -> Tuple[Dict[str, float], Dict[str, float], int, int]:
    """
    Calculate aggregated percentages for areas and feature prefixes.

    Args:
        json_data: Parsed JSON containing "Areas" and "Features_Relative_Importance".
        top_n_areas: Number of top areas to return.
        top_n_prefixes: Number of top feature prefixes to return.

    Returns:
        A tuple with:
          - top_areas: Dict of top N areas and their percentages (rounded).
          - top_prefixes: Dict of top N feature prefixes and their percentages (rounded).
          - total_area_count: Total distinct areas in the input.
          - total_prefix_count: Total distinct prefixes in the input.
    """
    # Aggregate area values
    area_totals: defaultdict[str, float] = defaultdict(float)
    for area, value in json_data.get("Areas", {}).items():
        area_totals[area] += value

    total_area_sum = sum(area_totals.values())
    if total_area_sum > 0:
        for area in area_totals:
            area_totals[area] = (area_totals[area] / total_area_sum) * 100

    # Aggregate prefix values from feature importance
    prefix_totals: defaultdict[str, float] = defaultdict(float)
    for feature, value in json_data.get("Features_Relative_Importance", {}).items():
        prefix = feature.split('_')[0]
        prefix_totals[prefix] += value

    total_prefix_sum = sum(prefix_totals.values())
    if total_prefix_sum > 0:
        for prefix in prefix_totals:
            prefix_totals[prefix] = (prefix_totals[prefix] / total_prefix_sum) * 100

    # Sort and select top N
    sorted_areas = sorted(area_totals.items(), key=lambda x: x[1], reverse=True)
    sorted_prefixes = sorted(prefix_totals.items(), key=lambda x: x[1], reverse=True)

    top_areas = {k: round(v, 2) for k, v in sorted_areas[:top_n_areas]}
    top_prefixes = {k: round(v, 2) for k, v in sorted_prefixes[:top_n_prefixes]}

    return top_areas, top_prefixes, len(area_totals), len(prefix_totals)



from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import os

def make_color_map(keys: list[str], cmap_name: str = "tab20", other_color: str = "lightgray") -> dict[str, tuple]:
    """
    Crea una mappa di colori usando una colormap di Matplotlib.

    - keys: lista di etichette ordinate
    - cmap_name: nome della colormap
    - other_color: colore assegnato all'etichetta "Others"
    """
    cmap = get_cmap(cmap_name, len(keys))
    colors = [cmap(i) for i in range(len(keys))]
    m = dict(zip(keys, colors))
    m["Others"] = other_color
    return m


def plot_results(
    area_importance: dict,
    prefix_totals: dict,
    ctA: dict,
    ctP: dict,
    top_n_areas: int = 5,
    top_n_prefixes: int = 10,
    out_file: str = None
) -> plt.Figure:
    """
    Disegna due pie chart: uno per le aree e uno per i prefissi di feature.

    - area_importance: dict di percentuali per area
    - prefix_totals: dict di percentuali per prefisso
    - ctA: dict dei conteggi assoluti per area (opzionale)
    - ctP: dict dei conteggi assoluti per prefisso (opzionale)
    - top_n_areas: numero di aree principali da mostrare
    - top_n_prefixes: numero di prefissi principali da mostrare
    - out_file: percorso del file dove salvare la figura (se provided)

    Ritorna la figura Matplotlib.
    """
    # Genera le mappe colori univoche
    all_areas = sorted(area_importance.keys())
    all_prefixes = sorted(prefix_totals.keys())
    area_color_map = make_color_map(all_areas)
    feat_color_map = make_color_map(all_prefixes)

    # Preparazione figura
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)

    def draw_pie(ax, data: dict, title: str, color_map: dict, top_n: int) -> None:
        # Seleziona i top N basati sul valore
        sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:top_n]
        labels, sizes = zip(*top_items)
        labels, sizes = list(labels), list(sizes)

        # Calcola e aggiunge "Others" se rimane percentuale
        total_shown = sum(sizes)
        others = round(100 - total_shown, 2)
        if others > 0:
            labels.append("Others")
            sizes.append(others)

        # Assegna colori coerenti tramite la mappa
        colors = [color_map[label] for label in labels]

        ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140
        )
        ax.set_title(title)
        ax.axis('equal')

    # Disegna i due grafici
    draw_pie(
        axes[0],
        area_importance,
        f"Top {top_n_areas} Areas",
        area_color_map,
        top_n_areas
    )
    draw_pie(
        axes[1],
        prefix_totals,
        f"Top {top_n_prefixes} Feature Prefixes",
        feat_color_map,
        top_n_prefixes
    )

    # Salva figura se richiesto
    if out_file:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        fig.savefig(out_file)

    return fig


def save_shap_global_summary_and_get_importance(
    shap_values: Union[np.ndarray, shap.Explanation],
    X_data: pd.DataFrame,
    df_features_info: pd.DataFrame,
    n_subjects: int,
    top_k: Optional[int] = None,
    output_dir: str = "output_img",
    dpi: int = 300
) -> Tuple[pd.DataFrame, Dict]:
    """
    Save the global SHAP summary plot and return top features with their importance,
    plus a dict mapping each brain area to its relative importance percentage.

    Parameters
    ----------
    shap_values : np.ndarray or shap.Explanation
        SHAP values array [samples, features] or [samples, features, classes].
    X_data : pandas.DataFrame
        DataFrame for plot axis labels.
    df_features_info : pandas.DataFrame
        DataFrame with ['channel_sample', 'brain_area'] indexed by feature indices.
    n_subjects : int
        Number of subjects for filename and title.
    top_k : int or None
        Number of top features to return (default None returns all).
    output_dir : str
        Directory to save plots (default "output_img").
    dpi : int
        DPI for saved images (default 300).

    Returns
    -------
    Tuple[pandas.DataFrame, Dict]
        - DataFrame filtered by top_k with ['channel_sample','brain_area',
          'mean_abs_shap','rel_importance'].
        - Dict with Subjects, Top_K, Areas, Features_Relative_Importance.
    """
    os.makedirs(output_dir, exist_ok=True)
    shap_dir = os.path.join(output_dir, "shap_global")
    os.makedirs(shap_dir, exist_ok=True)

    plt.figure(figsize=(7, 10))

    # Extract SHAP values for class 1 if multi-class; else use all features
    if len(shap_values.shape) == 3:
        if shap_values.shape[2] > 1:
            shap_vals = shap_values[:, :, 1]
            shap.summary_plot(shap_vals, X_data, show=False)
        else:
            shap_vals = shap_values[:, :, 0]
            shap.summary_plot(shap_vals, X_data, show=False)
    else:
        shap_vals = shap_values
        shap.summary_plot(shap_vals, X_data, show=False)

    plt.title(f"# Subjects: {n_subjects}", fontsize=6)
    plt.tight_layout()

    filepath = os.path.join(shap_dir, f"shap_summary_global_{n_subjects}S.png")
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"✅ Global SHAP summary plot saved: {filepath}")

    # Compute mean absolute SHAP and sort
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    total_shap_sum = mean_abs_shap.sum()
    sorted_indices = np.argsort(mean_abs_shap)[::-1]

    top_indices = sorted_indices if top_k is None else sorted_indices[:top_k]

    df_filtered = df_features_info.loc[top_indices].copy()
    df_filtered['mean_abs_shap'] = mean_abs_shap[top_indices]

    df_filtered['rel_importance'] = df_filtered['mean_abs_shap'] / total_shap_sum

    area_importance = df_filtered.groupby('brain_area')['mean_abs_shap'].sum()
    total_area_importance = area_importance.sum()
    area_percent = 100 * area_importance / total_area_importance
    area_percent = area_percent.round(2)

    df_filtered['mean_abs_shap'] = df_filtered['mean_abs_shap'].apply(lambda x: round(float(x), 6))
    df_filtered['rel_importance'] = df_filtered['rel_importance'].apply(lambda x: round(float(x), 5))

    area_percent_dict = area_percent.to_dict()
    area_percent_dict_with_topk = {
        "Subjects": n_subjects,
        "Top_K": top_k,
        "Areas": area_percent_dict,
        "Features_Relative_Importance": {
            df_filtered.iloc[i]['channel_sample']: round(float(df_filtered.iloc[i]['rel_importance']), 5)
            for i in range(len(df_filtered))
        }
    }

    save_dict_as_json(area_percent_dict_with_topk, n_subjects, top_k)

    return df_filtered.reset_index(drop=False), area_percent_dict_with_topk



def save_dict_as_json(
    data_dict: Dict,
    n_subjects: int,
    top_k: Optional[int]
):
    """
    Save a dictionary as a JSON file in the folder 'top_features_shap'.

    Parameters
    ----------
    data_dict : dict
        Dictionary to save as JSON.
    n_subjects : int
        Number of subjects (used for filename).
    top_k : int or None
        Number of top features considered (used for filename).
    """
    folder_path = "top_features_shap"
    os.makedirs(folder_path, exist_ok=True)

    top_k_str = top_k if top_k is not None else "all"
    filename = f"top{top_k_str}_shap_{n_subjects}S.json"
    filepath = os.path.join(folder_path, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)

    print(f"✅ JSON saved to: {filepath}")



def create_feature_info_df(
    target_channels: list,
    n_features_total: int,
    json_path: str = "signals_keys/eeg_channels.json"
) -> pd.DataFrame:
    """
    Create a DataFrame mapping each feature index to its channel+sample name and brain area from JSON.

    Parameters
    ----------
    target_channels : list of str
        List of channel names (e.g., ["C3", "C4", ...]).
    n_features_total : int
        Total number of features (e.g., 3600).
    json_path : str, optional
        Path to JSON file containing channel to brain area mapping (default "signals_keys/eeg_channels.json").

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ['channel_sample', 'brain_area'], indexed by feature indices.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        eeg_channels_info = json.load(f)

    brain_area_map = {
        entry['Clean Label'].lower(): entry['Brain Area']
        for entry in eeg_channels_info
    }

    n_channels = len(target_channels)
    samples_per_channel = n_features_total // n_channels

    records = []
    for feature_index in range(n_features_total):
        channel_idx = feature_index // samples_per_channel
        sample_idx = feature_index % samples_per_channel
        channel_name = target_channels[channel_idx]

        channel_sample = f"{channel_name}_sample{sample_idx}"
        brain_area = brain_area_map.get(channel_name.lower(), "Unknown")

        records.append({
            "channel_sample": channel_sample,
            "brain_area": brain_area
        })

    df = pd.DataFrame(records)
    return df
