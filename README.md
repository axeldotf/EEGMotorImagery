# EEG Motor Imagery Classification for BCI Applications

**Author:** Alessandro Frullo

![Python 3.11.9](https://img.shields.io/badge/python-3.11.9-blue.svg)
![VS Code 1.102.0.0](https://img.shields.io/badge/vscode-1.102.0.0-87CEEB.svg)

[![AIMed_Project_Frullo.ipynb](https://img.shields.io/badge/AIMed_Project_Frullo.ipynb-00008B.svg)](https://github.com/axeldotf/EEGMotorImagery/blob/main/AIMed_Project_Frullo.ipynb)
[![Open in Colab](https://img.shields.io/badge/Open_in_Colab-orange.svg?logo=google-colab&logoColor=white)](https://colab.research.google.com/github/axeldotf/EEGMotorImagery/blob/main/AIMed_Project_Frullo.ipynb)

---

> ⚠️ **Educational Use Only**  
> This project is provided strictly for educational and research purposes. Commercial use of the code or data is not permitted without appropriate permissions.

## Project Overview

This repository implements an end‑to‑end machine learning pipeline for classifying EEG signals into **motor imagery** (imagined movement) vs. **rest** states. Accurate motor imagery detection is a key challenge in **Brain-Computer Interface (BCI)** research, with applications in neurorehabilitation and assistive technologies.

<p align="center">
  <img src="https://www.sify.com/wp-content/uploads/2024/03/brain_computer_interface_freepik.jpg"
       alt="Brain-Computer Interface (BCI)" width="610" />
  <img src="https://drive.google.com/uc?export=view&id=1Zloo6MGLD9dGHzN0yib4VXzSxLXvBxYy"
       alt="Electrodes position map" width="330" />
</p>


Key features:

- **Data preprocessing** of EDF‑formatted EEG recordings (PhysioNet Motor Movement/Imagery Dataset)
- **Epoch extraction** around event markers
- **Feature engineering** (mean band‑power per channel)
- **Model training** using a Random Forest classifier
- **Evaluation** with accuracy, ROC AUC, precision‑recall, confusion matrix, MCC, balanced accuracy
- **Interpretability** via SHAP value analysis
- Modular utilities in `utils/step*_tools.py` and logging/plotting functions

## Repository Structure
```

├── README.md                     # This file
├── Frullo_AIMed_Report.pdf       # Project report
├── AIMed_Project_Frullo.ipynb    # Project Notebook
├── requirements.txt              # Python dependencies
├── emmi_dataset/                 # Raw EDF files (not included)
│ └── files/                      # Folder containing subject subfolders
│  └── S001...                    # EDF files divided by subject
├── saved_datasets/               # .npz and .pkl datasets (created during execution, not included)
├── signals_keys/                 # .json for signal mapping (created during execution, not included)
├── utils/                        # Utility modules
│ ├── step1_tools.py              # Data loading & epoch extraction
│ ├── step2_tools.py              # Feature extraction & plotting
│ ├── step3_tools.py              # Model training & evaluation
│ └── step4_tools.py              # Interpretability & visualization via SHAP
├── output_img/                   # Generated figures (ROC, PR, SHAP, etc.)
├── top_features_shap/            # Main features related data and figures
└── log/                          # Run‑by‑run log files and metrics

````

## Getting Started

Follow these steps to set up and explore the project:

1. **Clone the repository**
   ```bash
   git clone https://github.com/axeldotf/EEGMotorImagery.git
   ````

2. **Download and extract the dataset**

   * Go to the [EEG Motor Movement/Imagery Dataset v1.0.0](https://physionet.org/content/eegmmidb/1.0.0/) on PhysioNet and download the archive.
   * Extract the downloaded file to your local machine.

3. **Prepare the dataset folder**

   * Rename the extracted dataset folder to `emmi_dataset`.
   * Move `emmi_dataset` into the root of the cloned repository (the working folder).
   * Review the ***Repository Structure*** section above to confirm the correct layout of files and folders.

4. **Launch the notebook**

   * **Locally**: Open `AIMed_Project_Frullo.ipynb` with Jupyter Notebook or VS Code and start exploring the data and code to follow the analysis pipeline.  
   * **On Google Colab**:  
     1. Click the “Open in Colab” badge.
     2. Once the notebook loads, run the provided `Colab setup` cell to mount your Drive.  
     3. Update the dataset path to point at your Drive‑mounted working folder (e.g. `/content/drive/MyDrive/AIMed_Project/`), then proceed with the analysis.

## EDF+ Visualization Tool (Signals & Annotations)

To **manually inspect EDF+ files visually**, both raw signals and their **event/annotation tracks**, this repository includes a lightweight utility: `edf_reader.py`.

### What it does
- Loads EDF/EDF+ files (tested on PhysioNet EMMI data).
- Plots selected channels over a chosen time window.
- Displays and exports annotations (onset, duration, label).
- Works **both from the command line _and_ via a simple GUI** (launch it with no arguments).
- Optionally saves figures to `output_img/`.

### Quick usage

#### 1. Graphical mode (no arguments)
```bash
python edf_reader.py
````
A small GUI will open, just pick your EDF file, channels, time window, and whether to show/export annotations.

#### 2. From command line

```bash
# Basic: plot first 10 seconds of all channels
python edf_reader.py --edf emmi_dataset/files/S001/S001R01.edf --tstart 0 --tend 10

# Pick specific channels and show annotations
python edf_reader.py \
  --edf emmi_dataset/files/S001/S001R01.edf \
  --channels Fz,C3,C4 \
  --show-annotations

# Export annotations to CSV
python edf_reader.py --edf ... --export-annots annots_S001R01.csv
````

| Flag                 | Description                                   | Default            |
| -------------------- | --------------------------------------------- | ------------------ |
| `--edf`              | Path to the EDF/EDF+ file                     | **required** (CLI) |
| `--channels`         | Comma-separated list of channel names to plot | All channels       |
| `--tstart`, `--tend` | Start/end time in seconds for plotting        | Entire recording   |
| `--show-annotations` | Overlay annotations on the plot               | Off                |
| `--export-annots`    | Path to save annotations as CSV               | None               |
| `--outfig`           | Path to save the generated figure             | None               |

---

```text
# requirements.txt

numpy==2.0.2
pandas==2.2.3
matplotlib==3.10.0
scikit-learn==1.6.1
shap==0.48.0
pyedflib==0.1.40
tqdm==4.67.1
```
