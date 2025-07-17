# EEG Motor Imagery Classification for BCI Applications

**Author:** Alessandro Frullo

![Python 3.11.9](https://img.shields.io/badge/python-3.11.9-blue.svg)
[![Open Notebook](https://img.shields.io/badge/Open-Notebook-blue)](https://github.com/axeldotf/EEGMotorImagery/blob/main/AIMed_Project_Frullo.ipynb)

---

> ⚠️ **Educational Use Only**  
> This project is provided strictly for educational and research purposes. Commercial use of the code or data is not permitted without appropriate permissions.

## Project Overview

This repository implements an end‑to‑end machine learning pipeline for classifying EEG signals into **motor imagery** (imagined movement) vs. **rest** states. Accurate motor imagery detection is a key challenge in Brain–Computer Interface (BCI) research, with applications in neurorehabilitation and assistive technologies.

Key features:

- **Data preprocessing** of EDF‑formatted EEG recordings (PhysioNet Motor Movement/Imagery Dataset)
- **Epoch extraction** around event markers
- **Feature engineering** (mean band‑power per channel)
- **Model training** using a Random Forest classifier
- **Evaluation** with accuracy, ROC AUC, precision‑recall, confusion matrix, MCC, balanced accuracy
- **Interpretability** via SHAP value analysis
- Modular utilities in `utils/step*_tools.py` and logging/plotting functions

## Repository Structure
├── emmi_dataset/ # Raw EDF files (not included)
│ ├── files/ # folder containing every subject files
├── AIMed_Project_Frullo.ipynb # Main analysis notebook
├── utils/
│ ├── step1_tools.py # Data loading & epoch extraction
│ ├── step2_tools.py # Feature extraction & plotting
│ ├── step3_tools.py
│ └── step4_tools.py # Model training & evaluation
├── output_img/ # Generated figures (ROC, PR, SHAP, etc.)
├── log/ # Run‑by‑run log files and metrics
├── README.md # This file
└── requirements.txt # Python dependencies


## Getting Started

1. **Clone the repository**  
   ```bash
   git clone https://github.com/axeldotf/EEGMotorImagery

2. **Download dataset**

Register for and download the [EEG Motor Movement/Imagery Dataset v1.0.0 from PhysioNet](https://physionet.org/content/eegmmidb/1.0.0/). Extract the archive, name the folder `emmi_dataset` and place it in the working folder.

4. **Open Notebook**

Open the `.ipynb` file in the working folder to start analyzing the project.


```text
# requirements.txt

numpy==2.0.2
pandas==2.2.3
matplotlib==3.10.0
scikit-learn==1.6.1
shap==0.48.0
pyedflib==0.1.40
tqdm==4.67.1
