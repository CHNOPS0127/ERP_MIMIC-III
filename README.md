# ERP_MIMIC-III: Preprocessing & Modelling for Multimodal ICU LOS Prediction

Codebase for the dissertation **"Multimodal Learning for ICU Length-of-Stay Prediction: A Temporal and Multitasking Approach."**

This repository integrates ICU multimodal covariates from **MIMIC-III** through a comprehensive preprocessing pipeline, modelling and evaluate baselines (i.e., logistic regression and random forest), sequence models (i.e., BiLSTM and BiGRU), and a multitask framework across multiple configurations and observation window.

> **Attribution / Public Notice**
> 
> The job scripts in this repository are public. If you use or adapt them, please acknowledge the authour.

---

## Table of Contents

- [Data Access & Citation](#data-access--citation)
- [Requirements & Installation](#requirements--installation)
- [Environment Setup](#environment-setup)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Outputs & Directory Layout](#outputs--directory-layout)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Training Scripts (Models)](#training-scripts-models)
- [Reproducibility & System Notes](#reproducibility--system-notes)
- [License & Acknowledgments](#license--acknowledgments)

---

## Data Access & Citation

Datasets used in this dissertation are based on MIMIC-III, a large de-identified ICU database for credentialed users. Full references are included in the main paper.

**We do not provide the MIMIC-III database.** You must acquire the data yourself from https://mimic.physionet.org/

---

## Requirements & Installation

### Clone the Repository

```bash
# Clone the repository
git clone https://github.com/<USER_OR_ORG>/ERP_MIMIC-III.git
cd ERP_MIMIC-III

# Create & activate a virtual environment (bash/zsh)
python -m venv .venv && source .venv/bin/activate

# For Windows PowerShell
# .\.venv\Scripts\Activate.ps1
```

### System Requirements

- **Python**: ≥ 3.9 (tested on 3.9/3.10)
- **Storage**: Multi-GB CSV processing; recommend ≥100 GB free disk space
- **Memory**: Ample RAM recommended for large dataset processing

### Package Installation

```bash
pip install numpy pandas tqdm scikit-learn scipy torch pyarrow
```

*Note: pyarrow is optional but recommended for faster I/O operations.*

---

## Environment Setup

Set the following environment variables (edit paths to match your system):

```bash
export MIMIC3_ROOT="/path/to/mimic-iii-v1.4"
export ERP_ROOT="/path/to/erp_mimic"
export VARMAP="resources/updated_variable_selection_I.csv"
```

---

## Preprocessing Pipeline

The Subject Inclusion and Events Validation steps are adapted from the MIMIC-III benchmarks (Harutyunyan et al., 2019). Full reference is included in the dissertation.

### 1. Subject Inclusion (§3.2.1)

Takes raw MIMIC-III tables and generates per-subject directories with `stays.csv`, `diagnoses.csv`, and `events.csv`.

```bash
python extract_mimic3.py "$MIMIC3_ROOT" "$ERP_ROOT/subjects"
```

### 2. Events Validation (§3.2.2)

Takes per-subject `events.csv` and generates validated `events.csv` with reconciled `ICUSTAY_ID`s.

```bash
python validate_events.py "$ERP_ROOT/subjects"
```

### 3. Feature Selection & Data Cleaning (§3.2.3-3.2.4)

Generates integrated `static_data.csv` and unit-standardized, hour-aligned `wide_events.csv` with selected clinical variables.

```bash
python feature_selection.py "$ERP_ROOT"
python data_cleaning.py "$ERP_ROOT"
```

### 4. Final Preprocessing (§3.2.5-3.2.6)

Generates imputed and encoded `dynamic_data.csv`, `static_data.csv`, and `listfile.csv`.

```bash
python final_preprocessing.py "$ERP_ROOT"
```

### 5. Tensor Creation (§3.2.6)

Takes final processed data and generates model-ready tensors for sequence models. You may specify the prediction task and time window:

```bash
python tensor_creation.py "$ERP_ROOT" --task {los,multitask} --time_window {48,72}
```

### 6. Baseline Feature Engineering

Takes the final processed data and generates classical-model feature matrices:

```bash
python feature_engineering.py "$ERP_ROOT"
```

---

## Outputs & Directory Layout

### For Baseline Models:
- `features_lr.csv`
- `features_rf.csv`

### For Sequence Models:
- `X_padded_tensor_*.pt`
- `seq_lengths_*.pt`
- `static_tensor_*.pt`
- `y_total_class_tensor_*.pt`
- `icu_id_list_*.pt`

### Additional for Multitask:
- `y_mortality_tensor.pt`
- `y_hourly_tensor.pt`
- `hour_mask.pt`

---

## Exploratory Data Analysis

The following command:
- Combine **dynamic time-series data** from per-subject `wide_events.csv` files into a single `dynamic_data_eda.csv`.
- Ensure required clinical variables are included (adds missing columns as `NaN`).
- Merge **LOS (Length of Stay)** from `static_data.csv` into the combined dynamic dataset by `ICUSTAY_ID`.
- Generate plots and summary tables for:
  - Missingness patterns (static and dynamic data).
  - LOS and Remaining LOS distributions.
  - Dynamic variable trajectories (e.g., Heart Rate vs LOS groups).
  - Statistical associations between static diagnoses/demographics and LOS/mortality.


```bash
python eda.py --data-root "D:\DATA72000ERP\mimic3-data\data\trial\random_1000_subjects" \
--out-root "D:\DATA72000ERP\mimic3-data\data\trial\random_1000_subjects\eda" \
--combine-wide-events
```

---

## Training Scripts (Models)

### Baseline Models

Set tensor shortcuts for convenience:

```bash
export TENSOR_DIR="$ERP_ROOT/preprocessed/tensor"
export TW="48"  # or 72
```

#### Logistic Regression

Train multinomial logistic regression on LOS buckets:

```bash
python train_lr_los.py "$ERP_ROOT"
```

#### Random Forest

Train random forest on LOS buckets:

```bash
python train_rf_los.py "$ERP_ROOT"
```

### Sequence Models

#### BiLSTM-EndFuse

Train the model with cross-validation and bootstrap confidence intervals:

```bash
python bilstm_endfuse.py \
  --X_path "$TENSOR_DIR/X_padded_tensor_${TW}.pt" \
  --y_total_class_path "$TENSOR_DIR/y_total_class_tensor_${TW}.pt" \
  --static_path "$TENSOR_DIR/static_tensor_${TW}.pt" \
  --seq_lengths_path "$TENSOR_DIR/seq_lengths_${TW}.pt" \
  --icu_ids_path "$TENSOR_DIR/icu_id_list_${TW}.pt" \
  --results_dir "$ERP_ROOT/results/BiLSTM-EndFuse_${TW}" \
  --hidden_dim 128 \
  --num_layers 2 \
  --dropout 0.3 \
  --batch_size 64 \
  --learning_rate 1e-3
```

#### BiLSTM-AttenFuse

Train the model with attention mechanism:

```bash
python bilstm_attenfuse.py \
  --X_path "$TENSOR_DIR/X_padded_tensor_${TW}.pt" \
  --y_total_class_path "$TENSOR_DIR/y_total_class_tensor_${TW}.pt" \
  --static_path "$TENSOR_DIR/static_tensor_${TW}.pt" \
  --seq_lengths_path "$TENSOR_DIR/seq_lengths_${TW}.pt" \
  --icu_ids_path "$TENSOR_DIR/icu_id_list_${TW}.pt" \
  --results_dir "$ERP_ROOT/results/BiLSTM-AttenFuse_${TW}" \
  --hidden_dim 128 \
  --num_layers 2 \
  --dropout 0.3 \
  --batch_size 64 \
  --learning_rate 1e-3
```

#### BiGRU-EndFuse

Train bidirectional GRU with end fusion:

```bash
python bigru_endfuse.py \
  --X_path "$TENSOR_DIR/X_padded_tensor_${TW}.pt" \
  --y_total_class_path "$TENSOR_DIR/y_total_class_tensor_${TW}.pt" \
  --static_path "$TENSOR_DIR/static_tensor_${TW}.pt" \
  --seq_lengths_path "$TENSOR_DIR/seq_lengths_${TW}.pt" \
  --icu_ids_path "$TENSOR_DIR/icu_id_list_${TW}.pt" \
  --results_dir "$ERP_ROOT/results/BiGRU-EndFuse_${TW}"
```

#### BiGRU-AttenFuse

Train bidirectional GRU with attention fusion. Generates results directory with CV selection (linear κ), final model, test metrics, predictions, and bootstrap summary:

```bash
python bigru_attenfuse.py \
  --X_path "$TENSOR_DIR/X_padded_tensor_${TW}.pt" \
  --y_total_class_path "$TENSOR_DIR/y_total_class_tensor_${TW}.pt" \
  --static_path "$TENSOR_DIR/static_tensor_${TW}.pt" \
  --seq_lengths_path "$TENSOR_DIR/seq_lengths_${TW}.pt" \
  --icu_ids_path "$TENSOR_DIR/icu_id_list_${TW}.pt" \
  --results_dir "$ERP_ROOT/results/BiGRU-AttenFuse_${TW}" \
  --hidden_dim 128 \
  --num_layers 2 \
  --dropout 0.3 \
  --batch_size 64 \
  --learning_rate 1e-3 \
  --seed 42
```

#### Multitask BiLSTM

Train multitask model for total LOS, mortality, and hourly LOS prediction. Generates grid-searched/validated models, fold summaries, test metrics, and per-task prediction files:

```bash
python multitask.py \
  --X_path "$TENSOR_DIR/X_padded_tensor_${TW}.pt" \
  --y_total_class_path "$TENSOR_DIR/y_total_class_tensor_${TW}.pt" \
  --static_path "$TENSOR_DIR/static_tensor_${TW}.pt" \
  --seq_lengths_path "$TENSOR_DIR/seq_lengths_${TW}.pt" \
  --icu_ids_path "$TENSOR_DIR/icu_id_list_${TW}.pt" \
  --y_mortality_path "$TENSOR_DIR/y_mortality_tensor.pt" \
  --y_hourly_path "$TENSOR_DIR/y_hourly_tensor.pt" \
  --hour_mask_path "$TENSOR_DIR/hour_mask.pt" \
  --results_dir "$ERP_ROOT/results/Multitask_BiLSTM_${TW}" \
  --hidden_dim 128 \
  --num_layers 2 \
  --dropout 0.3 \
  --batch_size 64 \
  --learning_rate 1e-3 \
  --seed 42
```

---

## Reproducibility & System Notes

- All models use cross-validation and bootstrap confidence intervals for robust evaluation
- Random seeds are set for reproducibility where specified
- The preprocessing pipeline follows the methodology described in the dissertation sections referenced in each step
- Large memory requirements due to MIMIC-III dataset size - ensure adequate system resources

---

## License & Acknowledgments

This work builds upon the MIMIC-III benchmarks by Harutyunyan et al., 2019. Please cite the original MIMIC-III database and relevant benchmark papers when using this code.

For questions or issues, please open a GitHub issue or contact the repository maintainer.

---

**Note**: This codebase is specifically designed for the dissertation research and may require modifications for other use cases or datasets.
