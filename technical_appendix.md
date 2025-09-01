# Technical Appendix: Multimodal Learning for ICU Length-of-Stay Prediction
## A Complete Reproducibility Guide

**Author ID:** 14158989  
**Project:** Multimodal Learning for ICU Length-of-Stay Prediction: A Temporal and Multitasking Approach  
**Repository:** https://github.com/CHNOPS0127/ERP_MIMIC-III

---

## Table of Contents

1. [System Requirements & Environment Setup](#1-system-requirements--environment-setup)
2. [Data Access & Prerequisites](#2-data-access--prerequisites)
3. [Complete Preprocessing Pipeline](#3-complete-preprocessing-pipeline)
4. [Exploratory Data Analysis Reproduction](#4-exploratory-data-analysis-reproduction)
5. [Model Implementation & Training](#5-model-implementation--training)
6. [Evaluation & Results Reproduction](#6-evaluation--results-reproduction)
7. [Expected Outputs & Verification](#7-expected-outputs--verification)

---

## 1. System Requirements & Environment Setup

### 1.1 Hardware Requirements

- **Memory:** Minimum 32GB RAM recommended for full dataset processing
- **Storage:** ≥100GB free disk space for MIMIC-III data and intermediate files
- **GPU:** CUDA-compatible GPU with ≥8GB VRAM recommended (Tesla V100 or equivalent)
- **CPU:** Multi-core processor (8+ cores recommended)

### 1.2 Software Dependencies

**Python Version:** 3.9 

**Core Dependencies:**
```bash
numpy>=1.21.0
pandas>=1.3.0
torch>=1.9.0
scikit-learn>=1.0.0
scipy>=1.7.0
tqdm>=4.62.0
pyarrow>=5.0.0  # Optional but recommended for faster I/O
matplotlib>=3.4.0
seaborn>=0.11.0
```

### 1.3 Environment Setup Procedure

```bash
# Clone repository
git clone https://github.com/CHNOPS0127/ERP_MIMIC-III.git
cd ERP_MIMIC-III

# Create virtual environment
python -m venv .venv

# Activate environment
# Linux/Mac:
source .venv/bin/activate
# Windows:
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install numpy pandas tqdm scikit-learn scipy torch pyarrow matplotlib seaborn
```

### 1.4 Environment Variables Configuration

Set the following environment variables (critical for reproducibility):

```bash
export MIMIC3_ROOT="/path/to/mimic-iii-clinical-database-1.4"
export ERP_ROOT="/path/to/your/working/directory"
export VARMAP="resources/updated_variable_selection_I.csv"
```

**Windows PowerShell:**
```powershell
$env:MIMIC3_ROOT = "C:\path\to\mimic-iii-clinical-database-1.4"
$env:ERP_ROOT = "C:\path\to\your\working\directory"
$env:VARMAP = "resources/updated_variable_selection_I.csv"
```

---

## 2. Data Access & Prerequisites

### 2.1 MIMIC-III Database Access

**Data Source:** MIMIC-III Clinical Database v1.4  
**Access:** https://mimic.physionet.org/  
**Requirements:** Completed CITI training and signed data use agreement

**Required MIMIC-III Files:**
- `ADMISSIONS.csv`
- `PATIENTS.csv`
- `ICUSTAYS.csv`
- `CHARTEVENTS.csv`
- `LABEVENTS.csv`
- `OUTPUTEVENTS.csv`
- `NOTEEVENTS.csv`
- `DIAGNOSES_ICD.csv`
- `D_ITEMS.csv`
- `D_LABITEMS.csv`
- `D_ICD_DIAGNOSES.csv`

### 2.2 Directory Structure Preparation

Create the following directory structure:

```
ERP_ROOT/
├── subjects/           # Will contain per-subject directories
├── preprocessed/       # Final processed data
│   └── tensor/        # Model-ready tensors
├── results/           # Model outputs and evaluation
├── eda/               # EDA outputs
└── resources/         # Configuration files
```

---

## 3. Complete Preprocessing Pipeline

### 3.1 Step 1: Subject Inclusion

**Objective:** Extract per-subject directories with stays, diagnoses, and events

**Command:**
```bash
python extract_mimic3.py "$MIMIC3_ROOT" "$ERP_ROOT/subjects"
```

**Process Details:**
- Excludes patients <18 years old
- Caps age at 90 (HIPAA compliance)
- Filters unit transfers (FIRST_CAREUNIT ≠ LAST_CAREUNIT)
- Removes multiple ICU stays per admission
- Computes AGE = INTIME - DOB

**Expected Outputs:**
- `$ERP_ROOT/subjects/{SUBJECT_ID}/stays.csv`
- `$ERP_ROOT/subjects/{SUBJECT_ID}/diagnoses.csv`
- `$ERP_ROOT/subjects/{SUBJECT_ID}/events.csv`

**Verification:** Should produce ~40,000 subject directories

### 3.2 Step 2: Events Validation

**Objective:** Reconcile ICUSTAY_IDs and clean event timestamps

**Command:**
```bash
python validate_events.py "$ERP_ROOT/subjects"
```

**Process Details:**
- Validates ADMISSION_ID and ICUSTAY_ID consistency
- Imputes missing IDs using stays.csv mapping
- Removes events with unreconcilable identifiers
- Filters negative hours since ICU admission (~0.75% of events)

**Expected Outputs:**
- Updated `events.csv` files with validated identifiers
- Timestamped log of validation statistics

### 3.3 Step 3: Feature Selection

**Objective:** Apply clinically-motivated feature selection

**Command:**
```bash
python feature_selection.py "$ERP_ROOT"
```

**Key Parameters:**
- **Vital Signs & Labs:** 28 variables (HR, BP, RR, SpO2, etc.)
- **Chronic Diagnoses:** 120 ICD-9 conditions (reduced to 24 post-EDA)
- **Demographics:** 7 variables (age, gender, ethnicity, etc.)
- **Text Features:** 26 TF-IDF terms from clinical notes

**Item Harmonization Examples:**
- `FIO2 [Meas]`, `Inspired O2 Fraction`, `Vision FiO2` → `Fraction inspired oxygen`
- Multiple BP measurements → Standardized SBP/DBP/MBP

**Expected Outputs:**
- `$ERP_ROOT/preprocessed/wide_events.csv` (per-subject time series)
- Feature selection logs and mapping files

### 3.4 Step 4: Data Cleaning & Temporal Alignment

**Objective:** Standardize units and create hourly time grid

**Command:**
```bash
python data_cleaning.py "$ERP_ROOT"
```

**Unit Standardization:**
- Weight: lb, oz → kg
- Temperature: °F → °C  
- Pressure: mmHg standardized, etc.

**Temporal Processing:**
- Compute hours since ICU admission: `HOUR = (CHARTTIME - INTIME)/3600`
- Floor to hourly bins (hour 0.0-0.99 → bin 0)
- Aggregate within bins using "last value carried forward"

**Expected Outputs:**
- Hour-aligned `wide_events.csv` files
- Unit conversion logs
- Missing data pattern summaries

### 3.5 Step 5: Final Preprocessing

**Objective:** Imputation, encoding, and dataset finalization

**Command:**
```bash
python final_preprocessing.py "$ERP_ROOT"
```

**Imputation Strategy:**
1. **Forward filling** within each encounter
2. **ICU-wide median** for variables without prior values  
3. **Binary missing masks** to preserve missingness information

**Static Data Processing:**
- Categorical variables: One-hot encoding
- Ordinal variables (GCS): Ordinal encoding  
- Continuous variables: Z-score standardization
- Missing categories encoded as separate classes

**Exclusion Criteria:**
- ICU stays ≤4 hours (insufficient for robust prediction)
- Encounters with >95% missing data

**Expected Outputs:**
- `$ERP_ROOT/preprocessed/dynamic_data.csv`
- `$ERP_ROOT/preprocessed/static_data.csv`  
- `$ERP_ROOT/preprocessed/listfile.csv`

### 3.6 Step 6: Tensor Creation

**Objective:** Generate model-ready tensors for sequence models

**Command:**
```bash
# For single-task LOS prediction (48-hour window)
python tensor_creation.py "$ERP_ROOT" --task los --time_window 48

# For multitask framework (48-hour window)  
python tensor_creation.py "$ERP_ROOT" --task multitask --time_window 48

# For 72-hour window
python tensor_creation.py "$ERP_ROOT" --task los --time_window 72
```

**LOS Bucketing Scheme:**
- Bucket 0: 0-1 day
- Bucket 1: 1-2 days  
- Bucket 2: 2-3 days
- Bucket 3: 3-4 days
- Bucket 4: 4-5 days
- Bucket 5: 5-7 days
- Bucket 6: 7-14 days
- Bucket 7: ≥14 days

**Expected Outputs:**
- `X_padded_tensor_*.pt` - Dynamic sequences [N, T, D]
- `seq_lengths_*.pt` - Valid sequence lengths [N]  
- `static_tensor_*.pt` - Static features [N, S]
- `y_total_class_tensor_*.pt` - LOS bucket labels [N]
- `icu_id_list_*.pt` - ICU stay identifiers [N]

**Additional for Multitask:**
- `y_mortality_tensor.pt` - Binary mortality labels [N]
- `y_hourly_tensor.pt` - Hourly remaining LOS [N, T]
- `hour_mask.pt` - Valid hour indicators [N, T]

### 3.7 Step 7: Baseline Feature Engineering

**Objective:** Create handcrafted features for traditional ML models

**Command:**
```bash
python feature_engineering.py "$ERP_ROOT"
```

**Feature Engineering Details:**
- **Time Windows:** First 10% and 25% of ICU stay
- **Statistics per Variable:** Mean, std, min, max, AUC, slope, delta, entropy, reversals
- **Total Features:** 28 variables × 9 statistics × 2 windows = 504 temporal features
- **Additional:** Static demographics and diagnosis indicators

**Expected Outputs:**
- `features_lr.csv` - RFECV-selected features for logistic regression
- `features_rf.csv` - Full feature set for random forest

---

## 4. Exploratory Data Analysis Reproduction

### 4.1 Combined EDA Execution

**Command:**
```bash
python eda.py \
    --data-root "$ERP_ROOT" \
    --out-root "$ERP_ROOT/eda" \
    --combine-wide-events
```

**Generated Analyses:**
1. **Distribution Analysis:** LOS and remaining LOS distributions, bucketed analysis
2. **Missingness Patterns:** Temporal and variable-wise missing data analysis  
3. **Trajectory Analysis:** Vital sign trajectories grouped by LOS
4. **Feature Association:** Early predictors vs. LOS correlation analysis
5. **Clinical Correlations:** GCS, diagnoses, demographics vs. outcomes

### 4.2 Key EDA Outputs

**Expected Files:**
- `dynamic_data_eda.csv` - Combined time series data
- `missingness_analysis.png` - Figure 2 reproduction
- `temporal_missingness.png` - Figure 3 reproduction
- `los_distributions.png` - Figure 4 reproduction
- `hr_trajectories.png` - Figure 6 reproduction
- `clinical_associations/` - Statistical test results

**Verification Metrics:**
- Total samples: ~40,187 encounters
- Missing data patterns match Figures 2-3
- LOS distribution: Right-skewed with median ~2.2 days

---

## 5. Model Implementation & Training

### 5.1 Baseline Models

#### 5.1.1 Logistic Regression

**Command:**
```bash
python train_lr_los.py "$ERP_ROOT"
```

**Implementation Details:**
- **Algorithm:** Multinomial logistic regression with L2 regularization
- **Solver:** LBFGS for multiclass problems
- **Feature Selection:** RFECV with 5-fold stratified CV
- **Class Balancing:** Class-weighted loss function
- **Hyperparameters:** C ∈ [0.001, 0.01, 0.1, 1.0, 10.0] via grid search

**Expected Performance:** κ = 0.6451 (95% CI: 0.6343-0.6563)

#### 5.1.2 Random Forest

**Command:**
```bash
python train_rf_los.py "$ERP_ROOT"
```

**Implementation Details:**
- **Trees:** 100 estimators with bootstrap sampling
- **Splitting:** Gini impurity criterion
- **Feature Selection:** No explicit selection (embedded in tree splits)
- **Class Balancing:** Class-weighted sample importance
- **Hyperparameters:** max_depth ∈ [10, 20, None], min_samples_split ∈ [2, 5, 10]

**Expected Performance:** κ = 0.6558 (95% CI: 0.6448-0.6673)

### 5.2 Sequence Models

#### 5.2.1 BiLSTM-EndFuse

**Command:**
```bash
export TENSOR_DIR="$ERP_ROOT/preprocessed/tensor"
export TW="48"  # or 72

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
  --learning_rate 1e-3 \
  --seed 42
```

**Architecture Details:**
- **Encoder:** 2-layer bidirectional LSTM (hidden_dim=128)
- **Static MLP:** 2 layers with BatchNorm and Dropout
- **Fusion:** Simple concatenation of final hidden state + static embedding
- **Classifier:** Single linear layer with softmax

**Training Configuration:**
- **Optimizer:** Adam with ReduceLROnPlateau scheduler
- **Loss:** Class-weighted cross-entropy with label smoothing
- **Validation:** 5-fold stratified GroupKFold cross-validation
- **Early Stopping:** Patience = 10 epochs

**Expected Performance:**
- 48h: κ = 0.7092 (95% CI: 0.6991-0.7188)
- 72h: κ = 0.8142 (95% CI: 0.8070-0.8210)

#### 5.2.2 BiLSTM-AttenFuse

**Command:**
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
  --learning_rate 1e-3 \
  --seed 42
```

**Architecture Details:**
- **Temporal Convolutions:** Kernel sizes [3,5,7,9], 64 filters each
- **Attention Mechanism:** Single-head attention with positional encoding
- **Pooling Fusion:** Learnable combination of attention, mean, and max pooling
- **Static Gating:** Multiplicative gating with feature and gate pathways
- **Classifier:** 2-layer MLP with residual connection

**Expected Performance:**
- 48h: κ = 0.7126 (95% CI: 0.7032-0.7219)
- 72h: κ = 0.8235 (95% CI: 0.8192-0.8303) **[Best Overall]**

#### 5.2.3 BiGRU Models

**Commands:**
```bash
# EndFuse
python bigru_endfuse.py \
  --X_path "$TENSOR_DIR/X_padded_tensor_${TW}.pt" \
  --y_total_class_path "$TENSOR_DIR/y_total_class_tensor_${TW}.pt" \
  --static_path "$TENSOR_DIR/static_tensor_${TW}.pt" \
  --seq_lengths_path "$TENSOR_DIR/seq_lengths_${TW}.pt" \
  --icu_ids_path "$TENSOR_DIR/icu_id_list_${TW}.pt" \
  --results_dir "$ERP_ROOT/results/BiGRU-EndFuse_${TW}"

# AttenFuse  
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

**Architecture Notes:**
- Replace LSTM cells with GRU cells (fewer parameters)
- Same fusion strategies as LSTM counterparts
- Better performance on 48h window due to simpler architecture

**Expected Performance:**
- BiGRU-EndFuse 48h: κ = 0.7102 (95% CI: 0.7006-0.7201)
- BiGRU-AttenFuse 48h: κ = 0.7209 (95% CI: 0.7103-0.7311)

### 5.3 Multitask Framework

**Command:**
```bash
python multitask.py \
  --X_path "$TENSOR_DIR/X_padded_tensor_48.pt" \
  --y_total_class_path "$TENSOR_DIR/y_total_class_tensor_48.pt" \
  --static_path "$TENSOR_DIR/static_tensor_48.pt" \
  --seq_lengths_path "$TENSOR_DIR/seq_lengths_48.pt" \
  --icu_ids_path "$TENSOR_DIR/icu_id_list_48.pt" \
  --y_mortality_path "$TENSOR_DIR/y_mortality_tensor.pt" \
  --y_hourly_path "$TENSOR_DIR/y_hourly_tensor.pt" \
  --hour_mask_path "$TENSOR_DIR/hour_mask.pt" \
  --results_dir "$ERP_ROOT/results/Multitask_BiLSTM_48" \
  --hidden_dim 128 \
  --num_layers 2 \
  --dropout 0.3 \
  --batch_size 64 \
  --learning_rate 1e-3 \
  --seed 42
```

**Architecture Details:**
- **Shared Encoder:** 2-layer BiLSTM with late fusion
- **Task Heads:**
  - Total LOS: 2-layer MLP → 8-class classification
  - Mortality: Single linear layer → binary classification  
  - Hourly LOS: Time-distributed MLP → per-timestep prediction
- **Loss Function:** Weighted sum (λ_total = λ_mort = λ_hourly = 1.0)

**Expected Performance:**
- Total LOS: κ = 0.7171 (95% CI: 0.7077-0.7272)
- Hourly LOS: κ = 0.5716 (95% CI: 0.5597-0.5852) 
- Mortality: AUC-ROC = 0.9556, AUC-PR = 0.8084

---

## 6. Evaluation & Results Reproduction

### 6.1 Cross-Validation Setup

**Procedure:**
1. **Data Split:** 80% train, 20% test (encounter-level, no leakage)
2. **Model Selection:** 5-fold stratified GroupKFold on training set
3. **Hyperparameter Grid:** Learning rate [1e-4, 1e-3, 1e-2], hidden_dim [64, 128, 256]
4. **Final Training:** Best hyperparameters on full training set
5. **Test Evaluation:** Single evaluation on held-out test set

### 6.2 Bootstrap Confidence Intervals

**Implementation:**
```python
# Pseudocode for bootstrap evaluation
def bootstrap_metrics(y_true, y_pred, n_bootstrap=1000):
    n_samples = len(y_true)
    metrics = []
    
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Compute metric
        metric = cohen_kappa_score(y_true_boot, y_pred_boot, weights='linear')
        metrics.append(metric)
    
    # Return 95% CI
    return np.percentile(metrics, [2.5, 97.5])
```

### 6.3 Interpretability Analysis

**Attention Weight Extraction (AttenFuse):**
```bash
# Attention weights automatically saved during training
# Location: $RESULTS_DIR/attention_analysis/
# - temporal_attention_weights.npy
# - channel_attention_weights.npy  
# - static_gate_weights.npy
```

**Visualization Generation:**
- Temporal attention over time steps (Figure 19)
- Channel-wise attention across variables (Figure 20)
- Static feature gating importance (Figure 21)

---

## 7. Expected Outputs & Verification

### 7.1 Directory Structure Post-Execution

```
ERP_ROOT/
├── subjects/
│   └── {SUBJECT_ID}/
│       ├── stays.csv
│       ├── diagnoses.csv
│       └── events.csv
├── preprocessed/
│   ├── dynamic_data.csv
│   ├── static_data.csv
│   ├── listfile.csv
│   ├── features_lr.csv
│   ├── features_rf.csv
│   └── tensor/
│       ├── X_padded_tensor_{48,72}.pt
│       ├── y_total_class_tensor_{48,72}.pt
│       ├── static_tensor_{48,72}.pt
│       ├── seq_lengths_{48,72}.pt
│       ├── icu_id_list_{48,72}.pt
│       ├── y_mortality_tensor.pt
│       ├── y_hourly_tensor.pt
│       └── hour_mask.pt
├── results/
│   ├── BiLSTM-EndFuse_{48,72}/
│   ├── BiLSTM-AttenFuse_{48,72}/
│   ├── BiGRU-EndFuse_{48,72}/
│   ├── BiGRU-AttenFuse_{48,72}/
│   └── Multitask_BiLSTM_48/
│       ├── cv_results.json
│       ├── test_metrics.json
│       ├── predictions.csv
│       ├── bootstrap_summary.json
│       └── attention_analysis/
└── eda/
    ├── figures/
    ├── statistical_tests/
    └── combined_datasets/
```

### 7.2 Performance Verification Checklist

**✓ Preprocessing Pipeline:**
- [ ] ~40,187 final samples after exclusions
- [ ] Missing data patterns match Figures 2-3
- [ ] Feature dimensions: Dynamic [N,T,28], Static [N,57]

**✓ Baseline Performance:**
- [ ] Logistic Regression: κ = 0.6451 ± 0.011
- [ ] Random Forest: κ = 0.6558 ± 0.011

**✓ Sequence Model Performance:**
- [ ] BiLSTM-EndFuse 48h: κ = 0.7092 ± 0.010
- [ ] BiLSTM-AttenFuse 72h: κ = 0.8235 ± 0.006 (Best)
- [ ] BiGRU-AttenFuse 48h: κ = 0.7209 ± 0.010

**✓ Multitask Performance:**
- [ ] Total LOS: κ = 0.7171 ± 0.010
- [ ] Hourly LOS: κ = 0.5716 ± 0.013
- [ ] Mortality: AUC-ROC = 0.9556, AUC-PR = 0.8084

**✓ Reproducibility Markers:**
- [ ] All random seeds set (42 for deterministic results)
- [ ] GPU determinism enabled where possible
- [ ] Cross-validation folds identical across runs
- [ ] Bootstrap samples reproducible with fixed seed

### 7.3 Common Issues & Troubleshooting

**Memory Issues:**
- Reduce batch size if OOM errors occur
- Use gradient checkpointing for large sequences
- Process data in chunks if preprocessing fails

**Performance Discrepancies:**
- Verify identical data splits (check ICU_IDs in train/test)
- Confirm hyperparameter settings match exactly
- Check PyTorch/CUDA versions for consistency

**Missing Dependencies:**
- Install exact package versions if results differ
- Use pip freeze > requirements.txt to capture exact environment

### 7.4 Final Verification Command

```bash
# Run complete pipeline verification
python verify_reproduction.py "$ERP_ROOT" \
    --check-preprocessing \
    --check-models \
    --check-performance \
    --tolerance 0.005
```

This verification script compares:
- Sample counts at each preprocessing step
- Model architecture checksums  
- Performance metrics within tolerance
- File existence and structure

---

## Conclusion

This technical appendix provides complete instructions for reproducing the multimodal ICU LOS prediction framework. Following these procedures exactly should yield performance metrics within the specified confidence intervals. The modular design allows for replication of individual components or the complete pipeline.

For questions or issues during reproduction, refer to the codebase documentation at https://github.com/CHNOPS0127/ERP_MIMIC-III or verify against the expected outputs listed in Section 7.

**Estimated Total Runtime:** 12-48 hours depending on hardware configuration and selected models.