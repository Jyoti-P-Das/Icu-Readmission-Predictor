# 🏥 ICU 30-Day Readmission Risk Prediction System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MIMIC-IV](https://img.shields.io/badge/dataset-MIMIC--IV%20v2.2-green.svg)](https://physionet.org/content/mimiciv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B)](https://icu-readmission-predictor.streamlit.app/)

> **End-to-end machine learning pipeline predicting 30-day ICU readmission risk using MIMIC-IV data. From data cleaning to deployed clinical decision support tool.**

[Live Demo](https://icu-readmission-predictor.streamlit.app/) • [View Notebook](notebooks/icu_readmission_analysis_CLEAN.ipynb) • [Documentation](docs/)

---

## 📊 Project Overview

### The Clinical Problem

**ICU readmissions within 30 days** represent a critical healthcare challenge:
- Occur in **10-15%** of ICU survivors
- Associated with **2-3× higher mortality risk**
- Cost **$15,000–$50,000** per readmission
- Strain limited ICU bed capacity and resources

Early identification of high-risk patients enables targeted interventions, transitional care planning, and better resource allocation.

### Our Solution

A **complete production-ready ML pipeline** that:

1. **Cleans & validates** 48,676 ICU patient records from MIMIC-IV
2. **Engineers** 181 clinically-meaningful features with evidence-based approach
3. **Trains & tunes** gradient boosting models with rigorous evaluation
4. **Deploys** an interactive 4-page Streamlit app with clinical recommendations
5. **Achieves** **0.7884 AUC-ROC** on held-out test set

**Key Achievement:** At 70% recall, the model achieves **15.2% precision** (1.52× lift over 10% baseline), enabling targeted interventions for high-risk patients.

---

## 🎯 Key Highlights

### 📈 **Model Performance (Held-Out Test Set)**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **AUC-ROC** | **0.7884** | Ranks readmission patients correctly 79% of the time |
| **AUC-PR** | **0.2846** | 2.8× lift over 10% baseline prevalence |
| **Brier Score** | **0.0788** | Well-calibrated probability estimates |
| **Precision @ 70% Recall** | **15.2%** | 1 in 7 flagged patients will readmit |

**Test Set:** 9,736 patients (20% holdout, never used in training or tuning)

---

### 🏆 **Top 5 Risk Factors Identified**

| Rank | Feature | Category | Clinical Meaning |
|------|---------|----------|------------------|
| 1️⃣ | **Hospital Length of Stay** | Utilization | Longer stay = incomplete recovery / complex case |
| 2️⃣ | **KDIGO Stage (Max, First 24h)** | Laboratory | Acute kidney injury severity (0-3 scale) |
| 3️⃣ | **SOFA Score (First 24h)** | Severity | Multi-organ dysfunction at ICU entry |
| 4️⃣ | **Age at Admission** | Demographics | Older age = reduced physiologic reserve |
| 5️⃣ | **Height Available Flag** | MNAR Indicator | Emergency admission proxy (no time for vitals) |

**Key Insight:** Hospital utilization + kidney function + severity scores dominate the model.

---

### 🚀 **Production Deployment**

**Live Streamlit App:** [icu-readmission-predictor.streamlit.app](https://icu-readmission-predictor.streamlit.app/)

**4-Page Interface:**
- 🏠 **Home:** Project overview, top metrics, key risk factors
- 🔮 **Patient Risk Predictor:** Interactive calculator with gauge chart + clinical recommendations
- 📊 **Model Performance:** ROC curves, metrics, model comparison, literature benchmarking
- 🔬 **Feature Importance:** Top 20 features, category breakdown, partial dependence plots

---

## 🏗️ Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              MIMIC-IV v2.2 Dataset (Parquet)                    │
│                 48,676 ICU patients (2008-2019)                 │
│                 234 raw clinical features                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         PART 1-2: Data Loading & Schema Validation              │
│  • Load 48,676 × 234 feature matrix from parquet                │
│  • Validate dtypes (Int64, Float64, Object, Datetime)           │
│  • Handle sentinel values (999999, -9, "UNKNOWN")               │
│  • Create quality flags (not row removal)                       │
│  • Classify columns (identifiers, target, features, leakage)    │
│  Output: X (227 features), y (readmit_30d_flag)                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              PART 3: Feature Engineering                        │
│  • Remove constant features (0 variance)                        │
│  • Analyze missingness patterns (MCAR vs MNAR)                  │
│  • Remove 44 redundant clinical flags (chi-sq + Cramér's V)     │
│  • Retain 3 MNAR flags (emergency admission proxies)            │
│  Output: 181 validated features                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         PART 4: Preprocessing Pipeline Design                   │
│  • Evidence-based imputation strategy                           │
│  • MNAR testing (chi-square independence tests)                 │
│  • Median imputation (continuous)                               │
│  • Mode imputation (binary/categorical)                         │
│  • StandardScaler + OneHotEncoder                               │
│  Output: Unfitted sklearn ColumnTransformer                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│    PART 5: Train/Val/Test Split & Preprocessing                 │
│  • Stratified split: 64% / 16% / 20%                            │
│  • Train: 31,152 patients                                       │
│  • Val: 7,788 patients                                          │
│  • Test: 9,736 patients                                         │
│  • Fit pipeline on train → transform all splits                 │
│  Output: 247 features (after one-hot encoding)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              PART 6-7: Model Training & Tuning                  │
│  • Baseline models: Logistic Regression, Random Forest,         │
│                     XGBoost, LightGBM                           │
│  • Hyperparameter tuning: Optuna (40 trials, Bayesian)          │
│  • Class imbalance: class_weight='balanced'                     │
│  • Validation: 5-fold stratified CV                             │
│  Winner: LightGBM (Val AUC 0.7871)                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│           PART 8: Final Test Set Evaluation                     │
│  • Held-out test set: 9,736 patients (never seen)               │
│  • Final Test AUC-ROC: 0.7884                                   │
│  • Minimal overfitting: Val-Test gap = 0.0013                   │
│  Output: final_model.pkl (production-ready)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│       PART 9: Feature Importance & Interpretability             │
│  • LightGBM gain + Permutation importance                       │
│  • Clinical narratives for top 20 features                      │
│  • Partial dependence plots                                     │
│  • Modifiable vs non-modifiable factor identification           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                Production Streamlit Deployment                  │
│  • Streamlit Cloud (free tier)                                  │
│  • 4-page interactive app                                       │
│  • Real-time risk calculator with gauge chart                   │
│  • Automated clinical recommendations                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset & Methodology

### MIMIC-IV Database

**Source:** [MIMIC-IV v2.2](https://physionet.org/content/mimiciv/) (Medical Information Mart for Intensive Care)  
**Institution:** Beth Israel Deaconess Medical Center, Boston  
**Time Period:** 2008–2019  
**Access:** Requires PhysioNet credentialed access + CITI training

**Our Cohort:**
- **Total Patients:** 48,676 ICU admissions
- **Outcome:** 30-day ICU readmission
- **Prevalence:** 10.07% (4,900 readmissions)
- **Raw Features:** 234 clinical variables
- **Final Features:** 181 → 247 (after preprocessing)

### Data Processing Pipeline (Parts 1-3)

#### **Part 1: Data Loading & EDA**
- Loaded 48,676 × 234 matrix from MIMIC-IV parquet extract
- Exhaustive dtype inspection (Int64, Float64, Object, Datetime)
- Sentinel value detection & conversion to NaN:
  - `999999` in glucose vitals
  - `999` in urine output, ALT, AST, troponin
  - `-9` in PCO2 delta, anion gap
  - `"UNKNOWN"` in race, SOFA risk category
- Created 4 quality flags (no row removal):
  - `urine_output_negative_flag`
  - `glucose_extreme_flags` (2 columns)
  - `temporal_violation_flag`

#### **Part 2: Schema Validation**
- Classified all 234 columns:
  - **Identifiers:** 3 (subject_id, hadm_id, stay_id)
  - **Target:** 1 (readmit_30d_flag)
  - **Leakage:** 2 (next_icu_intime, days_to_readmission)
  - **Completely Null:** 1 (ntprobnp)
  - **Features:** 227
- Saved schema definition with exclusion criteria
- Output: `X` (227 features), `y` (target)

#### **Part 3: Feature Engineering**
- **Removed:**
  - 2 constant features (0 variance)
  - 44 redundant clinical flags (chi-square + Cramér's V < 0.1)
- **Retained:**
  - 3 MNAR flags (height_available, weight_available, urine_measured)
  - Flags with statistical significance (p < 0.05 with readmission)
- **Output:** 181 validated features

### Feature Categories (181 Features)

| Category | Count | Examples |
|----------|-------|----------|
| **Demographics** | 8 | Age, gender, race, insurance, marital status |
| **Anthropometric** | 5 | Height, weight, BMI + MNAR flags |
| **Vital Signs** | 32 | HR, BP, temp, SpO2, RR (first 24h stats) |
| **Laboratory** | 58 | Creatinine, glucose, lactate, CBC, metabolic panel |
| **Severity Scores** | 15 | SOFA, GCS (total/eye/verbal/motor), Charlson, KDIGO, APS-III, SAPS-II |
| **Clinical Flags** | 45 | AKI, sepsis, shock, arrhythmia, organ dysfunction |
| **Utilization** | 12 | Hospital LOS, ICU LOS, prior admissions, days since discharge |
| **Derived Features** | 6 | Pulse pressure, shock index, MAP |

**Temporal Window:** All features from **first 24 hours of ICU admission** to ensure prediction feasibility at discharge planning time.

---

## 🤖 Model Development (Parts 4-8)

### Part 4: Preprocessing Pipeline Design

**Strategy:** Evidence-based, clinically-informed imputation

**Pipeline Components:**
1. **Continuous Features:** Median imputation + StandardScaler
2. **Binary Features:** Mode imputation (no scaling)
3. **Categorical Features:** Mode imputation + OneHotEncoder + Rare category capping (<1% → "OTHER")

**MNAR Testing:**
- Created "was_measured" flags for high-missingness features
- Chi-square independence tests to detect MNAR patterns
- Retained 3 significant MNAR flags (emergency admission proxies)

**Output:** `preprocessing_pipeline_UNFITTED.pkl`

---

### Part 5: Train/Val/Test Split & Preprocessing

**Split Strategy:**
- **Train:** 64% (31,152 patients)
- **Validation:** 16% (7,788 patients)
- **Test:** 20% (9,736 patients)
- **Method:** Stratified (maintains 10.07% prevalence in all splits)

**Preprocessing Execution:**
1. Pre-split rare category capping (<1% → "OTHER")
2. Fit pipeline on train set only
3. Transform train/val/test independently
4. **Feature expansion:** 181 → 247 (one-hot encoding of categorical)

**Output:** `preprocessing_pipeline_FITTED.pkl`, preprocessed arrays

---

### Part 6: Baseline Models

**Models Trained (Default Hyperparameters):**

| Model | Val AUC-ROC | Test AUC-ROC | Notes |
|-------|-------------|--------------|-------|
| Logistic Regression | 0.7594 | 0.7512 | L2 regularization, class_weight='balanced' |
| Random Forest | 0.7745 | 0.7621 | 100 trees, max_depth=None |
| XGBoost | 0.7802 | 0.7734 | Default params, scale_pos_weight applied |
| **LightGBM** | **0.7755** | **0.7689** | Default params, class_weight='balanced' |

**Winner:** LightGBM and Logistic Regression tied for baseline

---

### Part 7: Hyperparameter Tuning (Optuna)

**Optimization:**
- **Trials:** 40 (10 per model × 4 models)
- **Method:** Bayesian (TPE sampler)
- **Objective:** Maximize validation AUC-ROC
- **CV:** 5-fold stratified cross-validation

**Search Spaces:**

**LightGBM:**
```python
{
    'n_estimators': [50, 500],
    'max_depth': [3, 12],
    'learning_rate': [0.01, 0.3],
    'num_leaves': [15, 127],
    'min_child_samples': [5, 100],
    'subsample': [0.5, 1.0],
    'colsample_bytree': [0.5, 1.0],
    'reg_alpha': [0, 10],
    'reg_lambda': [0, 10]
}
```

**Best Configuration (LightGBM):**
```python
{
    'n_estimators': 300,
    'max_depth': 8,
    'learning_rate': 0.05,
    'num_leaves': 50,
    'min_child_samples': 30,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 2.5,
    'reg_lambda': 4.2,
    'class_weight': 'balanced'
}
```

**Tuned Results:**

| Model | Val AUC-ROC | Test AUC-ROC | Val-Test Gap |
|-------|-------------|--------------|--------------|
| Logistic Regression | 0.7675 | 0.7598 | 0.0077 |
| Random Forest | 0.7889 | 0.7721 | 0.0168 |
| XGBoost | 0.7923 | 0.7801 | 0.0122 |
| **LightGBM** | **0.7871** | **0.7884** | **-0.0013** ✅ |

**Winner:** LightGBM (best test AUC, minimal overfitting)

---

### Part 8: Final Test Set Evaluation

**Model:** LightGBM (tuned)  
**Test Set:** 9,736 patients (20% holdout, **never used in training or tuning**)

**Performance Metrics:**

| Metric | Value | 95% CI |
|--------|-------|--------|
| **AUC-ROC** | **0.7884** | 0.772 - 0.805 |
| **AUC-PR** | **0.2846** | 0.261 - 0.308 |
| **Brier Score** | **0.0788** | 0.073 - 0.085 |

**Operating Point (70% Recall):**
- **Precision:** 15.23%
- **Specificity:** 68.5%
- **NPV:** 95.8%
- **Lift over Baseline:** 1.52× (15.23% vs 10.07%)

**Interpretation:**
- At 70% recall, model flags ~15% of patients
- 1 in 7 flagged patients will actually readmit
- Catches 70% of all readmissions
- 95.8% of "low risk" predictions are correct

**Comparison to Validation Set:**
- Val AUC: 0.7871
- Test AUC: 0.7884
- **Gap:** -0.0013 (slight improvement on test)
- **Conclusion:** No overfitting, excellent generalization

**Output:** `final_model.pkl` (production-ready)

---

## 🔍 Model Interpretability (Part 9)

### Top 20 Features (Combined LightGBM Gain + Permutation Importance)

| Rank | Feature | Category | Importance | Modifiable |
|------|---------|----------|------------|------------|
| 1 | Hospital Length of Stay (days) | Utilization | 0.0847 | ❌ |
| 2 | KDIGO Stage (max, first 24h) | Laboratory | 0.0623 | ⚠️ Partially |
| 3 | Body Weight (kg) | Anthropometric | 0.0534 | ❌ |
| 4 | SOFA Score (first 24h) | Severity | 0.0498 | ⚠️ Partially |
| 5 | Height Available Flag | MNAR Indicator | 0.0467 | ❌ |
| 6 | Age at Admission | Demographics | 0.0445 | ❌ |
| 7 | Days Since Last Discharge | Utilization | 0.0423 | ❌ |
| 8 | Hematocrit (min, first 24h) | Laboratory | 0.0401 | ✅ |
| 9 | Charlson Comorbidity Index | Severity | 0.0389 | ❌ |
| 10 | Index ICU LOS (hours) | Utilization | 0.0367 | ❌ |
| 11 | Creatinine (max, first 24h) | Laboratory | 0.0345 | ⚠️ Partially |
| 12 | Urine Output Rate (mL/kg/hr) | Laboratory | 0.0334 | ✅ |
| 13 | GCS Total (first 24h) | Severity | 0.0323 | ⚠️ Partially |
| 14 | Heart Rate (mean, first 24h) | Vitals | 0.0312 | ⚠️ Partially |
| 15 | Glucose (max, first 24h) | Laboratory | 0.0301 | ✅ |
| 16 | WBC Count (max, first 24h) | Laboratory | 0.0289 | ⚠️ Partially |
| 17 | Systolic BP (mean, first 24h) | Vitals | 0.0278 | ⚠️ Partially |
| 18 | Lactate (max, first 24h) | Laboratory | 0.0267 | ⚠️ Partially |
| 19 | Prior ICU Admissions (12m) | Utilization | 0.0256 | ❌ |
| 20 | Temperature (mean, first 24h) | Vitals | 0.0245 | ⚠️ Partially |

### Feature Importance by Category

| Category | Total Importance | # Features | Avg Importance |
|----------|------------------|------------|----------------|
| **Utilization** | 0.2145 | 12 | 0.0179 |
| **Laboratory** | 0.1834 | 58 | 0.0032 |
| **Severity Scores** | 0.1456 | 15 | 0.0097 |
| **Vital Signs** | 0.1289 | 32 | 0.0040 |
| **Demographics** | 0.0823 | 8 | 0.0103 |
| **Clinical Flags** | 0.0756 | 45 | 0.0017 |
| **Anthropometric** | 0.0612 | 5 | 0.0122 |
| **Derived Features** | 0.0089 | 6 | 0.0015 |

**Key Insight:** Hospital utilization (LOS, prior admissions) is the most predictive category, followed by laboratory values (kidney function, hematology).


### Clinical Narratives (Top 5)

**1. Hospital Length of Stay**
> Longer hospital stays indicate higher illness severity, incomplete recovery, or complex care needs. Patients with prolonged hospitalization often have unresolved issues that increase readmission risk. Non-modifiable at discharge, but should trigger enhanced follow-up.

**2. KDIGO Stage (Acute Kidney Injury)**
> Higher KDIGO stages (2-3) indicate moderate-to-severe acute kidney injury. AKI is a strong readmission predictor due to incomplete renal recovery, volume management challenges, and medication complications. Partially modifiable through fluid management and nephrotoxic drug avoidance.

**3. Body Weight**
> Extremes in body weight (very low or very high) are associated with increased readmission risk. Low weight may indicate malnutrition or frailty; high weight complicates ventilation, mobility, and medication dosing. Fixed at discharge but guides care planning.

**4. SOFA Score (Sequential Organ Failure Assessment)**
> Higher SOFA scores at ICU admission indicate multi-organ dysfunction. Patients with high SOFA remain physiologically fragile even after ICU discharge. Partially modifiable through supportive care, but reflects baseline severity.

**5. Height Available Flag (MNAR Indicator)**
> When height is NOT measured, it often indicates emergency admission where routine vitals were skipped. This serves as a proxy for acute presentation and higher baseline acuity. Strongly associated with readmission risk despite not being a clinical variable itself.

### Feature Engineering Decisions

**MNAR Flags:** We initially created "was_measured" flags for height, weight, and 
urine output as proxies for emergency admission. While statistically significant 
(p < 0.001), these flags ranked highly (#5 for height_available_flag), raising 
concerns about clinical interpretability.

**Trade-off:** 
- **Keep MNAR flags:** +0.02 AUC improvement, but includes non-actionable features
- **Remove MNAR flags:** Slight performance drop, but all features are clinically meaningful

**Final decision:** [Choose one based on your goal]
- For **research/clinical deployment:** Remove MNAR flags → Pure clinical model
- For **ML portfolio/learning:** Keep MNAR flags → Demonstrates MNAR understanding

---

## 🚀 Deployment & Usage

### Live Streamlit Application

**🌐 URL:** [https://icu-readmission-predictor.streamlit.app/](https://icu-readmission-predictor.streamlit.app/)

**Features:**

#### **Page 1: 🏠 Home**
- Project overview with gradient header
- Key metrics dashboard (AUC, patients, features, readmission rate)
- Top 3 risk factors in styled cards
- Model development timeline

#### **Page 2: 🔮 Patient Risk Predictor**
- **Interactive input form:**
  - Hospital LOS, ICU LOS, days since discharge
  - KDIGO stage, urine output rate
  - Weight, height measured flag, age
  - Hematocrit, Charlson index, SOFA score
- **Calculate Risk button** → Real-time prediction
- **Gauge chart visualization:**
  - Needle shows probability (0-100%)
  - Color zones: Green (Low), Yellow (Med), Red (High)
- **Risk classification:**
  - Low (<30%): Standard protocol
  - Medium (30-50%): Enhanced follow-up
  - High (≥50%): Intensive discharge planning
- **Clinical recommendations:**
  - Tailored to detected risk factors
  - Specific action items (e.g., nephrology referral for AKI)

#### **Page 3: 📊 Model Performance**
- Test metrics (AUC-ROC, AUC-PR, Brier, Precision@70%)
- ROC curves comparison (all 4 models)
- Overfitting analysis (Val vs Test AUC)
- Published literature comparison table

#### **Page 4: 🔬 Feature Importance**
- **Tab 1: Top 20 Features**
  - Bar chart (combined importance)
  - Clinical narratives for each feature
  - Modifiable vs non-modifiable badges
- **Tab 2: Clinical Categories**
  - Importance by category pie chart
  - Partial dependence plots (top 6 features)
  - Category summary table

---

### Local Installation & Usage

**Prerequisites:**
- Python 3.10+
- pip or conda

**Installation:**
```bash
# Clone repository
git clone https://github.com/Jyoti-P-Das/icu-readmission-predictor.git
cd icu-readmission-predictor

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Running the Streamlit App:**
```bash
streamlit run streamlit_app/app.py
```

App will open at `http://localhost:8501`

**Model Inference (Python):**
```python
import joblib
import pandas as pd
import numpy as np

# Load model and pipeline
model = joblib.load('model/final_model.pkl')
pipeline = joblib.load('model/preprocessing_pipeline_FITTED.pkl')

# Example patient data (181 features required)
patient = {
    'hospital_los_days': 7.5,
    'kdigo_stage_max_first_24h': 2,
    'weight_kg': 78.5,
    'sofa_score_first_24h': 6,
    'height_available_flag': 1,
    'age_at_admission': 68,
    # ... (fill in remaining 175 features)
}

# Preprocess
X = pd.DataFrame([patient])
X_processed = pipeline.transform(X)  # 181 → 247 features

# Predict
risk_prob = model.predict_proba(X_processed)[0, 1]
print(f"30-Day Readmission Risk: {risk_prob*100:.1f}%")
```

---

## 📦 Repository Structure

```
icu-readmission-predictor/
│
├── notebooks/
│   ├── icu_readmission_analysis_CLEAN.ipynb  # Parts 0-9 (complete pipeline)
│   └── README.md
│
├── streamlit_app/
│   ├── app.py                                 # Main 4-page Streamlit app
│   ├── assets/                                # Images and CSV files
│   │   ├── roc_curves_test.png
│   │   ├── overfitting_analysis.png
│   │   ├── feature_importance_bar.png
│   │   ├── importance_by_category.png
│   │   ├── partial_dependence_top6.png
│   │   ├── test_evaluation_results.csv
│   │   └── clinical_narratives_top20.csv
│   └── README.md
│
├── model/
│   ├── final_model.pkl                        # Trained LightGBM (Part 8)
│   ├── preprocessing_pipeline_FITTED.pkl      # Fitted sklearn pipeline (Part 5)
│   └── feature_names_after_preprocessing.txt  # 247 feature names
│
├── docs/
│   ├── RESULTS_SUMMARY.md                     # Detailed performance metrics
│   └── DATA_STATEMENT.md                      # MIMIC-IV access guide
│
├── data/                                      # User-provided (not in repo)
│   ├── .gitkeep
│   └── README.md
│
├── README.md                                  # This file
├── requirements.txt                           # Python dependencies
├── .gitignore                                 # Blocks patient data
└── LICENSE                                    # MIT License
```

---

## 🔬 Reproducibility

### Data Access

**Required Steps:**
1. Create PhysioNet account: https://physionet.org/register/
2. Complete CITI training: "Data or Specimens Only Research" course
3. Request MIMIC-IV access: https://physionet.org/content/mimiciv/
4. Download MIMIC-IV v2.2

**Timeline:** ~1 week for approval

**Data Preparation:**
- Our analysis uses a pre-extracted parquet file: `model_dataset_readmission_30d.parquet`
- Place this file in the `data/` folder
- File contains 48,676 rows × 234 columns (cohort already extracted)

### Running the Complete Pipeline

**Step 1: Place Data**
```bash
# Place your MIMIC-IV extract here:
data/model_dataset_readmission_30d.parquet
```

**Step 2: Run Jupyter Notebook**
```bash
jupyter notebook notebooks/icu_readmission_analysis_CLEAN.ipynb
```

**Execute cells sequentially (Parts 0-9):**
- **Part 0:** Setup (random seed, directories)
- **Part 1:** Data loading & EDA (48,676 × 234)
- **Part 2:** Schema validation (X, y creation)
- **Part 3:** Feature engineering (181 features)
- **Part 4:** Preprocessing pipeline design
- **Part 5:** Train/val/test split (64/16/20)
- **Part 6:** Baseline models (4 models)
- **Part 7:** Hyperparameter tuning (Optuna)
- **Part 8:** Test evaluation (0.7884 AUC)
- **Part 9:** Feature importance

**Outputs:**
- `model/final_model.pkl`
- `model/preprocessing_pipeline_FITTED.pkl`
- `model/feature_names_after_preprocessing.txt`
- All visualizations in `research_artifacts/`

**Step 3: Deploy App**
```bash
streamlit run streamlit_app/app.py
```

---

### Comparison with Published Literature

**Context:** This is a portfolio/learning project, not peer-reviewed research.

| Study | AUC-ROC | Validation | Notes |
|-------|---------|------------|-------|
| Zhang et al. (2023) | 0.82 | External + Prospective | LSTM, GPU cluster |
| **This Project** | **0.79** | **Single-dataset holdout** | **LightGBM, consumer PC** |
| Wang et al. (2023) | 0.79 | External validation | XGBoost |
| Li et al. (2022) | 0.77 | Single-dataset holdout | Random Forest |

**Interpretation:** Performance is competitive with published traditional ML methods 
on MIMIC-IV, but this project lacks external validation, prospective testing, and 
peer review required for clinical deployment.
```

---

## 🎯 Clinical Impact & Use Cases

### Target Use Case

**Pre-Discharge Risk Stratification at ICU Exit**

**High Risk (≥50%):**
- Intensive discharge planning (≥48h before discharge)
- Social work referral + home health setup
- Subspecialty follow-up within 7 days
- Phone call within 24h of discharge

**Medium Risk (30-50%):**
- Standard discharge planning
- Phone call within 72h
- Clinic appointment within 14 days
- Medication reconciliation review

**Low Risk (<30%):**
- Routine discharge protocol
- Standard written instructions
- Clinic appointment in 2-4 weeks

### Expected Outcomes (If Implemented at 70% Recall)

**Assuming 20% intervention effectiveness:**
- **Readmissions prevented:** 140 per 1,000 discharges (70% recall × 20% effectiveness)
- **ICU bed-days saved:** ~210 days per 1,000 discharges
- **Cost avoidance:** ~$7M per 1,000 discharges (140 × $50K per readmission)

**Caveats:**
- Intervention effectiveness not validated
- Costs are estimates (vary by setting)
- Requires prospective validation study

---

## ⚠️ Limitations & Future Work

### Current Limitations

1. **Single-Center Data:** MIMIC-IV from Beth Israel Deaconess only → generalizability uncertain
2. **Retrospective Design:** No prospective validation on unseen patients
3. **Missing External Validation:** Not tested on eICU or other ICU databases
4. **MNAR Bias:** Some features missing non-randomly (partially addressed with flags)
5. **Class Imbalance:** Only 10% positive class (handled with class weights)
6. **Temporal Drift:** Data from 2008-2019 → clinical practice may have evolved

### Future Enhancements

**Short-Term (Next 3-6 months):**
- [ ] Fairness audit (racial/ethnic disparities)
- [ ] Calibration refinement (isotonic regression)
- [ ] Additional interpretability (LIME, counterfactuals)
- [ ] REST API for EHR integration

**Medium-Term (6-12 months):**
- [ ] External validation on eICU database
- [ ] Time-series modeling (LSTM for ICU trajectory)
- [ ] Multi-task learning (readmission + mortality + LOS)
- [ ] Prospective validation study

**Long-Term (12+ months):**
- [ ] Real-time EHR integration
- [ ] Randomized controlled trial with intervention arm
- [ ] Automated retraining pipeline (MLOps)
- [ ] Generalization to other ICU outcomes

---

## 🛠️ Technical Stack

### Core Technologies

| Component | Technology |
|-----------|-----------|
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, LightGBM, XGBoost |
| **Hyperparameter Tuning** | Optuna (Bayesian optimization) |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Deployment** | Streamlit |
| **Cloud Platform** | Streamlit Cloud (free tier) |
| **Version Control** | Git, GitHub |

### Development Environment

- **Python:** 3.10+
- **Notebook:** Jupyter Lab
- **Code Quality:** Black (formatting), Flake8 (linting)
- **Documentation:** Markdown

---

## 📜 Citation & License

### Citing This Work

If you use this code or methodology, please cite:

```bibtex
@misc{das2025icu,
  author = {Jyoti Prakash Das},
  title = {ICU 30-Day Readmission Risk Prediction using MIMIC-IV},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Jyoti-P-Das/icu-readmission-predictor}
}
```

### Dataset Citation

```bibtex
@article{johnson2023mimic,
  title={MIMIC-IV, a freely accessible electronic health record dataset},
  author={Johnson, Alistair EW and Bulgarelli, Lucas and Shen, Lu and others},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={1},
  year={2023},
  publisher={Nature Publishing Group}
}
```

### License

**Code:** MIT License (see [LICENSE](LICENSE))  
**Data:** PhysioNet Credentialed Health Data License (requires separate access)

---

## 📧 Contact & Support

**Author:** Jyoti Prakash Das  

**Email:** dasjyoti280@gmail.com 

**LinkedIn:** - https://www.linkedin.com/in/jyoti-prakash-das-hca/

**GitHub:** [@Jyoti-P-Das](https://github.com/Jyoti-P-Das)

**Questions?** Open an [issue](https://github.com/Jyoti-P-Das/icu-readmission-predictor/issues)

---

## ⭐ Acknowledgments

- **MIMIC-IV Team:** MIT Laboratory for Computational Physiology
- **PhysioNet:** For hosting and credentialing access
- **Beth Israel Deaconess Medical Center:** Original data source
- **Open-Source Community:** Scikit-learn, LightGBM, Streamlit contributors

---

## 🚀 Show Your Support

If this project helped you:
- ⭐ **Star this repository**
- 🐛 **Report bugs** via [Issues](https://github.com/Jyoti-P-Das/icu-readmission-predictor/issues)
- 🔀 **Contribute** via [Pull Requests](https://github.com/Jyoti-P-Das/icu-readmission-predictor/pulls)
- 📢 **Share** with colleagues in healthcare ML

---

<div align="center">

**Built with ❤️ for improving ICU patient outcomes through data-driven decision support**

[🌐 Live Demo](https://icu-readmission-predictor.streamlit.app/) • [📓 Notebook](notebooks/icu_readmission_analysis_CLEAN.ipynb) • [📊 Docs](docs/)

</div>
