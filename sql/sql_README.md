# 🗄️ SQL Queries — MIMIC-IV Data Extraction

> Production-grade BigQuery SQL for extracting the ICU readmission cohort and 225+ clinical features from MIMIC-IV v2.2.

---

## 📁 File

| File | Lines | Description |
|------|-------|-------------|
| `01_mimic_iv_cohort_and_features.sql` | 6,303 | Full cohort definition + feature engineering + quality checks |

---

## 🎯 What This SQL Does

### 1️⃣ Cohort Definition
- Selects the **first ICU stay per patient** (index stay)
- **Inclusion criteria:** Age ≥ 18, ICU length of stay ≥ 24h, survived the admission
- Defines the **30-day readmission label** (readmitted within 1–30 days of ICU discharge)
- Final cohort: **48,676 patients**

### 2️⃣ Feature Engineering — 7 Clinical Domains

| Domain | Features |
|--------|----------|
| 🏋️ Anthropometry | Height, weight, BMI |
| ❤️ Vital Signs | HR, BP (sys/dia/mean), temperature, SpO₂, respiratory rate — mean/min/max over first 24h |
| 🧪 Laboratory Values | Creatinine, lactate, glucose, CBC (WBC, Hgb, platelets), metabolic panel, coagulation (PT, INR) |
| 🧠 Neurological | GCS motor, verbal, eye subscores |
| 💊 Medications | Vasopressors, sedatives, antibiotics |
| 🏥 Interventions | Mechanical ventilation, dialysis, blood products |
| 📋 Comorbidities & Severity | Charlson Comorbidity Index, Elixhauser, SOFA, APACHE II, KDIGO staging |

### 3️⃣ Prior History
- Previous hospital admissions (within 12 months)
- ICU utilization patterns
- Days since last discharge

### 4️⃣ Quality Checks (QC1–QC6)

| Check | What It Validates |
|-------|------------------|
| QC1 | Row count & primary key uniqueness |
| QC2 | Target distribution (~10% readmission rate) |
| QC3 | Feature coverage by domain (% non-null) |
| QC4 | Clinical range sanity checks (e.g., HR 20–300) |
| QC5 | Missingness vs. outcome bias analysis |
| QC6 | Final schema validation |

---

## 📤 Output Tables Created

```
readmission30/
├── mimiciv_index_cohort_30d             ← Base cohort + readmission label
├── feature_anthropometry
├── feature_vitals_first24h
├── feature_labs_first24h
├── feature_gcs_neurological
├── feature_medications_interventions
├── feature_comorbidities_severity
├── feature_prior_history_hemodynamics
└── model_dataset_readmission_30d        ← ✅ FINAL TABLE (48,676 × 225+ columns)
```

---

## ⚙️ Requirements

### Data Access
1. [PhysioNet](https://physionet.org/) account with **MIMIC-IV v2.2** approval
2. CITI *"Data or Specimens Only Research"* training certificate
3. Signed PhysioNet Data Use Agreement (DUA)

### Technical
- Google Cloud Platform account with BigQuery enabled
- `physionet-data.mimiciv_3_1_*` tables accessible in your project
- ~10 TB BigQuery quota available

---

## 🚀 How to Run

### Option 1 — BigQuery Web Console
```
1. Go to: https://console.cloud.google.com/bigquery
2. Create dataset: your-project.readmission30
3. Open the query editor
4. Paste contents of 01_mimic_iv_cohort_and_features.sql
5. Click Run
6. Wait ~45 minutes
7. Export final table → Google Cloud Storage → download as Parquet
```

### Option 2 — BigQuery CLI
```bash
bq mk --dataset your-project:readmission30
bq query --use_legacy_sql=false < 01_mimic_iv_cohort_and_features.sql
```

---

## 📊 Expected Quality Check Results

After a successful run you should see:

```
✅ 48,676 unique patients
✅ 10.07% readmission rate (4,902 positive cases)
✅ Zero duplicate keys (subject_id, hadm_id, index_stay_id)
✅ 95%+ coverage for vital signs
✅ 60–90% coverage for laboratory values
✅ All clinical ranges within expected physiological bounds
```

---

## 💡 Performance Notes

| Metric | Value |
|--------|-------|
| Estimated query cost | ~$10 USD (BigQuery on-demand) |
| Processing time | 40–50 minutes (full MIMIC-IV) |
| Memory | Handled by BigQuery distributed engine — no local RAM needed |

---

## 📜 Citation

**MIMIC-IV Dataset:**
```bibtex
@article{johnson2023mimic,
  title     = {MIMIC-IV, a freely accessible electronic health record dataset},
  author    = {Johnson, Alistair EW and Bulgarelli, Lucas and others},
  journal   = {Scientific Data},
  volume    = {10},
  number    = {1},
  pages     = {1},
  year      = {2023}
}
```

**This Project:**
```bibtex
@misc{das2025icu,
  author = {Jyoti Prakash Das},
  title  = {ICU 30-Day Readmission Risk Prediction using MIMIC-IV},
  year   = {2025},
  url    = {https://github.com/yourusername/icu-readmission-predictor}
}
```

---

## 📄 License

| Component | License |
|-----------|---------|
| SQL code (this file) | MIT License |
| MIMIC-IV data | [PhysioNet Credentialed Health Data License](https://physionet.org/content/mimiciv/view-license/2.2/) |

> ⚠️ **The data itself cannot be shared.** Only the code for querying it is published here, in compliance with the PhysioNet DUA.

---

*Questions? Open a GitHub Issue or refer to the [main project README](../../README.md).*
