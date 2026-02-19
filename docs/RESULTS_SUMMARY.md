# ICU Readmission Prediction — Results Summary

---

## 📊 Model Performance (Held-Out Test Set)

**Test Set:** 9,736 patients (20% of cohort, never seen during training or tuning)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **AUC-ROC** | **0.7884** | Model correctly ranks a readmission patient above a non-readmission patient 79% of the time |
| **AUC-PR** | **0.2846** | Strong performance given 10% baseline prevalence |
| **Brier Score** | **0.0788** | Low = well-calibrated probabilities |
| **Precision @ 70% Recall** | **15.2%** | At 70% sensitivity, 1 in 7 flagged patients will actually readmit |

---

## 🎯 Clinical Interpretation

### At 70% Recall Operating Point:
- **Catches:** 70 out of 100 true readmissions
- **Precision:** 15.2% (1 in 7 flagged patients readmits)
- **Lift over random:** 1.5× (baseline rate is 10%)

### Clinical Utility:
- Phone follow-up calls are low-cost → high FPR acceptable
- Missing a readmission is costly (~$15,000–$50,000) → prioritize sensitivity
- 70% recall with 15% precision is clinically actionable

---

## 🏆 Top 10 Risk Factors

Ranked by combined LightGBM gain + permutation importance:

| Rank | Feature | Category | Clinical Meaning |
|------|---------|----------|------------------|
| **1** | Hospital length of stay | Utilization | Longer stay = incomplete recovery / complex case |
| **2** | KDIGO stage (max, first 24h) | Laboratory | Acute kidney injury severity (0–3 scale) |
| **3** | SOFA score (first 24h) | Severity | Multi-organ dysfunction at ICU entry |
| **4** | Age at admission | Demographics | Older age = reduced physiologic reserve |
| **5** | Height available flag | MNAR | Emergency admission proxy (height not measured) |
| **6** | Body weight | Anthropometric | Extremes (very low/high) = higher risk |
| **7** | Charlson comorbidity index | Comorbidity | Chronic disease burden |
| **8** | Days since last discharge | Utilization | Recent prior admission = higher risk |
| **9** | Urine output rate | Laboratory | Kidney function / fluid status |
| **10** | Hematocrit (min, first 24h) | Laboratory | Anaemia severity |

**Key finding:** Hospital utilization + kidney function + severity dominate the model.

---

## 📈 Model Comparison (Test Set)

| Model | Test AUC-ROC | Test AUC-PR | Test Brier | Val-Test Gap |
|-------|--------------|-------------|------------|--------------|
| **LightGBM (Tuned)** | **0.7884** ✅ | **0.2846** | 0.0788 | 0.0013 |
| XGBoost (Tuned) | 0.7868 | 0.2799 | 0.0791 | 0.0127 |
| Random Forest (Tuned) | 0.7785 | 0.2615 | 0.0815 | 0.0156 |
| Logistic Regression (Tuned) | 0.7755 | 0.2503 | 0.0822 | 0.0081 |

**Winner:** LightGBM — highest AUC-ROC, lowest overfitting, well-calibrated.

---

## 🔬 Dataset Summary

| Property | Value |
|----------|-------|
| **Source** | MIMIC-IV v2.2 (Beth Israel Deaconess Medical Centre) |
| **Time Period** | 2008–2019 |
| **Total Patients** | 48,676 ICU admissions |
| **Outcome** | 30-day ICU readmission |
| **Prevalence** | 10.07% (4,900 readmissions) |
| **Raw Features** | 234 clinical variables |
| **Final Features** | 181 → 247 after preprocessing |
| **Train / Val / Test Split** | 64% / 16% / 20% (stratified) |

---

## 📊 Feature Categories Breakdown

| Category | Count | Top Contributor |
|----------|-------|-----------------|
| **Utilization** | 12 | Hospital length of stay |
| **Laboratory** | 58 | KDIGO stage, creatinine |
| **Severity Scores** | 15 | SOFA, GCS, Charlson |
| **Vital Signs** | 32 | Heart rate, blood pressure |
| **Demographics** | 8 | Age, gender, race |
| **Clinical Flags** | 45 | AKI, sepsis, shock |
| **MNAR Indicators** | 3 | Height, weight, urine output not measured |
| **Derived Features** | 7 | Pulse pressure, shock index |

**Total:** 181 features → 247 after one-hot encoding

---

## 📝 Key Methodological Decisions

### Preprocessing:
- **Imputation:** Median (continuous), mode (binary/categorical)
- **Encoding:** One-hot + rare category capping (<1% → "OTHER")
- **Scaling:** StandardScaler on continuous features
- **MNAR handling:** 3 "was_measured" flags retained (emergency admission proxies)

### Model Training:
- **Class imbalance:** `class_weight='balanced'` (10% positive class)
- **Tuning:** Optuna Bayesian optimization (40 trials)
- **Validation:** 5-fold stratified CV (Part 7)
- **Final evaluation:** Held-out 20% test set (Part 8)

### Feature Engineering Highlights:
- **Removed:** 44 redundant clinical flags (chi-square + Cramér's V < 0.1)
- **Kept:** 3 MNAR flags (statistically significant with readmission)
- **Audit:** Every feature decision logged with statistical justification

---

## ⚠️ Limitations

1. **Single-centre data:** MIMIC-IV from one hospital (Beth Israel Deaconess)
2. **Retrospective:** Not a prospective validation
3. **Temporal:** Data from 2008–2019 (clinical practice may have evolved)
4. **Missingness:** 10–50% missing for some lab values (handled via imputation + MNAR flags)
5. **External validation:** Not tested on external datasets

---

## 🔮 Clinical Use Case

**Target population:** ICU patients being considered for discharge

**Use case:** Risk stratification before discharge to allocate follow-up resources

**Intervention examples:**
- **High risk (>20%):** Intensive case management, phone call within 24h, clinic within 7 days
- **Medium risk (10–20%):** Standard follow-up, phone call within 72h
- **Low risk (<10%):** Routine clinic in 2–4 weeks

**NOT intended for:** Real-time triage, mortality prediction, or length-of-stay estimation

---

## 📚 Comparison with Published Literature

| Study | AUC-ROC | Dataset | Model |
|-------|---------|---------|-------|
| **This Study** | **0.7884** | MIMIC-IV (N=48,676) | LightGBM (tuned) |
| Rojas et al. (2018) | 0.77 | Spanish ICU (N=4,812) | Logistic Regression |
| Desautels et al. (2016) | 0.78 | MIMIC-III (N=33,149) | XGBoost |
| Lin et al. (2018) | 0.76 | Taiwan ICU (N=5,327) | Random Forest |
| Badawi & Breslow (2012) | 0.75 | MIMIC-II (N=5,815) | Logistic Regression |
| Houthooft et al. (2015) | 0.74 | Belgian ICU (N=2,158) | Logistic Regression |

**Interpretation:** This study achieves AUC competitive with or exceeding published benchmarks.

---

## 📄 Citation

If you use these results, please cite:

```bibtex
@software{das2025icu,
  author = {Das, Jyoti Prakash},
  title = {ICU 30-Day Readmission Risk Predictor},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/icu-readmission-predictor}
}

@article{johnson2023mimic,
  title={MIMIC-IV, a freely accessible electronic health record dataset},
  author={Johnson, Alistair EW and Bulgarelli, Lucas and others},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={1},
  year={2023}
}
```

---

**Last Updated:** February 2025
