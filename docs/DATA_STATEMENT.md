# Data Access Statement

---

## 📊 Dataset Source

This project uses **MIMIC-IV v2.2** (Medical Information Mart for Intensive Care, version IV):

- **Institution:** Beth Israel Deaconess Medical Centre, Boston, MA
- **Time Period:** 2008–2019
- **Patient Population:** De-identified ICU admissions
- **Size:** ~70,000 ICU stays
- **Public Access:** Yes (with credentialing)

**Official Links:**
- **Main website:** https://mimic.mit.edu/
- **PhysioNet repository:** https://physionet.org/content/mimiciv/2.2/
- **Documentation:** https://mimic.mit.edu/docs/iv/

---

## 🔐 Data Access Requirements

### **This repository does NOT include the raw MIMIC-IV data.**

Users must obtain independent access through PhysioNet:

### **Step-by-Step Access Process:**

1. **Complete Required Training**
   - Program: CITI "Data or Specimens Only Research"
   - Link: https://about.citiprogram.org/
   - Duration: ~2–3 hours
   - Certificate required for PhysioNet application

2. **Create PhysioNet Account**
   - Register at: https://physionet.org/register/
   - Use institutional email address (recommended)

3. **Request MIMIC-IV Access**
   - Navigate to: https://physionet.org/content/mimiciv/2.2/
   - Click "Request Access"
   - Upload CITI certificate
   - Sign Data Use Agreement

4. **Approval Timeline**
   - Review typically takes 1–3 business days
   - Approval notification sent via email

5. **Download Dataset**
   - After approval, download MIMIC-IV v2.2
   - File format: Compressed CSV / Parquet
   - Size: ~100 GB compressed

---

## 📜 Data Use Agreement (DUA) — Key Terms

By accessing MIMIC-IV, you agree to:

✅ **Permitted Uses:**
- Research and education only
- Publication of aggregate results
- Sharing derived models (without patient-level data)

❌ **Prohibited:**
- Patient re-identification attempts
- Sharing raw data with unauthorized users
- Commercial use without separate agreement
- Uploading patient-level data to public repositories

**Full DUA:** https://physionet.org/content/mimiciv/view-license/2.2/

---

## 🔄 Reproducibility — How to Use This Repository

### **Option 1: Run Analysis from Scratch**

If you have MIMIC-IV access:

```bash
# 1. Clone this repository
git clone https://github.com/yourusername/icu-readmission-predictor.git
cd icu-readmission-predictor

# 2. Place your MIMIC-IV extract here:
#    data/model_dataset_readmission_30d.parquet
#    (Not provided — you must extract from MIMIC-IV yourself)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the analysis notebook
jupyter notebook notebooks/icu_readmission_analysis.ipynb
```

**Note:** The notebook assumes a pre-processed MIMIC-IV extract. If starting from raw MIMIC-IV tables, additional SQL extraction is required (see MIMIC-IV documentation).

---

### **Option 2: Use Pre-Trained Model (No Data Required)**

If you do NOT have MIMIC-IV access but want to use the model:

```bash
# The model artifacts are included in this repository:
# - model/final_model.pkl
# - model/preprocessing_pipeline_FITTED.pkl
# - model/feature_names_after_preprocessing.txt

# Run the Streamlit app (demo mode):
streamlit run streamlit_app/app.py
```

**Note:** The Streamlit app demonstrates the model but uses simplified inputs. Full clinical deployment would require all 181 raw features.

---

## 🧬 Dataset Schema

### **Cohort Definition:**

**Inclusion Criteria:**
- Age ≥18 years at ICU admission
- ICU stay ≥4 hours
- Survived index ICU admission

**Exclusion Criteria:**
- Paediatric patients (<18 years)
- Died during index ICU stay
- Very short ICU stays (<4 hours, likely administrative)

**Final Cohort:** 48,676 ICU admissions

---

### **Feature Categories:**

| Category | Count | Examples |
|----------|-------|----------|
| Demographics | 8 | Age, gender, race, insurance |
| Vital Signs | 32 | HR, BP, temperature, SpO2, respiratory rate |
| Laboratory | 58 | Creatinine, glucose, lactate, electrolytes |
| Severity Scores | 15 | SOFA, GCS, Charlson, APS-III, SAPS-II |
| Clinical Flags | 45 | Sepsis, AKI, shock, arrhythmia |
| Utilization | 12 | ICU LOS, hospital LOS, prior admissions |
| MNAR Indicators | 3 | Height, weight, urine output not measured |
| Derived Features | 7 | Pulse pressure, shock index, BMI |

**Total:** 181 features (before preprocessing) → 247 (after one-hot encoding)

---

### **Outcome Variable:**

- **Name:** `readmit_30d_flag`
- **Definition:** Binary indicator (0/1)
  - **0:** No ICU readmission within 30 days of discharge
  - **1:** ICU readmission within 30 days of discharge
- **Prevalence:** 10.07% (4,900 / 48,676)

**Note:** Readmissions to the same hospital only (transfers to other facilities not tracked).

---

## 🔒 Privacy & De-identification

### **MIMIC-IV De-identification Methods:**

1. **Dates shifted:** All timestamps shifted randomly (±3 years per patient, consistent within patient)
2. **Ages >89 capped:** Patients >89 years set to 300+ (comply with HIPAA Safe Harbor)
3. **Rare diagnoses suppressed:** Very uncommon diagnoses removed
4. **Names removed:** All patient/provider names redacted
5. **Geographic info removed:** Only hospital location retained

**Result:** Re-identification risk is minimal under HIPAA standards.

---

## 🧪 Ethical Approval

### **Institutional Review Board (IRB):**

- **Approval:** MIMIC-IV has blanket IRB approval from MIT and Beth Israel Deaconess
- **Status:** Approved for unrestricted research use
- **Consent waiver:** Granted (retrospective de-identified data)

**Reference:**  
Johnson et al. (2023). "MIMIC-IV, a freely accessible electronic health record dataset." *Scientific Data*, 10(1), 1.

---

## 📚 Recommended Citation

If you use MIMIC-IV in your work, please cite:

```bibtex
@article{johnson2023mimic,
  title={MIMIC-IV, a freely accessible electronic health record dataset},
  author={Johnson, Alistair EW and Bulgarelli, Lucas and Pollard, Tom and 
          Horng, Steven and Celi, Leo Anthony and Mark, Roger},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={1},
  year={2023},
  publisher={Nature Publishing Group},
  doi={10.1038/s41597-022-01899-x}
}
```

---

## 🤝 Acknowledgements

We thank:
- **PhysioNet** for hosting and distributing MIMIC-IV
- **MIT Laboratory for Computational Physiology** for creating and maintaining MIMIC
- **Beth Israel Deaconess Medical Centre** for contributing the data
- **NIH / NIBIB** for funding MIMIC development

---

## 📧 Data Questions

**MIMIC-IV Technical Questions:**  
- Forum: https://github.com/MIT-LCP/mimic-code/discussions
- Email: mimic-support@physionet.org

**This Repository Questions:**  
- GitHub Issues: https://github.com/yourusername/icu-readmission-predictor/issues
- Contact: jyotiprakash.das@example.com

---

**Last Updated:** February 2025
