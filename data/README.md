# Data Folder

⚠️ **This folder should contain your MIMIC-IV data extract (NOT included in repository).**

---

## 📁 Expected File

Place your MIMIC-IV extract here:

```
data/
└── model_dataset_readmission_30d.parquet  ← Your MIMIC-IV extract
```

**Size:** ~200–500 MB  
**Format:** Parquet (preferred) or CSV

---

## 🚫 DO NOT COMMIT DATA TO GITHUB

The `.gitignore` file automatically blocks:
- `*.parquet`
- `*.csv`
- `*.npy`
- Any patient-level data files

**Why?**
- MIMIC-IV Data Use Agreement prohibits public sharing
- GitHub has 100 MB file size limit
- Privacy / HIPAA compliance

---

## 🔐 How to Obtain MIMIC-IV Data

### **Step 1: Complete Training**
- Program: CITI "Data or Specimens Only Research"
- Link: https://about.citiprogram.org/
- Duration: ~2–3 hours

### **Step 2: Request Access**
1. Create PhysioNet account: https://physionet.org/register/
2. Navigate to MIMIC-IV: https://physionet.org/content/mimiciv/2.2/
3. Click "Request Access"
4. Upload CITI certificate
5. Sign Data Use Agreement

### **Step 3: Download**
- Approval typically takes 1–3 business days
- Download MIMIC-IV v2.2 (~100 GB compressed)

---

## 🧮 Extracting the Dataset

The analysis notebook expects a **pre-processed** MIMIC-IV extract with these columns:

### **Required Features (181 total):**
- Demographics (age, gender, race, etc.)
- Vital signs (HR, BP, temperature, SpO2, etc.)
- Laboratory values (creatinine, glucose, lactate, etc.)
- Severity scores (SOFA, GCS, Charlson, etc.)
- Clinical flags (sepsis, AKI, shock, etc.)
- Utilization (ICU LOS, hospital LOS, prior admissions, etc.)

### **Target Variable:**
- `readmit_30d_flag` (binary: 0 = no readmission, 1 = readmitted within 30 days)

---

## 🔄 Preprocessing Pipeline

If starting from raw MIMIC-IV tables, you'll need to:

1. **Extract cohort** (SQL queries on MIMIC-IV database)
2. **Compute features** (aggregations, first-24h values, etc.)
3. **Apply exclusions** (age <18, died in-hospital, ICU LOS <4h)
4. **Save to Parquet** (`model_dataset_readmission_30d.parquet`)

**Reference:** See MIMIC-IV documentation for table schemas and SQL examples:
- https://mimic.mit.edu/docs/iv/

---

## ✅ Verification

Once you place the file, verify it:

```python
import pandas as pd

# Load data
df = pd.read_parquet("data/model_dataset_readmission_30d.parquet")

# Check shape
print(f"Shape: {df.shape}")  # Expected: ~48,676 rows × ~234 columns

# Check target
print(df['readmit_30d_flag'].value_counts())
# Expected:
#   0    43776  (no readmission)
#   1     4900  (readmission)
```

---

## 📧 Questions?

**MIMIC-IV Access Issues:**  
- Email: mimic-support@physionet.org
- Forum: https://github.com/MIT-LCP/mimic-code/discussions

**This Repository Issues:**  
- GitHub Issues: https://github.com/yourusername/icu-readmission-predictor/issues
