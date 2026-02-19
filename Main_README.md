

## 📦 Project Artifacts

### 🗄️ SQL Queries — Data Extraction Pipeline
Production-grade BigQuery SQL for MIMIC-IV cohort extraction and feature engineering

📂 **[View SQL Queries →](sql/)**

- **6,303 lines** of modular, well-documented BigQuery SQL
- **7 feature engineering domains** (vitals, labs, meds, comorbidities & more)
- **6 built-in quality checks** (QC1–QC6) covering coverage, ranges, and bias
- Estimated runtime: ~45 minutes on full MIMIC-IV

---

### 📓 Jupyter Notebook — Analysis Pipeline
Complete reproducible analysis from raw data to trained model

📂 **[View Notebook →](notebooks/ICU_readmission_analysis_CLEAN.ipynb)**

- Data cleaning, validation, and exploratory analysis
- Feature engineering and missingness handling
- Model training, hyperparameter tuning (Optuna), and evaluation
- SHAP-based interpretability and clinical validation

---

### 🚀 Streamlit App — Live Demo
Interactive risk calculator deployed on Streamlit Cloud

🌐 **[Launch Live App →](https://app.streamlit.app)** &nbsp;|&nbsp; 📂 **[View Code →](streamlit_app/)**

- Patient-level 30-day readmission risk score
- Model performance dashboard
- SHAP feature importance visualization
- Clinical recommendations engine

---

### 📊 Project Presentation
Comprehensive slide deck covering methodology, results, and deployment strategy

📂 **[View Presentation →](docs/presentation/ICU_Readmission_Presentation.pdf)**

- Problem statement & clinical significance
- Data engineering and feature extraction walkthrough
- Model development and benchmarking
- Deployment architecture and impact assessment
- Duration: ~15–20 minutes

---

### 📄 Documentation

| Document | Description |
|----------|-------------|
| 📄 [Results Summary →](docs/RESULTS_SUMMARY.md) | Full performance metrics and analysis |
| 📄 [Data Access Guide →](docs/DATA_STATEMENT.md) | How to obtain MIMIC-IV access |
| 📄 [Methodology →](docs/METHODOLOGY.md) | Detailed technical approach |

---
