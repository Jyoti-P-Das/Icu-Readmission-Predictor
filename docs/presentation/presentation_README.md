# 📊 Project Presentation

> Comprehensive slide deck covering the end-to-end ICU 30-Day Readmission Risk Prediction project — from raw clinical data to a live deployed application.

---

## 📥 View Presentation

**👉 [ICU_Readmission_Presentation.pdf](ICU_Readmission_Presentation.pdf)**

*Click the link above — GitHub will render the PDF directly in your browser. No download needed.*

---

## 🎯 At a Glance

| Detail | Info |
|--------|------|
| **Topic** | ICU 30-Day Readmission Risk Prediction |
| **Dataset** | MIMIC-IV v2.2 (48,676 patients) |
| **Model Performance** | AUC-ROC: 0.7884 on held-out test set |
| **Audience** | Data science hiring managers, healthcare analytics teams, ML engineers |
| **Duration** | ~15–20 minutes |
| **Format** | PDF (converted from PowerPoint) |

---

## 📋 Slide Breakdown

### 🔴 Section 1 — Problem & Clinical Background *(Slides 1–3)*
- Why ICU readmissions matter clinically and financially
- Current gaps in predictive tools
- Project goals and success criteria

### 🟡 Section 2 — Data & Methodology *(Slides 4–6)*
- MIMIC-IV dataset overview (48,676 patients, 225+ features)
- SQL-based data extraction pipeline (6,303 lines of BigQuery SQL)
- Feature engineering across 7 clinical domains

### 🟢 Section 3 — Model Development *(Slides 7–10)*
- Baseline models evaluated: Logistic Regression, Random Forest, XGBoost, LightGBM
- Hyperparameter tuning with Optuna (40 trials)
- Model selection rationale and cross-validation strategy

### 🔵 Section 4 — Results *(Slides 11–14)*
- Test set performance: **AUC-ROC 0.7884**
- Benchmarked against 6 published MIMIC-IV studies
- SHAP-based feature importance and clinical interpretability
- Missingness analysis (MNAR detection)

### 🟣 Section 5 — Deployment *(Slides 15–17)*
- Streamlit app architecture
- Live risk calculator interface walkthrough
- Clinical recommendation engine design

### ⚫ Section 6 — Impact & Next Steps *(Slides 18–20)*
- Potential clinical use cases
- Limitations and ethical considerations
- Future work and improvements

---

## 🏆 Key Takeaways

1. **End-to-End Ownership** — SQL extraction → Python modeling → Streamlit deployment
2. **Production-Grade Engineering** — 6,303 lines of SQL with 6 quality checks
3. **Clinical Validity** — Evidence-based features, MNAR analysis, clinically meaningful thresholds
4. **Strong Performance** — 0.7884 AUC-ROC, competitive with published literature
5. **Live Demo** — Deployed on Streamlit Cloud, accessible to anyone

---

## 🔗 Related Links

| Resource | Link |
|----------|------|
| 🚀 Live Streamlit App | [Launch App](https://your-app-name.streamlit.app) |
| 🗄️ SQL Queries | [View SQL →](../../sql/) |
| 📓 Analysis Notebook | [View Notebook →](../../notebooks/) |
| 📄 Results Summary | [View Results →](../RESULTS_SUMMARY.md) |
| 📄 Data Access Guide | [View Data Statement →](../DATA_STATEMENT.md) |
| 🏠 Main Repository | [Back to README →](../../README.md) |

---

*Questions? Open a GitHub Issue or contact: your.email@example.com*
