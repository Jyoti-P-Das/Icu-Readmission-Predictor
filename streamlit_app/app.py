import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ICU Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GLOBAL CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0f1923;
    }
    section[data-testid="stSidebar"] * {
        color: #e0e8f0 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #c0d0e0 !important;
        font-size: 0.95rem;
    }

    /* Main background */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Cards */
    .risk-card {
        border-radius: 14px;
        padding: 22px 26px;
        margin: 6px 0;
        border: 1.5px solid rgba(255,255,255,0.12);
    }

    /* Metric overrides */
    [data-testid="metric-container"] {
        background: #f7f9fc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1a56db, #1e40af);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.4rem;
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1e40af, #1e3a8a);
        transform: translateY(-1px);
        box-shadow: 0 4px 18px rgba(26, 86, 219, 0.38);
    }

    /* Section headers */
    h3 { color: #0f172a; font-weight: 600; }
    h4 { color: #1e40af; font-weight: 600; font-size: 1rem; letter-spacing: 0.4px; text-transform: uppercase; }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 500;
        font-size: 0.96rem;
    }

    /* Divider */
    hr { border-color: #e2e8f0; margin: 1.2rem 0; }

    /* Info boxes */
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── PATHS (GITHUB-COMPATIBLE) ──────────────────────────────────────────────────
# Get the repository root directory
REPO_ROOT = Path(__file__).resolve().parent.parent

# Define paths relative to repo root
ASSETS_DIR = REPO_ROOT / "streamlit_app" / "assets"
MODEL_DIR = REPO_ROOT / "model"

# ── CACHED LOADERS ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained LightGBM model"""
    model_path = MODEL_DIR / "final_model.pkl"
    if not model_path.exists():
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()
    return joblib.load(model_path)

@st.cache_data
def load_feature_names():
    """Load feature names after preprocessing"""
    path = MODEL_DIR / "feature_names_after_preprocessing.txt"
    if not path.exists():
        st.error(f"❌ Feature names file not found: {path}")
        st.stop()
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

@st.cache_data
def load_results():
    """Load test evaluation results CSV"""
    path = ASSETS_DIR / "test_evaluation_results.csv"
    if not path.exists():
        st.warning(f"⚠️ Results file not found: {path}")
        return None
    return pd.read_csv(path)

@st.cache_data
def load_feature_importance():
    """Load feature importance CSV"""
    path = ASSETS_DIR / "feature_importance_combined.csv"
    if not path.exists():
        st.warning(f"⚠️ Feature importance file not found: {path}")
        return None
    return pd.read_csv(path)

@st.cache_data
def load_narratives():
    """Load clinical narratives CSV"""
    path = ASSETS_DIR / "clinical_narratives_top20.csv"
    if not path.exists():
        st.warning(f"⚠️ Narratives file not found: {path}")
        return None
    return pd.read_csv(path)

# ── HELPERS ────────────────────────────────────────────────────────────────────
def get_risk_label(prob):
    if prob >= 0.5:
        return "HIGH RISK", "#dc2626", "🔴"
    elif prob >= 0.3:
        return "MEDIUM RISK", "#d97706", "🟡"
    else:
        return "LOW RISK", "#16a34a", "🟢"

# ── GAUGE CHART ────────────────────────────────────────────────────────────────
def make_gauge_chart(probability):
    """
    Cartesian Wedge gauge.
    prob=0   → 180° → LEFT  (LOW)
    prob=0.5 →  90° → TOP   (MED)
    prob=1   →   0° → RIGHT (HIGH)
    """
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    # Shadow ring
    shadow = Wedge((0,0), 1.07, 0, 180, width=0.42, color="#e2e8f0", zorder=1)
    ax.add_patch(shadow)

    # Colour zones
    ax.add_patch(Wedge((0,0), 1.04, 120, 180, width=0.36, facecolor="#22c55e", alpha=0.90, zorder=2))
    ax.add_patch(Wedge((0,0), 1.04,  60, 120, width=0.36, facecolor="#f59e0b", alpha=0.90, zorder=2))
    ax.add_patch(Wedge((0,0), 1.04,   0,  60, width=0.36, facecolor="#ef4444", alpha=0.90, zorder=2))

    # Inner white circle (donut)
    ax.add_patch(Wedge((0,0), 0.68, 0, 180, width=0.01, facecolor="white", edgecolor="white", zorder=3))
    ax.add_patch(plt.Circle((0,0), 0.67, color="white", zorder=4))

    # Boundary ticks
    for deg in [0, 60, 120, 180]:
        r = np.radians(deg)
        ax.plot([0.67*np.cos(r), 0.70*np.cos(r)], [0.67*np.sin(r), 0.70*np.sin(r)],
                color="#64748b", lw=2.2, zorder=5)

    # Minor ticks
    for deg in range(0, 181, 12):
        if deg % 60 == 0: continue
        r = np.radians(deg)
        ax.plot([0.67*np.cos(r), 0.685*np.cos(r)], [0.67*np.sin(r), 0.685*np.sin(r)],
                color="#94a3b8", lw=1, zorder=5)

    # NEEDLE
    needle_deg = 180.0 - (float(probability) * 180.0)
    needle_rad = np.radians(needle_deg)
    tip_x  =  0.80 * np.cos(needle_rad)
    tip_y  =  0.80 * np.sin(needle_rad)
    tail_x = -0.12 * np.cos(needle_rad)
    tail_y = -0.12 * np.sin(needle_rad)

    ax.plot([tail_x, tip_x], [tail_y, tip_y], color="#0f172a", linewidth=3.8,
            solid_capstyle="round", zorder=7)
    ax.annotate("",
        xy=(tip_x, tip_y),
        xytext=(tail_x*0.3 + tip_x*0.7, tail_y*0.3 + tip_y*0.7),
        arrowprops=dict(arrowstyle="-|>", color="#0f172a", lw=2.8, mutation_scale=18),
        zorder=8)

    # Pivot
    ax.add_patch(plt.Circle((0,0), 0.075, color="#0f172a", zorder=9))
    ax.add_patch(plt.Circle((0,0), 0.042, color="white",   zorder=10))

    # Probability text
    _, risk_color, _ = get_risk_label(probability)
    ax.text(0, -0.20, f"{probability:.1%}",
            ha="center", va="center", fontsize=26, fontweight="bold",
            color=risk_color, zorder=11,
            fontfamily="DejaVu Sans")

    # Zone labels
    ax.text(-0.93, 0.12, "LOW",  ha="center", fontsize=9, fontweight="bold", color="#15803d", zorder=11)
    ax.text( 0.00, 0.76, "MED",  ha="center", fontsize=9, fontweight="bold", color="#92400e", zorder=11)
    ax.text( 0.93, 0.12, "HIGH", ha="center", fontsize=9, fontweight="bold", color="#991b1b", zorder=11)

    # Boundary labels
    for deg, label in [(180,"0%"), (120,"33%"), (60,"67%"), (0,"100%")]:
        r = np.radians(deg)
        ax.text(0.54*np.cos(r), 0.54*np.sin(r), label,
                ha="center", va="center", fontsize=6.5, color="#64748b", zorder=11)

    ax.set_xlim(-1.38, 1.38)
    ax.set_ylim(-0.45, 1.28)
    plt.tight_layout(pad=0.1)
    return fig


# ── CLINICAL RECOMMENDATION ENGINE ────────────────────────────────────────────
def get_clinical_recommendations(factors):
    """
    Given a list of active risk factors, return specific
    patient-tailored clinical recommendations.
    """
    recs = {}

    for label, value, impact, is_risk in factors:
        key = label

        if "Long hospital stay" in label:
            recs[label] = {
                "icon": "🏥",
                "title": "Extended Stay — Discharge Planning",
                "points": [
                    "Initiate multidisciplinary discharge planning ≥48 h before planned discharge",
                    "Arrange home health or step-down facility if functional status is reduced",
                    "Ensure all pending investigations are reviewed before discharge",
                    "Medication reconciliation: stop/adjust in-hospital-only medications",
                ],
                "color": "#fef2f2", "border": "#ef4444"
            }

        elif "AKI" in label and is_risk:
            recs[label] = {
                "icon": "🫘",
                "title": "Acute Kidney Injury — Renal Management",
                "points": [
                    "Hold or dose-adjust nephrotoxic agents (NSAIDs, ACE-I/ARB, contrast agents)",
                    "Ensure adequate fluid resuscitation — target urine output ≥0.5 mL/kg/hr",
                    "Monitor creatinine and electrolytes daily until trend improves",
                    "Arrange nephrology outpatient follow-up within 4 weeks of discharge",
                    "Educate patient on signs of worsening kidney function (oliguria, oedema)",
                ],
                "color": "#fff7ed", "border": "#f97316"
            }

        elif "No acute kidney injury" in label:
            recs[label] = {
                "icon": "✅",
                "title": "Kidney Function — Protective",
                "points": [
                    "Maintain adequate hydration throughout admission",
                    "Routine renal function check at follow-up in 4–6 weeks",
                ],
                "color": "#f0fdf4", "border": "#22c55e"
            }

        elif "Emergency-type admission" in label:
            recs[label] = {
                "icon": "⚡",
                "title": "Emergency Presentation — High Acuity Care",
                "points": [
                    "Full clinical assessment including weight, height, BMI before discharge",
                    "Ensure primary care physician is notified of admission and plan",
                    "Schedule GP follow-up within 5 days of discharge",
                    "Provide written safety-netting instructions and red-flag symptoms",
                    "Consider social work referral if support network is limited",
                ],
                "color": "#fefce8", "border": "#eab308"
            }

        elif "Recent prior hospitalisation" in label:
            recs[label] = {
                "icon": "🔄",
                "title": "Frequent Admissions — Readmission Prevention",
                "points": [
                    "Conduct root-cause analysis: why was the patient readmitted so soon?",
                    "Arrange intensive case management or a dedicated re-admission prevention programme",
                    "Phone follow-up within 24–48 h of this discharge",
                    "Review if underlying chronic disease is adequately controlled",
                    "Involve pharmacy for medication adherence review",
                ],
                "color": "#fef2f2", "border": "#dc2626"
            }

        elif "Severe anaemia" in label:
            recs[label] = {
                "icon": "🩸",
                "title": "Severe Anaemia — Haematological Management",
                "points": [
                    "Assess aetiology: iron deficiency, haemolysis, blood loss, renal anaemia",
                    "Consider IV iron infusion if iron-deficient and haemodynamically stable",
                    "Transfusion if Hb <7 g/dL or symptomatic (dyspnoea, chest pain)",
                    "Arrange haematology or primary care follow-up with repeat FBC in 2–4 weeks",
                    "Avoid discharge with untreated Hb <8 g/dL in cardiac patients",
                ],
                "color": "#fff1f2", "border": "#f43f5e"
            }

        elif "High comorbidity burden" in label:
            recs[label] = {
                "icon": "📋",
                "title": "High Comorbidity Burden — Multimorbidity Management",
                "points": [
                    "Prioritise the 3–5 most clinically active conditions in discharge plan",
                    "Polypharmacy review: de-prescribe medications with poor benefit-to-risk ratio",
                    "Arrange single coordinated outpatient review rather than multiple clinic letters",
                    "Refer to chronic disease management programme if available",
                    "Ensure clear documentation of advance care preferences",
                ],
                "color": "#f8faff", "border": "#6366f1"
            }

        elif "High organ dysfunction" in label:
            recs[label] = {
                "icon": "⚠️",
                "title": "High SOFA Score — Multi-Organ Dysfunction",
                "points": [
                    "Do not discharge until SOFA trend is clearly improving for ≥24 h",
                    "Identify the leading organ dysfunction (respiratory, renal, hepatic, coagulation)",
                    "Daily senior review until SOFA ≤4",
                    "Consider step-down HDU/IMC rather than direct ward discharge",
                    "Arrange specialist outpatient follow-up for the dominant organ involved",
                ],
                "color": "#fef2f2", "border": "#dc2626"
            }

        elif "Advanced age" in label:
            recs[label] = {
                "icon": "👴",
                "title": "Advanced Age — Frailty & Falls Assessment",
                "points": [
                    "Perform Clinical Frailty Scale or equivalent assessment",
                    "Physiotherapy review for functional independence before discharge",
                    "Occupational therapy home assessment if mobility is reduced",
                    "Review polypharmacy — deprescribe high falls-risk medications (sedatives, antihypertensives)",
                    "Ensure carer or family member is briefed on discharge plan",
                ],
                "color": "#f5f3ff", "border": "#8b5cf6"
            }

        elif "Prolonged ICU stay" in label:
            recs[label] = {
                "icon": "🛏️",
                "title": "Prolonged ICU Stay — Post-ICU Rehabilitation",
                "points": [
                    "Early physiotherapy: passive then active exercises to prevent ICU-acquired weakness",
                    "Nutritional optimisation: dietitian review, ensure adequate caloric intake",
                    "Psychological support: ICU survivors have high rates of PTSD and depression",
                    "Arrange ICU follow-up clinic at 3 months if available",
                    "Warn patient and family about post-intensive care syndrome (PICS)",
                ],
                "color": "#f0f9ff", "border": "#0ea5e9"
            }

    return recs


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 16px 0;">
        <div style="font-size:1.35rem;font-weight:700;color:#e2e8f0;letter-spacing:0.3px;">
            🏥 ICU Readmission
        </div>
        <div style="font-size:0.82rem;color:#94a3b8;margin-top:3px;">Prediction Tool · MIMIC-IV</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔮 Patient Risk Predictor", "📊 Model Performance", "🔬 Feature Importance"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("""
    <div style="font-size:0.82rem;color:#64748b;line-height:1.8;">
        <div><span style="color:#94a3b8;">Model</span> &nbsp; LightGBM (Tuned)</div>
        <div><span style="color:#94a3b8;">AUC</span> &nbsp;&nbsp;&nbsp; 0.7884</div>
        <div><span style="color:#94a3b8;">Data</span> &nbsp;&nbsp;&nbsp; MIMIC-IV</div>
        <div><span style="color:#94a3b8;">N</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 48,676 patients</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":

    st.markdown("""
    <div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a8a 60%,#1d4ed8 100%);
                padding:40px 36px;border-radius:16px;color:white;margin-bottom:28px;
                box-shadow:0 8px 32px rgba(30,64,175,0.25);">
        <div style="font-size:0.8rem;letter-spacing:2px;color:#93c5fd;text-transform:uppercase;
                    font-weight:500;margin-bottom:10px;">Clinical Decision Support</div>
        <h1 style="color:white;margin:0;font-size:2rem;font-weight:700;line-height:1.3;">
            ICU 30-Day Readmission<br>Prediction System
        </h1>
        <p style="color:#bfdbfe;font-size:1.05em;margin-top:12px;max-width:640px;line-height:1.6;">
            Machine learning tool built on MIMIC-IV data to identify high-risk ICU patients
            before discharge and guide targeted post-discharge interventions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test AUC-ROC", "0.7884", "+0.013 vs baseline")
    c2.metric("Patients",     "48,676", "MIMIC-IV 2008–2019")
    c3.metric("Features",     "247",    "clinical variables")
    c4.metric("Readmission",  "10.07%", "30-day ICU rate")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🎯 What This Tool Does")
        st.markdown("""
Predicts the probability that an ICU patient will be **readmitted within 30 days** of discharge.

**Workflow:**
1. Enter patient data in the Predictor page
2. Model calculates readmission probability (0–100 %)
3. Risk classified as Low / Medium / High
4. Personalised clinical recommendations generated

**Intended for:** ICU physicians, discharge nurses, clinical leads
        """)

    with col_b:
        st.markdown("#### 📊 Model Development")
        st.markdown("""
| Step | Details |
|------|---------|
| Dataset | MIMIC-IV · Beth Israel Deaconess |
| Patients | 48,676 ICU admissions |
| Algorithms tested | LR, RF, XGBoost, LightGBM |
| Tuning | Optuna — 40 trials |
| Test set | 9,736 patients (held-out) |

**Final model: LightGBM · AUC = 0.7884**
        """)

    st.markdown("---")
    st.markdown("#### 🔬 Top 3 Risk Factors")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
<div style="background:#eff6ff;padding:18px;border-radius:12px;border-left:4px solid #2563eb;">
<b>🏥 Hospital Length of Stay</b><br>
<span style="color:#374151;font-size:0.9em;">Longer stays signal higher severity and elevated re-admission risk.</span>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div style="background:#f0fdf4;padding:18px;border-radius:12px;border-left:4px solid #16a34a;">
<b>🫘 KDIGO AKI Stage</b><br>
<span style="color:#374151;font-size:0.9em;">Acute kidney injury stage is the second strongest predictor.</span>
</div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
<div style="background:#fff7ed;padding:18px;border-radius:12px;border-left:4px solid #ea580c;">
<b>⚖️ Body Weight</b><br>
<span style="color:#374151;font-size:0.9em;">Influences dosing, ventilation settings, and recovery trajectory.</span>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.info("👈 Navigate to **Patient Risk Predictor** in the sidebar to use the model.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PATIENT RISK PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Patient Risk Predictor":

    st.markdown("""
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
                padding:22px 26px;margin-bottom:20px;">
        <h2 style="margin:0 0 6px 0;color:#0f172a;">🔮 Patient Risk Predictor</h2>
        <p style="margin:0;color:#64748b;font-size:0.95rem;">
            Complete the clinical parameters below and click <b>Calculate Risk</b>.
            Results and personalised recommendations will appear immediately below.
        </p>
    </div>
    """, unsafe_allow_html=True)

    try:
        model         = load_model()
        feature_names = load_feature_names()
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        st.stop()

    # ── INPUT FORM ─────────────────────────────────────────────────────────────
    st.markdown("#### 📋 Patient Clinical Data")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""<div style="background:#f0f7ff;border-radius:10px;padding:4px 14px 10px;
                        border-left:3px solid #3b82f6;margin-bottom:10px;">
            <span style="font-size:0.78rem;font-weight:600;color:#1d4ed8;letter-spacing:1px;
                         text-transform:uppercase;">🏥 Hospital Utilisation</span></div>""",
            unsafe_allow_html=True)
        hospital_los = st.slider("Hospital Length of Stay (days)", 0.0, 60.0, 5.0, 0.5,
                                  help="Total days in hospital during this admission")
        icu_los_hours = st.slider("ICU Length of Stay (hours)", 0.0, 500.0, 48.0, 1.0,
                                   help="Total hours spent in ICU")
        days_since_dc = st.number_input("Days Since Last Hospital Discharge", 0, 365, 90,
                                         help="0 = readmission straight away — higher risk")

        st.markdown("""<div style="background:#fff7ed;border-radius:10px;padding:4px 14px 10px;
                        border-left:3px solid #f97316;margin:14px 0 10px;">
            <span style="font-size:0.78rem;font-weight:600;color:#c2410c;letter-spacing:1px;
                         text-transform:uppercase;">🫘 Kidney Function</span></div>""",
            unsafe_allow_html=True)
        kdigo = st.selectbox("KDIGO Acute Kidney Injury Stage", [0, 1, 2, 3],
                              help="0=None · 1=Mild · 2=Moderate · 3=Severe")
        urine_rate = st.number_input("Urine Output Rate (mL/kg/hr)", 0.0, 5.0, 0.8, 0.1,
                                      help="Normal: 0.5–1.0 mL/kg/hr")

    with col2:
        st.markdown("""<div style="background:#f5f3ff;border-radius:10px;padding:4px 14px 10px;
                        border-left:3px solid #8b5cf6;margin-bottom:10px;">
            <span style="font-size:0.78rem;font-weight:600;color:#6d28d9;letter-spacing:1px;
                         text-transform:uppercase;">👤 Demographics</span></div>""",
            unsafe_allow_html=True)
        weight_kg = st.number_input("Body Weight (kg)", 30.0, 250.0, 75.0, 0.5)
        height_q  = st.radio("Was height measured at admission?",
                              ["Yes — routine admission",
                               "No — emergency (no time for routine vitals)"],
                              help="Emergency admissions often skip anthropometrics — this itself is a risk signal")
        height_flag = 1 if "Yes" in height_q else 0
        age = st.slider("Age at Admission (years)", 18, 100, 65)

        st.markdown("""<div style="background:#f0fdf4;border-radius:10px;padding:4px 14px 10px;
                        border-left:3px solid #22c55e;margin:14px 0 10px;">
            <span style="font-size:0.78rem;font-weight:600;color:#15803d;letter-spacing:1px;
                         text-transform:uppercase;">🔬 Laboratory & Severity Scores</span></div>""",
            unsafe_allow_html=True)
        hematocrit = st.number_input("Hematocrit Minimum — first 24 h (%)", 10.0, 60.0, 35.0, 0.5,
                                      help="Normal: 36–50 %. Below 30% = significant anaemia.")
        charlson = st.slider("Charlson Comorbidity Index", 0, 15, 2,
                              help="Chronic disease burden. Higher = more comorbidities.")
        sofa = st.slider("SOFA Score — first 24 h", 0, 24, 3,
                          help="Organ dysfunction severity. Higher = more severe.")

    # ── CALCULATE BUTTON (right here, between form and results) ───────────────
    st.markdown("<div style='margin:18px 0 4px 0;'>", unsafe_allow_html=True)
    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        calculate = st.button(
            "🔮  CALCULATE READMISSION RISK",
            use_container_width=True,
            type="primary"
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ── RESULTS (only shown after button click) ────────────────────────────────
    if calculate:
        # Build feature vector
        X_input = np.zeros((1, len(feature_names)))
        feature_map = {
            "continuous__hospital_los_days":               hospital_los,
            "continuous__kdigo_stage_max_first_24h":       float(kdigo),
            "continuous__weight_kg":                       weight_kg,
            "binary__height_available_flag":               float(height_flag),
            "continuous__index_icu_los_hours":             icu_los_hours,
            "continuous__index_icu_los_minutes":           icu_los_hours * 60.0,
            "continuous__hematocrit_first_24h_min":        hematocrit,
            "continuous__days_since_last_discharge":       float(days_since_dc),
            "continuous__bmi":                             weight_kg / (1.70 ** 2),
            "continuous__charlson_comorbidity_index":      float(charlson),
            "continuous__sofa_score_first_24h":            float(sofa),
            "continuous__age_at_admission":                float(age),
            "continuous__urine_output_rate_ml_per_kg_hr":  urine_rate,
        }
        for fname, val in feature_map.items():
            if fname in feature_names:
                X_input[0, feature_names.index(fname)] = val

        prob = float(model.predict_proba(X_input)[0, 1])
        risk_label, risk_color, risk_icon = get_risk_label(prob)

        # ── GAUGE + VERDICT ────────────────────────────────────────────────────
        st.markdown("### 📊 Prediction Result")
        col_g, col_r = st.columns([1, 1], gap="large")

        with col_g:
            st.pyplot(make_gauge_chart(prob), use_container_width=True)

        with col_r:
            st.markdown(f"""
<div style="background:{risk_color}14;padding:30px 26px;border-radius:14px;
            border:2px solid {risk_color};text-align:center;margin-top:16px;">
    <div style="font-size:0.78rem;letter-spacing:2px;color:{risk_color};
                text-transform:uppercase;font-weight:600;margin-bottom:6px;">
        30-Day Readmission Probability
    </div>
    <div style="font-size:3.4rem;font-weight:800;color:{risk_color};line-height:1.1;">
        {prob:.1%}
    </div>
    <div style="margin-top:12px;background:{risk_color}22;border-radius:8px;
                padding:8px 14px;display:inline-block;">
        <span style="font-size:1.15rem;font-weight:700;color:{risk_color};">
            {risk_icon} {risk_label}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:14px;'>", unsafe_allow_html=True)
            if prob >= 0.5:
                st.error("""
**⚠️ High Risk — Immediate Actions Required:**
- Extended monitoring before discharge
- Intensive discharge planning & social work review
- GP / specialist follow-up within 7 days
- Refer to home health services
                """)
            elif prob >= 0.3:
                st.warning("""
**⚡ Medium Risk — Enhanced Follow-up:**
- Phone check-in within 72 hours of discharge
- Clinic follow-up within 14 days
- Written medication instructions provided
                """)
            else:
                st.success("""
**✅ Low Risk — Standard Protocol:**
- Routine clinic follow-up in 2–4 weeks
- Standard written discharge instructions
                """)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── KEY FACTORS ────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔍 Key Risk Factors for This Patient")

        factors = []
        if hospital_los > 10:
            factors.append(("🏥 Long hospital stay",         f"{hospital_los:.0f} days",                 "Risk ↑",  True))
        if kdigo >= 2:
            factors.append(("🩺 Significant AKI",            f"KDIGO Stage {kdigo}",                     "Risk ↑↑", True))
        if kdigo == 0:
            factors.append(("🩺 No acute kidney injury",     "KDIGO Stage 0",                            "Risk ↓",  False))
        if height_flag == 0:
            factors.append(("⚡ Emergency-type admission",   "Height not recorded",                      "Risk ↑",  True))
        if days_since_dc < 30:
            factors.append(("🔄 Recent prior hospitalisation", f"{days_since_dc} days since discharge",  "Risk ↑↑", True))
        if hematocrit < 30:
            factors.append(("🩸 Severe anaemia",             f"Hct {hematocrit:.0f}% (normal 36–50%)",  "Risk ↑",  True))
        if charlson >= 5:
            factors.append(("📋 High comorbidity burden",    f"Charlson = {charlson}",                   "Risk ↑",  True))
        if sofa >= 8:
            factors.append(("⚠️ High organ dysfunction",    f"SOFA = {sofa}",                           "Risk ↑↑", True))
        if age >= 75:
            factors.append(("👴 Advanced age",               f"Age {age} years",                         "Risk ↑",  True))
        if icu_los_hours > 120:
            factors.append(("🛏️ Prolonged ICU stay",        f"{icu_los_hours:.0f} h in ICU",            "Risk ↑",  True))

        if not factors:
            st.info("No major individual risk flags detected for the values entered.")
        else:
            # Factor summary cards
            cols = st.columns(min(len(factors), 3))
            for i, (label, value, impact, is_risk) in enumerate(factors[:6]):
                bg     = "#fef2f2" if is_risk else "#f0fdf4"
                border = "#ef4444" if is_risk else "#22c55e"
                icon_c = "#dc2626" if is_risk else "#16a34a"
                with cols[i % 3]:
                    st.markdown(f"""
<div style="background:{bg};padding:14px 16px;border-radius:11px;
            border-left:4px solid {border};margin:5px 0;min-height:80px;">
    <div style="font-weight:600;color:#0f172a;font-size:0.92rem;">{label}</div>
    <div style="color:#475569;font-size:0.85rem;margin-top:3px;">{value}</div>
    <div style="color:{icon_c};font-weight:700;font-size:0.88rem;margin-top:5px;">{impact}</div>
</div>""", unsafe_allow_html=True)

            # ── PER-FACTOR CLINICAL RECOMMENDATIONS ────────────────────────────
            st.markdown("---")
            st.markdown("### 💊 Personalised Clinical Recommendations")
            st.markdown("""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
            padding:12px 18px;margin-bottom:18px;">
    <span style="color:#475569;font-size:0.9rem;">
        The following recommendations are tailored to the risk factors identified above.
        Each section corresponds to a detected risk factor and provides specific
        clinical actions for stabilisation and safe discharge.
    </span>
</div>
""", unsafe_allow_html=True)

            recs = get_clinical_recommendations(factors)

            if recs:
                for label, rec in recs.items():
                    with st.expander(
                        f"{rec['icon']}  {rec['title']}",
                        expanded=True
                    ):
                        st.markdown(f"""
<div style="background:{rec['color']};border-left:4px solid {rec['border']};
            border-radius:0 10px 10px 0;padding:14px 18px;">
""", unsafe_allow_html=True)
                        for point in rec['points']:
                            st.markdown(f"""
<div style="display:flex;align-items:flex-start;margin-bottom:9px;">
    <span style="color:{rec['border']};font-weight:700;margin-right:10px;
                 margin-top:1px;flex-shrink:0;">▸</span>
    <span style="color:#1e293b;font-size:0.93rem;line-height:1.55;">{point}</span>
</div>""", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No specific recommendations triggered — review standard discharge protocol.")

        # ── DISCLAIMER ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
            padding:12px 18px;">
    <span style="font-size:0.82rem;color:#64748b;">
        ⚠️ <b>Clinical Disclaimer:</b> This tool is for decision support only and does not replace
        clinical judgment. All outputs must be interpreted alongside full clinical assessment.
        Model trained on MIMIC-IV data (Beth Israel Deaconess Medical Centre, 2008–2019).
    </span>
</div>
""", unsafe_allow_html=True)

    else:
        # Placeholder when button not yet clicked
        st.markdown("""
<div style="background:#f8fafc;border:2px dashed #cbd5e1;border-radius:14px;
            padding:48px 24px;text-align:center;">
    <div style="font-size:2.5rem;margin-bottom:12px;">🔮</div>
    <div style="font-size:1.1rem;font-weight:600;color:#475569;margin-bottom:6px;">
        Results will appear here
    </div>
    <div style="color:#94a3b8;font-size:0.92rem;">
        Complete the patient data above and click
        <b style="color:#1d4ed8;">Calculate Readmission Risk</b>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":

    st.markdown("""
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
                padding:22px 26px;margin-bottom:20px;">
        <h2 style="margin:0 0 6px 0;color:#0f172a;">📊 Model Performance</h2>
        <p style="margin:0;color:#64748b;font-size:0.95rem;">
            All figures from the <b>held-out test set</b> (9,736 patients — never used in training or tuning).
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC-ROC",              "0.7884", "+0.013 vs baseline")
    c2.metric("AUC-PR",               "0.3569", "+0.046 vs baseline")
    c3.metric("Precision @ 70% Rec.", "0.2266")
    c4.metric("Brier Score",          "0.1668", help="Lower is better — 0 = perfect")

    st.markdown("---")
    st.markdown("#### 📊 All Models — Test Set Comparison")

    results = load_results()
    if results is not None:
        display_cols = [c for c in
            ["model","test_auc","test_auc_pr","test_precision_70","val_auc","val_test_gap"]
            if c in results.columns]
        rename_map = {
            "model":             "Model",
            "test_auc":          "Test AUC ✅",
            "test_auc_pr":       "Test AUC-PR ✅",
            "test_precision_70": "Precision @ 70% Recall",
            "val_auc":           "Val AUC ⚠️",
            "val_test_gap":      "Overfitting Gap"
        }
        st.dataframe(results[display_cols].rename(columns=rename_map), use_container_width=True)
    else:
        st.warning("Test evaluation results CSV not found in assets folder")

    st.markdown("---")
    col_i1, col_i2 = st.columns(2)
    roc_img  = ASSETS_DIR / "roc_curves_test.png"
    over_img = ASSETS_DIR / "overfitting_analysis.png"

    with col_i1:
        st.markdown("#### ROC Curves (Test Set)")
        if roc_img.exists():
            st.image(str(roc_img), use_container_width=True)
        else:
            st.warning("roc_curves_test.png not found in assets folder")

    with col_i2:
        st.markdown("#### Validation vs Test Gap")
        if over_img.exists():
            st.image(str(over_img), use_container_width=True)
        else:
            st.warning("overfitting_analysis.png not found in assets folder")

    st.markdown("---")
    st.markdown("#### 💡 Interpreting the Metrics")
    col1, col2 = st.columns(2)

    with col1:
        st.info("""
**AUC-ROC = 0.7884**

The model correctly ranks a readmission patient above a non-readmission patient
**79 times out of 100**. Published ICU readmission AUC range is 0.74–0.80.
This model sits above the published average.
        """)
    with col2:
        st.info("""
**Precision @ 70% Recall = 0.2266**

Catching 70% of all readmissions means ~1 in 4 flagged patients will actually readmit.
At a 10% baseline rate, this is a **2.25× lift** over random screening.
        """)

    st.markdown("---")
    st.markdown("#### 📚 Comparison with Published MIMIC-IV Literature")
    st.caption("Comparing only with studies using the same dataset (MIMIC-IV) for fair performance evaluation")

    # Load literature comparison CSV
    lit_path = ASSETS_DIR / "literature_comparison_mimiciv.csv"
    if lit_path.exists():
        lit = pd.read_csv(lit_path)
        lit = lit.sort_values("AUC-ROC", ascending=False)
        
        # Display table
        st.dataframe(
            lit[["Study", "AUC-ROC", "Method", "Year", "Cohort_Size"]].rename(columns={
                "Study": "Study",
                "AUC-ROC": "AUC-ROC",
                "Method": "Model/Method",
                "Year": "Year",
                "Cohort_Size": "Cohort Size"
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Add interpretation note
        st.info("""
        **Interpretation:** Our model (AUC 0.7884) ranks in the **middle-to-upper range** among 
        published MIMIC-IV ICU readmission studies. Performance is competitive with traditional 
        ML methods (Random Forest, XGBoost) and slightly below deep learning approaches.
        """)
    else:
        # Fallback if CSV not found
        st.warning("Literature comparison CSV not found. Please add `literature_comparison_mimiciv.csv` to assets folder.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Feature Importance":

    st.markdown("""
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
                padding:22px 26px;margin-bottom:20px;">
        <h2 style="margin:0 0 6px 0;color:#0f172a;">🔬 Feature Importance & Clinical Insights</h2>
        <p style="margin:0;color:#64748b;font-size:0.95rem;">
            Understanding <b>why</b> the model predicts readmission risk.
        </p>
    </div>
    """, unsafe_allow_html=True)

    narratives  = load_narratives()
    feat_imp    = load_feature_importance()
    data_loaded = narratives is not None

    tab1, tab2, tab3 = st.tabs(["📊 Top 20 Features", "🏥 Clinical Categories", "💊 Modifiable Factors"])

    with tab1:
        st.markdown("#### 🏆 Top 20 Predictors of 30-Day ICU Readmission")
        st.caption("Ranked by combined LightGBM Gain + Permutation Importance")

        bar_img = ASSETS_DIR / "feature_importance_bar.png"
        if bar_img.exists():
            st.image(str(bar_img), use_container_width=True)
        else:
            st.warning("feature_importance_bar.png not found in assets folder")

        st.markdown("---")
        col_m, col_nm = st.columns(2)
        with col_m:
            st.markdown("""
<div style="background:#f0fdf4;padding:16px;border-radius:10px;border-left:4px solid #22c55e;">
<b>🟢 Modifiable</b><br>
<span style="font-size:0.9em;color:#374151;">Can be improved by clinical action before/after discharge.</span>
</div>""", unsafe_allow_html=True)
        with col_nm:
            st.markdown("""
<div style="background:#eff6ff;padding:16px;border-radius:10px;border-left:4px solid #3b82f6;">
<b>🔵 Non-modifiable</b><br>
<span style="font-size:0.9em;color:#374151;">Fixed characteristic — useful for risk stratification.</span>
</div>""", unsafe_allow_html=True)

        if data_loaded and narratives is not None:
            st.markdown("---")
            st.markdown("#### 📋 Clinical Meaning of Each Feature")
            feat_col   = "feature_clean" if "feature_clean" in narratives.columns else "feature"
            interp_col = "clinical_interpretation" if "clinical_interpretation" in narratives.columns else "interpretation"

            for _, row in narratives.head(20).iterrows():
                feat_name = str(row.get(feat_col, ""))
                cat       = str(row.get("category", "Clinical"))
                interp    = str(row.get(interp_col, ""))
                mod       = str(row.get("modifiable", "No"))
                imp       = float(row.get("importance", 0))
                rank      = int(row.get("rank", 0))
                mod_badge = "🟢 Modifiable" if mod == "Yes" else "🔵 Non-modifiable"

                with st.expander(f"**{rank}. {feat_name}**  ·  {cat}  ·  {mod_badge}", expanded=False):
                    ca, cb = st.columns([1, 3])
                    with ca:
                        st.metric("Importance", f"{imp:.4f}")
                        st.caption(f"Category: {cat}")
                        st.caption(f"Modifiable: {mod}")
                    with cb:
                        st.markdown(f"""<div style="background:#f8fafc;padding:14px;border-radius:8px;
                            border-left:3px solid #94a3b8;font-size:0.93em;line-height:1.6;color:#1e293b;">
                            {interp}</div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("#### 🏥 Feature Importance by Clinical Category")
        cat_img = ASSETS_DIR / "importance_by_category.png"
        if cat_img.exists():
            st.image(str(cat_img), use_container_width=True)
        else:
            st.warning("importance_by_category.png not found in assets folder")

        if data_loaded and narratives is not None and "category" in narratives.columns:
            summary = (narratives.groupby("category")
                       .agg(Features=("rank","count"),
                            Avg_Importance=("importance","mean"),
                            Total_Importance=("importance","sum"))
                       .round(4).sort_values("Total_Importance", ascending=False))
            st.markdown("#### Category Summary")
            st.dataframe(summary, use_container_width=True)

        pdp_img = ASSETS_DIR / "partial_dependence_top6.png"
        if pdp_img.exists():
            st.markdown("---")
            st.markdown("#### 📈 Partial Dependence — How Features Affect Risk")
            st.caption("Shows how predicted probability changes as each feature increases. Red dashed line = baseline readmission rate (10.07%).")
            st.image(str(pdp_img), use_container_width=True)
        else:
            st.warning("partial_dependence_top6.png not found in assets folder")

    with tab3:
        st.markdown("#### 💊 Modifiable Risk Factors — Intervention Targets")
        st.markdown("""
<div style="background:#fefce8;padding:14px 18px;border-radius:10px;border-left:4px solid #ca8a04;margin-bottom:18px;">
<b>What does Modifiable mean?</b> &nbsp; These features can be
<b>improved through direct clinical intervention</b> — correcting anaemia, optimising kidney function,
controlling glucose, managing fluid balance. Targeting these before and after discharge has
the greatest potential to reduce readmission risk.
</div>
""", unsafe_allow_html=True)

        if data_loaded and narratives is not None:
            feat_col   = "feature_clean" if "feature_clean" in narratives.columns else "feature"
            interp_col = "clinical_interpretation" if "clinical_interpretation" in narratives.columns else "interpretation"
            mods = narratives[narratives.get("modifiable", pd.Series(dtype=str)) == "Yes"]

            if len(mods) > 0:
                for _, row in mods.iterrows():
                    feat  = str(row.get(feat_col, ""))
                    cat   = str(row.get("category", "Clinical"))
                    interp = str(row.get(interp_col, ""))
                    imp   = float(row.get("importance", 0))
                    with st.expander(f"✅  **{feat}**  ·  {cat}  ·  Score: {imp:.4f}", expanded=False):
                        st.markdown(f"""<div style="background:#f0fdf4;padding:14px;border-radius:8px;
                            border-left:3px solid #22c55e;font-size:0.93em;line-height:1.6;color:#1e293b;">
                            {interp}</div>""", unsafe_allow_html=True)
            else:
                st.info("No modifiable features found. Please re-run Part 9.")

        st.markdown("---")
        st.markdown("#### 📋 Clinical Action Guide")
        ca, cb = st.columns(2)
        with ca:
            st.markdown("""
**Before Discharge:**
- 🫘 Manage AKI, hold nephrotoxic drugs
- 🩸 Correct anaemia (IV iron / transfusion)
- 💧 Optimise fluid balance & urine output
- 🔬 Address all SOFA organ dysfunction components
- 🍬 Tighten glycaemic control
            """)
        with cb:
            st.markdown("""
**Post-Discharge Planning:**
- 📞 Phone follow-up within 48 h
- 🏃 Clinic within 7 days (HIGH risk)
- 💊 Medication reconciliation before leaving
- 🏠 Home health referral if indicated
- 📋 Written instructions in plain language
            """)
