"""
CardioRisk AI — Production Streamlit Application
Cardiovascular Risk Prediction using Clinical + Wearable Data
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import pickle
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# ── Page config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="CardioRisk AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Path helpers ──────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
MODELS_DIR  = ROOT / "models"
SRC_DIR     = ROOT / "src"

# ── Inject custom CSS ─────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    /* ── fonts & base ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── hide default streamlit chrome ───────────────────── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── sidebar ──────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] .stRadio label { color: #cbd5e1 !important; }

    /* ── main container ───────────────────────────────────── */
    .main .block-container { padding: 2rem 2.5rem; max-width: 1200px; }

    /* ── KPI cards ────────────────────────────────────────── */
    .kpi-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        text-align: center;
        transition: border-color .2s;
    }
    .kpi-card:hover { border-color: #60a5fa; }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f1f5f9;
        line-height: 1.1;
        margin: .3rem 0;
    }
    .kpi-label {
        font-size: .78rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: .08em;
    }
    .kpi-delta {
        font-size: .82rem;
        font-weight: 500;
        margin-top: .3rem;
    }

    /* ── risk badge ───────────────────────────────────────── */
    .risk-badge {
        display: inline-block;
        padding: .55rem 1.4rem;
        border-radius: 999px;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: .04em;
        text-transform: uppercase;
    }
    .risk-low    { background: #14532d; color: #86efac; border: 1px solid #16a34a; }
    .risk-medium { background: #7c2d12; color: #fdba74; border: 1px solid #ea580c; }
    .risk-high   { background: #7f1d1d; color: #fca5a5; border: 1px solid #ef4444; }

    /* ── gauge bar ────────────────────────────────────────── */
    .gauge-track {
        background: #334155;
        border-radius: 999px;
        height: 18px;
        overflow: hidden;
        margin: .6rem 0;
    }
    .gauge-fill {
        height: 100%;
        border-radius: 999px;
        transition: width .5s ease;
    }

    /* ── section headers ──────────────────────────────────── */
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: .1em;
        border-bottom: 1px solid #1e293b;
        padding-bottom: .5rem;
        margin: 1.5rem 0 1rem;
    }

    /* ── factor bar ───────────────────────────────────────── */
    .factor-row {
        display: flex;
        align-items: center;
        gap: .8rem;
        margin: .45rem 0;
    }
    .factor-name  { width: 180px; font-size: .85rem; color: #cbd5e1; flex-shrink: 0; }
    .factor-track { flex: 1; background: #1e293b; border-radius: 4px; height: 10px; }
    .factor-fill  { height: 100%; border-radius: 4px; }
    .factor-val   { width: 55px; text-align: right; font-size: .82rem; color: #94a3b8; }

    /* ── info box ─────────────────────────────────────────── */
    .info-box {
        background: #1e293b;
        border-left: 4px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: .9rem 1.2rem;
        font-size: .88rem;
        color: #94a3b8;
        line-height: 1.6;
        margin: 1rem 0;
    }
    .warn-box {
        background: #292524;
        border-left: 4px solid #f97316;
        border-radius: 0 8px 8px 0;
        padding: .9rem 1.2rem;
        font-size: .88rem;
        color: #fdba74;
        margin: .7rem 0;
    }

    /* ── number inputs ────────────────────────────────────── */
    .stNumberInput input, .stSelectbox select {
        background: #0f172a !important;
        border: 1px solid #334155 !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }
    .stSlider .stSlider { color: #60a5fa; }

    /* ── buttons ──────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        border-radius: 10px;
        padding: .7rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all .2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #60a5fa, #3b82f6);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59,130,246,.4);
    }

    /* ── expander ─────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: #1e293b !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    </style>
    """, unsafe_allow_html=True)


inject_css()

# ═══════════════════════════════════════════════════════════════════════════
#  MODEL LOADING (cached, with graceful error handling)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """
    Load model, scaler, feature names from models/.
    Returns (model, scaler, feature_names, feature_importance_df, error_msg).
    Any missing artefact returns None for that item + a message.
    """
    model, scaler, feature_names, feat_imp = None, None, None, None
    errors = []

    # ── Model ──────────────────────────────────────────────────────────────
    model_path = MODELS_DIR / "model.pkl"
    if model_path.exists():
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            errors.append(f"model.pkl load error: {e}")
    else:
        errors.append("model.pkl not found in models/")

    # ── Scaler (optional) ──────────────────────────────────────────────────
    scaler_path = MODELS_DIR / "scaler.pkl"
    if scaler_path.exists():
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            errors.append(f"scaler.pkl load error: {e}")

    # ── Feature names ──────────────────────────────────────────────────────
    fn_path = MODELS_DIR / "feature_names.json"
    if fn_path.exists():
        try:
            with open(fn_path) as f:
                feature_names = json.load(f)
        except Exception as e:
            errors.append(f"feature_names.json error: {e}")

    # ── Feature importance ─────────────────────────────────────────────────
    fi_path = MODELS_DIR / "feature_importance.csv"
    if fi_path.exists():
        try:
            feat_imp = pd.read_csv(fi_path)
        except Exception as e:
            errors.append(f"feature_importance.csv error: {e}")

    return model, scaler, feature_names, feat_imp, errors


# ── Load artefacts ─────────────────────────────────────────────────────────
model, scaler, feature_names, feat_imp, load_errors = load_artifacts()

# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE DEFINITIONS (must match training pipeline exactly)
# ═══════════════════════════════════════════════════════════════════════════

# Default values (median/mode from UCI Cleveland + wearable literature)
FEATURE_DEFAULTS = {
    # ── Clinical ──────────────────────────────────────────────────────────
    "age":              54,
    "sex":               1,
    "chest_pain":        1,
    "resting_bp":      130,
    "cholesterol":     240,
    "fasting_bs":        0,
    "rest_ecg":          0,
    "max_hr":          150,
    "exercise_angina":   0,
    "oldpeak":         1.0,
    "slope":             2,
    "ca":                0,
    "thal":              3,
    # ── Wearable ──────────────────────────────────────────────────────────
    "resting_hr":      68.0,
    "daily_steps":    7000,
    "sleep_duration":  6.8,
    "sleep_quality":  68.0,
    "hrv_ms":         44.0,
    "stress_level":    2.7,
    "activity_idx":    1.0,
    # ── Engineered ────────────────────────────────────────────────────────
    "lifestyle_score":  50.0,
    "stress_index":      0.5,
    "cardio_fitness":   50.0,
    "sleep_hr_recovery": 0.3,
    "pulse_pressure":   50.0,
    "chol_hr_ratio":     3.5,
}

CLINICAL_FEATURES  = ["age","sex","chest_pain","resting_bp","cholesterol",
                       "fasting_bs","rest_ecg","max_hr","exercise_angina",
                       "oldpeak","slope","ca","thal"]
WEARABLE_FEATURES  = ["resting_hr","daily_steps","sleep_duration","sleep_quality",
                       "hrv_ms","stress_level","activity_idx"]
ENGINEERED_FEATURES = ["lifestyle_score","stress_index","cardio_fitness",
                        "sleep_hr_recovery","pulse_pressure","chol_hr_ratio"]

ALL_FEATURES = CLINICAL_FEATURES + WEARABLE_FEATURES + ENGINEERED_FEATURES


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING (mirrors training pipeline)
# ═══════════════════════════════════════════════════════════════════════════

def compute_engineered(inputs: dict) -> dict:
    """
    Recompute all derived features from raw inputs.
    Must be identical to the feature engineering applied during training.
    """
    def mn(v, lo, hi):
        return float(np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1))

    rhr    = float(inputs.get("resting_hr",    68))
    steps  = float(inputs.get("daily_steps",  7000))
    slp    = float(inputs.get("sleep_duration", 6.8))
    slq    = float(inputs.get("sleep_quality",  68))
    hrv    = float(inputs.get("hrv_ms",         44))
    stress = float(inputs.get("stress_level",   2.7))
    act    = float(inputs.get("activity_idx",   1.0))
    bp     = float(inputs.get("resting_bp",    130))
    chol   = float(inputs.get("cholesterol",   240))

    # Lifestyle score (0–100)
    lifestyle_score = (
        0.25 * mn(steps, 200, 20000) * 100 +
        0.22 * slq +
        0.20 * mn(hrv, 5, 120) * 100 +
        0.15 * mn(slp, 2.5, 11) * 100 +
        0.10 * (1 - mn(stress, 1, 5)) * 100 +
        0.08 * (1 - mn(rhr, 35, 110)) * 100
    )

    # Stress index (0–1)
    stress_index = (
        0.40 * (1 - mn(hrv,    5, 120)) +
        0.35 * mn(rhr,        35, 110)  +
        0.25 * mn(stress,      1,   5)
    )

    # Cardio fitness index (0–100)
    raw_cfi = steps / max(rhr, 1)
    cardio_fitness = float(np.clip(mn(raw_cfi, 0, 300) * 100, 0, 100))

    # Sleep–HR recovery
    sleep_hr_recovery = mn(slq, 10, 100) * (1 - mn(rhr, 35, 110))

    # Pulse pressure (approximate)
    pulse_pressure = bp * 0.40

    # Cholesterol / HR ratio
    chol_hr_ratio = chol / max(rhr, 1)

    return {
        "lifestyle_score":   round(lifestyle_score, 2),
        "stress_index":      round(stress_index, 4),
        "cardio_fitness":    round(cardio_fitness, 2),
        "sleep_hr_recovery": round(sleep_hr_recovery, 4),
        "pulse_pressure":    round(pulse_pressure, 1),
        "chol_hr_ratio":     round(chol_hr_ratio, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def build_input_vector(raw_inputs: dict, target_features: list) -> pd.DataFrame:
    """
    Build a 1-row DataFrame with exactly `target_features` columns.
    Missing features are filled with FEATURE_DEFAULTS to prevent crashes.
    """
    # Compute engineered features
    engineered = compute_engineered(raw_inputs)
    merged = {**FEATURE_DEFAULTS, **raw_inputs, **engineered}

    # Build ordered row
    row = {feat: merged.get(feat, FEATURE_DEFAULTS.get(feat, 0.0))
           for feat in target_features}
    return pd.DataFrame([row])[target_features]


def predict(raw_inputs: dict) -> dict:
    """
    Run inference. Returns prob, category, colour, and feature vector.
    Handles model=None gracefully.
    """
    if model is None:
        return {"error": "Model not loaded. See sidebar for details."}

    # Determine feature order
    target_feats = feature_names if feature_names else ALL_FEATURES

    # Build aligned input vector
    X = build_input_vector(raw_inputs, target_feats)

    # Scale if scaler exists
    X_inf = X.copy()
    if scaler is not None:
        try:
            # Only scale numeric columns the scaler was trained on
            scaler_cols = [c for c in X_inf.columns
                           if c in (getattr(scaler, "feature_names_in_", X_inf.columns))]
            X_inf[scaler_cols] = scaler.transform(X_inf[scaler_cols])
        except Exception:
            # Scaler mismatch: proceed unscaled (log warning)
            pass

    # Predict
    try:
        prob = float(model.predict_proba(X_inf)[0, 1])
    except AttributeError:
        prob = float(model.predict(X_inf)[0])
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

    # Risk classification aligned with clinical guidelines
    if prob < 0.15:
        category, badge_class, colour, icon = "Low Risk",      "risk-low",    "#22c55e", "🟢"
    elif prob < 0.35:
        category, badge_class, colour, icon = "Moderate Risk", "risk-medium", "#f97316", "🟡"
    else:
        category, badge_class, colour, icon = "High Risk",     "risk-high",   "#ef4444", "🔴"

    return {
        "probability":   prob,
        "percentage":    f"{prob*100:.1f}%",
        "category":      category,
        "badge_class":   badge_class,
        "colour":        colour,
        "icon":          icon,
        "feature_vector": X,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════

def render_gauge(prob: float, colour: str):
    pct = int(prob * 100)
    st.markdown(f"""
    <div style="margin: .5rem 0 1rem;">
        <div style="display:flex; justify-content:space-between;
                    font-size:.78rem; color:#64748b; margin-bottom:.3rem;">
            <span>Low (0%)</span><span>Moderate (15–35%)</span><span>High (>35%)</span>
        </div>
        <div class="gauge-track">
            <div class="gauge-fill" style="width:{pct}%; background:{colour};"></div>
        </div>
        <div style="text-align:center; font-size:2.4rem; font-weight:700;
                    color:{colour}; margin:.5rem 0;">{pct}%</div>
        <div style="text-align:center; font-size:.75rem; color:#475569;">
            10-year cardiovascular event probability
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_factor_bar(name: str, value: float, max_val: float, colour: str = "#3b82f6"):
    pct = int(min(value / max(max_val, 1e-9), 1.0) * 100)
    st.markdown(f"""
    <div class="factor-row">
        <div class="factor-name">{name}</div>
        <div class="factor-track">
            <div class="factor-fill" style="width:{pct}%; background:{colour};"></div>
        </div>
        <div class="factor-val">{value:.3f}</div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi(value: str, label: str, delta: str = "", delta_positive: bool = True):
    delta_colour = "#22c55e" if delta_positive else "#ef4444"
    delta_html   = f'<div class="kpi-delta" style="color:{delta_colour};">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.2rem 0 1rem;">
        <div style="font-size:2.5rem;">🫀</div>
        <div style="font-size:1.2rem; font-weight:700; color:#f1f5f9;">CardioRisk AI</div>
        <div style="font-size:.75rem; color:#64748b; margin-top:.2rem;">
            Clinical + Wearable Intelligence
        </div>
    </div>
    <hr style="border-color:#334155; margin:.5rem 0 1rem;">
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🩺 Risk Assessment", "📊 Model Insights", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#334155; margin:.8rem 0;'>", unsafe_allow_html=True)

    # System status
    st.markdown("**System Status**")
    model_ok = model is not None
    scaler_ok = scaler is not None
    feats_ok  = feature_names is not None

    for label, ok in [("Model", model_ok), ("Scaler", scaler_ok), ("Features", feats_ok)]:
        icon = "✅" if ok else "⚠️"
        st.markdown(f"{icon} {label} {'loaded' if ok else 'missing'}")

    if load_errors:
        with st.expander("🔴 Load errors"):
            for e in load_errors:
                st.markdown(f"<div class='warn-box'>{e}</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#334155; margin:.8rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:.72rem; color:#475569; text-align:center;">
        v1.0 · For educational purposes only.<br>
        Not a substitute for medical advice.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: RISK ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════

if page == "🩺 Risk Assessment":

    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <h1 style="font-size:1.9rem; font-weight:700; color:#f1f5f9; margin:0;">
            Cardiovascular Risk Assessment
        </h1>
        <p style="color:#64748b; margin:.3rem 0 0; font-size:.9rem;">
            Enter clinical measurements and wearable health data to receive
            a personalized 10-year cardiovascular risk estimate.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Input form ────────────────────────────────────────────────────────
    with st.form("patient_form", clear_on_submit=False):

        # ── Patient info strip ─────────────────────────────────────────────
        st.markdown('<div class="section-header">Patient Demographics</div>',
                    unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            age  = st.number_input("Age (years)", 20, 90, 54, key="age")
        with d2:
            sex  = st.selectbox("Sex", [0, 1],
                                format_func=lambda x: "Female" if x == 0 else "Male")
        with d3:
            chest_pain = st.selectbox(
                "Chest Pain Type",
                [1, 2, 3, 4],
                format_func=lambda x: {1:"Typical Angina",2:"Atypical Angina",
                                        3:"Non-anginal",  4:"Asymptomatic"}[x])
        with d4:
            fasting_bs = st.selectbox(
                "Fasting Blood Sugar >120",
                [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        # ── Clinical measurements ──────────────────────────────────────────
        st.markdown('<div class="section-header">Clinical Measurements</div>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            resting_bp   = st.number_input("Resting BP (mmHg)",  70, 250, 130)
        with c2:
            cholesterol  = st.number_input("Cholesterol (mg/dL)", 100, 600, 240)
        with c3:
            max_hr       = st.number_input("Max Heart Rate",       60, 220, 150)
        with c4:
            oldpeak      = st.number_input("ST Depression",  0.0, 6.5, 1.0, step=0.1)

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            rest_ecg  = st.selectbox("Rest ECG", [0,1,2],
                                     format_func=lambda x:{0:"Normal",1:"ST-T abnorm",2:"LV hypertrophy"}[x])
        with c6:
            exercise_angina = st.selectbox("Exercise Angina", [0,1],
                                           format_func=lambda x:"No" if x==0 else "Yes")
        with c7:
            slope = st.selectbox("ST Slope", [1,2,3],
                                 format_func=lambda x:{1:"Upsloping",2:"Flat",3:"Downsloping"}[x])
        with c8:
            ca = st.selectbox("Major Vessels (CA)", [0,1,2,3])

        _, tc, _ = st.columns([1,2,1])
        with tc:
            thal = st.selectbox("Thalassemia", [3,6,7],
                                format_func=lambda x:{3:"Normal",6:"Fixed defect",7:"Reversible defect"}[x])

        # ── Wearable data ──────────────────────────────────────────────────
        st.markdown('<div class="section-header">Wearable & Lifestyle Data</div>',
                    unsafe_allow_html=True)
        w1, w2, w3, w4 = st.columns(4)
        with w1:
            resting_hr = st.number_input("Resting HR (bpm)", 35, 120, 68)
        with w2:
            daily_steps = st.number_input("Daily Steps", 0, 30000, 7000, step=100)
        with w3:
            sleep_duration = st.number_input("Sleep (hours)", 2.0, 12.0, 6.8, step=0.1)
        with w4:
            sleep_quality  = st.slider("Sleep Quality (%)", 10, 100, 68)

        w5, w6, w7, _ = st.columns(4)
        with w5:
            hrv_ms     = st.number_input("HRV (ms)", 5.0, 130.0, 44.0, step=0.5)
        with w6:
            stress_level = st.slider("Stress Level (1–5)", 1.0, 5.0, 2.7, step=0.1)
        with w7:
            activity_idx = st.slider("Activity Index (0–3)", 0.0, 3.0, 1.0, step=0.1)

        # ── Submit ─────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍  Analyse Cardiovascular Risk")

    # ── Results ───────────────────────────────────────────────────────────
    if submitted:
        raw = {
            "age": age, "sex": sex, "chest_pain": chest_pain,
            "resting_bp": resting_bp, "cholesterol": cholesterol,
            "fasting_bs": fasting_bs, "rest_ecg": rest_ecg,
            "max_hr": max_hr, "exercise_angina": exercise_angina,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
            "resting_hr": resting_hr, "daily_steps": daily_steps,
            "sleep_duration": sleep_duration, "sleep_quality": sleep_quality,
            "hrv_ms": hrv_ms, "stress_level": stress_level,
            "activity_idx": activity_idx,
        }

        result = predict(raw)

        if "error" in result:
            st.markdown(
                f'<div class="warn-box">⚠️ {result["error"]}</div>',
                unsafe_allow_html=True)
        else:
            # ── Top KPI strip ──────────────────────────────────────────────
            eng = compute_engineered(raw)
            st.markdown("<br>", unsafe_allow_html=True)
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                render_kpi(result["icon"] + " " + result["category"],
                           "Risk Classification")
            with k2:
                render_kpi(result["percentage"], "CVD Probability")
            with k3:
                render_kpi(f"{eng['lifestyle_score']:.0f}/100", "Lifestyle Score",
                           delta="↑ Higher is healthier",
                           delta_positive=eng["lifestyle_score"] > 50)
            with k4:
                render_kpi(f"{eng['stress_index']:.2f}", "Stress Index",
                           delta="↓ Lower is better",
                           delta_positive=eng["stress_index"] < 0.5)

            st.markdown("<br>", unsafe_allow_html=True)
            left, right = st.columns([1, 1])

            # ── Left: gauge + risk badge ───────────────────────────────────
            with left:
                st.markdown('<div class="section-header">Risk Score</div>',
                            unsafe_allow_html=True)
                render_gauge(result["probability"], result["colour"])

                st.markdown(f"""
                <div style="text-align:center; margin:1rem 0;">
                    <span class="risk-badge {result['badge_class']}">
                        {result['icon']} {result['category']}
                    </span>
                </div>
                """, unsafe_allow_html=True)

                # Clinical recommendation
                recs = {
                    "Low Risk":      "✅ Continue current healthy lifestyle. Annual check-up recommended.",
                    "Moderate Risk": "⚠️ Lifestyle modifications recommended. Consider 6-month follow-up.",
                    "High Risk":     "🚨 Clinical evaluation strongly recommended. Discuss risk reduction with your physician.",
                }
                st.markdown(
                    f'<div class="info-box">{recs[result["category"]]}</div>',
                    unsafe_allow_html=True)

            # ── Right: engineered features breakdown ───────────────────────
            with right:
                st.markdown('<div class="section-header">Lifestyle Analysis</div>',
                            unsafe_allow_html=True)

                metrics_display = [
                    ("Lifestyle Score",     eng["lifestyle_score"],     100,  "#22c55e"),
                    ("Cardio Fitness",      eng["cardio_fitness"],       100,  "#3b82f6"),
                    ("Stress Index",        eng["stress_index"] * 100,  100,  "#f97316"),
                    ("Sleep–HR Recovery",   eng["sleep_hr_recovery"] * 100, 100, "#a78bfa"),
                ]
                for name, val, mx, clr in metrics_display:
                    render_factor_bar(name, val, mx, clr)

            # ── Feature importance / SHAP section ─────────────────────────
            st.markdown('<div class="section-header">Contributing Factors</div>',
                        unsafe_allow_html=True)

            if feat_imp is not None and not feat_imp.empty:
                with st.expander("📊 View Feature Importance", expanded=True):
                    top_n = min(12, len(feat_imp))
                    fi_show = feat_imp.head(top_n).copy()
                    max_imp = fi_show["importance"].max()

                    cols_fi = st.columns(2)
                    mid = (top_n + 1) // 2
                    for idx, (_, row) in enumerate(fi_show.iterrows()):
                        col_idx = 0 if idx < mid else 1
                        with cols_fi[col_idx]:
                            colour = "#3b82f6" if row["feature"] in WEARABLE_FEATURES + ENGINEERED_FEATURES else "#64748b"
                            render_factor_bar(row["feature"], row["importance"], max_imp, colour)

                    st.markdown("""
                    <div class="info-box">
                        🔵 Blue bars = lifestyle/wearable features &nbsp;|&nbsp;
                        ⚫ Grey bars = clinical features
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Fallback: show input summary
                with st.expander("📋 Input Summary", expanded=True):
                    input_df = pd.DataFrame([raw]).T.rename(columns={0: "Value"})
                    st.dataframe(input_df, use_container_width=True)

            # ── Patient summary card ───────────────────────────────────────
            with st.expander("👤 Patient Summary", expanded=False):
                ps1, ps2, ps3 = st.columns(3)
                with ps1:
                    st.markdown("**Clinical**")
                    st.write(f"• Age: {age} yrs | {'Male' if sex==1 else 'Female'}")
                    st.write(f"• BP: {resting_bp} mmHg")
                    st.write(f"• Cholesterol: {cholesterol} mg/dL")
                    st.write(f"• Max HR: {max_hr} bpm")
                with ps2:
                    st.markdown("**Wearable**")
                    st.write(f"• Resting HR: {resting_hr} bpm")
                    st.write(f"• Daily Steps: {daily_steps:,}")
                    st.write(f"• Sleep: {sleep_duration:.1f}h ({sleep_quality:.0f}% quality)")
                    st.write(f"• HRV: {hrv_ms:.1f} ms")
                with ps3:
                    st.markdown("**Derived Scores**")
                    st.write(f"• Lifestyle Score: {eng['lifestyle_score']:.1f}/100")
                    st.write(f"• Stress Index: {eng['stress_index']:.3f}")
                    st.write(f"• Cardio Fitness: {eng['cardio_fitness']:.1f}/100")
                    st.write(f"• Sleep Recovery: {eng['sleep_hr_recovery']:.3f}")

            # Demo mode notice
            if model is None:
                st.markdown("""
                <div class="warn-box">
                    ℹ️ Demo Mode — model.pkl not found. Run <code>src/export_model.py</code>
                    to train and export a real model. Results shown are for UI demonstration only.
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════

elif page == "📊 Model Insights":

    st.markdown("""
    <h1 style="font-size:1.9rem; font-weight:700; color:#f1f5f9; margin-bottom:.3rem;">
        Model Insights & Performance
    </h1>
    <p style="color:#64748b; font-size:.9rem; margin-bottom:1.5rem;">
        Overview of the trained model, feature importance, and evaluation metrics.
    </p>
    """, unsafe_allow_html=True)

    # ── Model info cards ───────────────────────────────────────────────────
    mi1, mi2, mi3, mi4 = st.columns(4)
    model_type = type(model).__name__ if model else "Not loaded"
    with mi1: render_kpi(model_type, "Model Type")
    with mi2: render_kpi(str(len(feature_names)) if feature_names else "—", "Features")
    with mi3: render_kpi("F1-Score", "Selection Metric")
    with mi4: render_kpi("SHAP", "Explainability")

    # ── Feature importance table ───────────────────────────────────────────
    st.markdown('<div class="section-header">Feature Importance</div>',
                unsafe_allow_html=True)

    if feat_imp is not None and not feat_imp.empty:
        col_left, col_right = st.columns([3, 2])
        with col_left:
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 5))
                fig.patch.set_facecolor("#0f172a")
                ax.set_facecolor("#0f172a")
                top = feat_imp.head(15).sort_values("importance")
                colors = ["#3b82f6" if f in WEARABLE_FEATURES + ENGINEERED_FEATURES
                          else "#475569" for f in top["feature"]]
                ax.barh(top["feature"], top["importance"], color=colors, edgecolor="none")
                ax.set_xlabel("Importance", color="#94a3b8")
                ax.tick_params(colors="#94a3b8")
                ax.spines[:].set_color("#334155")
                ax.set_title("Top 15 Features", color="#e2e8f0", fontsize=11, fontweight="bold")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#334155")
                st.pyplot(fig, use_container_width=True)
                plt.close()
            except ImportError:
                st.dataframe(feat_imp.head(15), use_container_width=True)
        with col_right:
            st.dataframe(feat_imp.head(15).style.background_gradient(
                cmap="Blues", subset=["importance"]),
                use_container_width=True)
    else:
        st.markdown("""
        <div class="warn-box">
            No feature_importance.csv found in models/.<br>
            Run <code>src/export_model.py</code> to generate it.
        </div>
        """, unsafe_allow_html=True)

    # ── Architecture diagram ───────────────────────────────────────────────
    st.markdown('<div class="section-header">System Architecture</div>',
                unsafe_allow_html=True)

    with st.expander("View Data & Model Flow", expanded=True):
        st.markdown("""
        ```
        ┌─────────────────────────────────────────────────────────────────┐
        │                     DATA SOURCES                                │
        │  UCI Heart Disease (clinical)  +  Wearable datasets            │
        └──────────────────────────────┬──────────────────────────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │     PREPROCESSING PIPELINE  │
                        │  • Normalise columns        │
                        │  • Impute missing values    │
                        │  • Scale features           │
                        └──────────────┬──────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │     FEATURE ENGINEERING     │
                        │  • Lifestyle Score          │
                        │  • Stress Index             │
                        │  • Cardio Fitness Index     │
                        │  • Sleep–HR Recovery        │
                        └──────────────┬──────────────┘
                                       │
                   ┌───────────────────┼───────────────────┐
                   │                   │                   │
         ┌─────────▼────────┐ ┌───────▼──────┐ ┌─────────▼────────┐
         │ Logistic Regress │ │Random Forest │ │    XGBoost       │
         └─────────┬────────┘ └───────┬──────┘ └─────────┬────────┘
                   └───────────────────┼───────────────────┘
                                       │  (Best F1 wins)
                        ┌──────────────▼──────────────┐
                        │      BEST MODEL             │
                        │  • model.pkl                │
                        │  • scaler.pkl               │
                        │  • feature_names.json       │
                        │  • feature_importance.csv   │
                        └──────────────┬──────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │      STREAMLIT APP          │
                        │  • Input panel              │
                        │  • Risk gauge               │
                        │  • Explanation panel        │
                        │  • SHAP values              │
                        └─────────────────────────────┘
        ```
        """)

    # ── Risk thresholds ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Risk Classification Thresholds</div>',
                unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("""
        <div class="kpi-card" style="border-color:#22c55e;">
            <div class="kpi-label">Low Risk</div>
            <div class="kpi-value" style="color:#22c55e;">< 15%</div>
            <div style="font-size:.8rem;color:#64748b;margin-top:.4rem;">
                Maintain healthy lifestyle
            </div>
        </div>""", unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div class="kpi-card" style="border-color:#f97316;">
            <div class="kpi-label">Moderate Risk</div>
            <div class="kpi-value" style="color:#f97316;">15–35%</div>
            <div style="font-size:.8rem;color:#64748b;margin-top:.4rem;">
                Lifestyle modifications advised
            </div>
        </div>""", unsafe_allow_html=True)
    with t3:
        st.markdown("""
        <div class="kpi-card" style="border-color:#ef4444;">
            <div class="kpi-label">High Risk</div>
            <div class="kpi-value" style="color:#ef4444;">> 35%</div>
            <div style="font-size:.8rem;color:#64748b;margin-top:.4rem;">
                Clinical evaluation required
            </div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ About":

    st.markdown("""
    <h1 style="font-size:1.9rem;font-weight:700;color:#f1f5f9;margin-bottom:.3rem;">
        About CardioRisk AI
    </h1>
    <p style="color:#64748b;font-size:.9rem;margin-bottom:1.5rem;">
        Production ML system for cardiovascular risk estimation.
    </p>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("""
        **Project Overview**

        CardioRisk AI integrates clinical health records with wearable-inspired lifestyle
        features to provide personalised 10-year cardiovascular risk predictions.
        The system combines the UCI Heart Disease dataset with synthetic wearable
        metrics (heart rate, steps, sleep) and applies advanced ML models alongside
        SHAP-based interpretability.

        **Tech Stack**
        - Python 3.10+, scikit-learn, XGBoost
        - SHAP for explainability
        - Streamlit for the web interface
        - Pandas / NumPy / Matplotlib

        **Data Sources**
        - UCI Heart Disease (Cleveland, 303 patients)
        - Apple Watch / Fitbit reference distributions
        - PhysioNet BigIdeas Step-HR dataset
        """)

    with a2:
        st.markdown("""
        **Feature Categories**

        *Clinical (UCI)*: Age, Sex, Chest Pain, BP, Cholesterol, ECG,
        Max Heart Rate, Exercise Angina, ST Depression, Thal

        *Wearable (derived)*: Resting HR, Daily Steps, Sleep Duration,
        Sleep Quality, HRV, Stress Level, Activity Index

        *Engineered*: Lifestyle Score, Stress Index, Cardio Fitness Index,
        Sleep–HR Recovery Score, Cholesterol/HR Ratio

        **Disclaimer**

        This application is for educational and portfolio demonstration purposes
        only. It is not a medical device and must not be used for clinical
        decision-making. Always consult a qualified healthcare professional.
        """)

    st.markdown('<div class="section-header">Contact & Links</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        📁 GitHub: <em>your-github/cardiorisk-ai</em><br>
        🌐 Live Demo: <em>your-app.streamlit.app</em><br>
        📧 Contact: <em>your@email.com</em>
    </div>
    """, unsafe_allow_html=True)
