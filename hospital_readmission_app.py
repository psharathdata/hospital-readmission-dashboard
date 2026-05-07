# ============================================================
#  Hospital Readmission Analytics — Streamlit Web App
#  Run: streamlit run hospital_readmission_app.py
#  Install: pip install streamlit pandas numpy matplotlib
#           plotly scikit-learn xgboost shap imbalanced-learn
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import roc_auc_score, roc_curve, confusion_matrix
from xgboost                 import XGBClassifier
from imblearn.over_sampling  import SMOTE

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Hospital Readmission Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1rem; }
    .insight-box {
        background: #1a1d27;
        border-left: 4px solid #e85d5d;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        font-size: 0.9rem;
        color: #c8cad8;
    }
    .insight-box.teal  { border-left-color: #4db8a4; }
    .insight-box.amber { border-left-color: #f0a030; }
    .section-title {
        font-size: 1.3rem; font-weight: 700;
        color: #e0e2f0; margin: 24px 0 12px 0;
        padding-bottom: 6px; border-bottom: 1px solid #2e3148;
    }
    div[data-testid="stSidebar"] {
        background-color: #12141f;
        border-right: 1px solid #2e3148;
    }
    div[data-testid="metric-container"] {
        background: #1a1d27;
        border: 1px solid #2e3148;
        border-radius: 10px; padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

ACCENT = "#e85d5d"
SAFE   = "#4db8a4"
AMBER  = "#f0a030"

# ── HELPERS ───────────────────────────────────────────────────
def map_diagnosis(code):
    try:
        code = str(code)
        if code.startswith("V") or code.startswith("E"):
            return "External"
        c = float(code)
        if   390 <= c < 460: return "Circulatory (Heart)"
        elif 460 <= c < 520: return "Respiratory"
        elif 240 <= c < 280: return "Endocrine/Diabetes"
        elif 520 <= c < 580: return "Digestive"
        elif 580 <= c < 630: return "Genitourinary"
        elif 140 <= c < 240: return "Cancer"
        elif 290 <= c < 320: return "Mental Disorders"
        elif 800 <= c < 1000: return "Injury"
        else:                 return "Other"
    except:
        return "Other"

# ── LOAD DATA ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    # First look for diabetic_data.csv in the same folder as this app.
    # If not found, fall back to your original Desktop path.
    local_path = Path(__file__).resolve().parent / "diabetic_data.csv"
    desktop_path = Path("/Users/ghost_man/Desktop/PROJECT 2 HOSPITAL READ/archive/diabetic_data.csv")
    data_path = local_path if local_path.exists() else desktop_path
    df = pd.read_csv(data_path)
    df.replace("?", np.nan, inplace=True)
    drop_cols = ["encounter_id","patient_nbr","weight","payer_code","medical_specialty"]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    df["Readmitted30"]  = (df["readmitted"] == "<30").astype(int)
    df["age_numeric"]   = df["age"].str.extract(r"\[(\d+)").astype(int)
    df["DiagCategory"]  = df["diag_1"].apply(map_diagnosis)
    los_bins   = [0,2,4,7,14,100]
    los_labels = ["1–2 days","3–4 days","5–7 days","8–14 days","14+ days"]
    df["LOS_Band"] = pd.cut(df["time_in_hospital"], bins=los_bins, labels=los_labels)
    return df

@st.cache_resource
def train_model(df):
    df_ml = df.copy()
    cat_c = df_ml.select_dtypes(include="object").columns.tolist()
    for c in ["readmitted"]:
        if c in cat_c: cat_c.remove(c)
    for c in cat_c:
        df_ml[c] = LabelEncoder().fit_transform(df_ml[c].astype(str))
    drop = ["readmitted","DiagCategory","LOS_Band"]
    df_ml.drop(columns=drop, errors="ignore", inplace=True)
    X = df_ml.drop(columns=["Readmitted30"])
    y = df_ml["Readmitted30"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=42, stratify=y)
    X_sm, y_sm = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
    model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                           use_label_encoder=False, eval_metric="logloss",
                           random_state=42, n_jobs=-1)
    model.fit(X_sm, y_sm)
    probs = model.predict_proba(X)[:,1]
    auc   = roc_auc_score(y_te, model.predict_proba(X_te)[:,1])
    return model, X, y, auc, X.columns.tolist(), probs

def align_features_for_model(X, feature_names):
    """Keep prediction/SHAP input exactly same as training columns.
    Prevents XGBoost feature_names mismatch caused by generated columns
    such as RiskScore and RiskFlag being added to df after training.
    """
    X = X.drop(columns=["RiskScore", "RiskFlag"], errors="ignore")

    missing_cols = [c for c in feature_names if c not in X.columns]
    if missing_cols:
        raise ValueError(f"Missing model input columns: {missing_cols}")

    return X[feature_names]

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Hospital Readmission\n### Analytics Dashboard")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Overview",
        "📊 EDA Analysis",
        "🤖 ML Model",
        "🚨 Patient Risk",
        "💰 ROI Calculator",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
    **Dataset:** Diabetes 130-US Hospitals
    **Records:** 101,766 patients
    **Tools:** Python · XGBoost · SHAP
    """)

# ── LOAD ─────────────────────────────────────────────────────
try:
    df = load_data()
    model, X_feat, y_feat, auc_score, feat_names, all_probs = train_model(df)
    df["RiskScore"]  = (all_probs * 100).round(1)
    df["RiskFlag"]   = pd.cut(df["RiskScore"], bins=[0,30,60,100],
                               labels=["Low Risk","Medium Risk","High Risk"])
    data_loaded = True
except FileNotFoundError:
    data_loaded = False

if not data_loaded:
    st.error("⚠️ Dataset not found!")
    st.info("""
    📥 Download: `https://www.kaggle.com/datasets/brandao/diabetes`

    📁 File name: `diabetic_data.csv` — place in same folder as this app.

    Then run: `streamlit run hospital_readmission_app.py`
    """)
    st.stop()

rate_30 = df["Readmitted30"].mean() * 100

# ══════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# 🏥 Hospital Readmission Analytics Dashboard")
    st.markdown("*Predicting 30-day readmissions with Explainable AI — 101,766 patient records*")
    st.markdown("---")

    total      = len(df)
    readmit    = df["Readmitted30"].sum()
    high_risk  = (df["RiskScore"] > 60).sum()
    cost_total = readmit * 15000

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🏥 Total Patients",         f"{total:,}")
    c2.metric("🔴 30-Day Readmission Rate", f"{rate_30:.1f}%")
    c3.metric("📋 Readmissions",           f"{readmit:,}")
    c4.metric("💰 Est. Annual Cost",        f"${cost_total/1e6:.0f}M")
    c5.metric("🚨 High Risk Patients",      f"{high_risk:,}")

    st.markdown("---")
    c1, c2 = st.columns([1, 2])

    with c1:
        sizes  = [(df["readmitted"]=="NO").sum(),
                  (df["readmitted"]==">30").sum(),
                  readmit]
        labels = ["Not Readmitted",">30 Days","<30 Days (Target)"]
        fig = px.pie(values=sizes, names=labels,
                     color_discrete_sequence=[SAFE, AMBER, ACCENT],
                     hole=0.5, title="Readmission Overview")
        fig.update_layout(paper_bgcolor="#0f1117", font_color="#e0e2f0",
                          margin=dict(t=40,b=10,l=10,r=10), height=300)
        st.plotly_chart(fig, width="stretch")

    with c2:
        age_stats = (df.groupby("age")["Readmitted30"]
                       .agg(["sum","count","mean"])
                       .reset_index())
        age_stats["rate"] = age_stats["mean"] * 100
        age_order = ["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)",
                     "[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"]
        age_stats["age"] = pd.Categorical(age_stats["age"],
                                           categories=age_order, ordered=True)
        age_stats = age_stats.sort_values("age")

        fig2 = px.bar(age_stats, x="age", y="rate",
                      text=age_stats["rate"].round(1).astype(str)+"%",
                      color="rate",
                      color_continuous_scale=["#4db8a4","#f0a030","#e85d5d"],
                      title="30-Day Readmission Rate by Age Group")
        fig2.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                           font_color="#e0e2f0", coloraxis_showscale=False,
                           height=300, title_font_size=13)
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown("""<div class="insight-box">
        🔴 <b>Previous admissions = strongest signal</b><br>
        5+ prior visits → 2–3× the average readmission rate
        </div>""", unsafe_allow_html=True)
    with i2:
        st.markdown("""<div class="insight-box amber">
        🟠 <b>Elderly + Multi-Admit + Long Stay</b><br>
        High-risk segment: ~22% rate vs 11.2% average
        </div>""", unsafe_allow_html=True)
    with i3:
        st.markdown(f"""<div class="insight-box teal">
        🟢 <b>Model AUC: {auc_score:.3f}</b><br>
        Consistent with published academic benchmarks for this dataset
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════
elif page == "📊 EDA Analysis":
    st.markdown("# 📊 Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(
        ["🏥 Diagnosis & LOS", "👤 Age & Prior Visits", "🔥 Risk Segments"]
    )

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            diag_stats = (df.groupby("DiagCategory")["Readmitted30"]
                            .agg(["sum","count","mean"])
                            .reset_index())
            diag_stats["rate"] = diag_stats["mean"] * 100
            diag_stats = diag_stats[diag_stats["count"] > 200].sort_values("rate", ascending=True)

            fig = px.bar(diag_stats, x="rate", y="DiagCategory", orientation="h",
                         text=diag_stats["rate"].round(1).astype(str)+"%",
                         color="rate",
                         color_continuous_scale=["#4db8a4","#e85d5d"],
                         title="Readmission Rate by Diagnosis")
            fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                              font_color="#e0e2f0", coloraxis_showscale=False,
                              height=400, title_font_size=13)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, width="stretch"))
            st.markdown("""<div class="insight-box">
            Circulatory (heart) and respiratory patients have the highest
            readmission rates — and the highest treatment costs per episode.
            </div>""", unsafe_allow_html=True)

        with c2:
            los_stats = (df.groupby("LOS_Band", observed=True)["Readmitted30"]
                           .agg(["sum","count","mean"])
                           .reset_index())
            los_stats["rate"] = los_stats["mean"] * 100

            fig2 = px.bar(los_stats, x="LOS_Band", y="rate",
                          text=los_stats["rate"].round(1).astype(str)+"%",
                          color="rate",
                          color_continuous_scale=["#4db8a4","#f0a030","#e85d5d"],
                          title="Readmission Rate by Length of Stay")
            fig2.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                               font_color="#e0e2f0", coloraxis_showscale=False,
                               height=400, title_font_size=13)
            fig2.update_traces(textposition="outside")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("""<div class="insight-box amber">
            Short stays (<2 days) AND long stays (14+ days) both show high risk.
            Short = possibly discharged too early. Long = complex case.
            </div>""", unsafe_allow_html=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            prev_stats = (df[df["number_inpatient"] <= 8]
                            .groupby("number_inpatient")["Readmitted30"]
                            .agg(["sum","count","mean"])
                            .reset_index())
            prev_stats["rate"] = prev_stats["mean"] * 100

            fig = px.line(prev_stats, x="number_inpatient", y="rate",
                          markers=True, title="Readmission Rate by Prior Admissions",
                          color_discrete_sequence=[ACCENT])
            fig.add_hline(y=rate_30, line_dash="dash", line_color="yellow",
                          annotation_text=f"Avg {rate_30:.1f}%")
            fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                              font_color="#e0e2f0", height=380,
                              xaxis_title="Previous Inpatient Visits",
                              yaxis_title="Readmission Rate (%)")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""<div class="insight-box">
            5+ previous admissions = 2–3× average readmission rate.
            This single variable is visible at the moment of admission.
            </div>""", unsafe_allow_html=True)

        with c2:
            med_stats = (df.groupby(pd.cut(df["num_medications"],
                                            bins=[0,5,10,15,20,80]))["Readmitted30"]
                           .mean().reset_index())
            med_stats.columns = ["Medications","Rate"]
            med_stats["Rate"] *= 100
            med_stats["Medications"] = med_stats["Medications"].astype(str)

            fig2 = px.bar(med_stats, x="Medications", y="Rate",
                          color="Rate",
                          color_continuous_scale=["#4db8a4","#e85d5d"],
                          title="Readmission Rate by Number of Medications")
            fig2.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                               font_color="#e0e2f0", coloraxis_showscale=False,
                               height=380, title_font_size=13)
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("""<div class="insight-box amber">
            Patients on 15+ medications show higher readmission risk —
            complex drug regimens increase post-discharge complication risk.
            </div>""", unsafe_allow_html=True)

    with tab3:
        segments = {
            "Elderly(60+)+MultiAdmit+LongStay":
                df[(df["age_numeric"]>=60)&(df["number_inpatient"]>=2)&
                   (df["time_in_hospital"]>=7)]["Readmitted30"].mean()*100,
            "Elderly(60+)+MultiAdmit":
                df[(df["age_numeric"]>=60)&(df["number_inpatient"]>=2)]["Readmitted30"].mean()*100,
            "MultiAdmit only (>=2)":
                df[df["number_inpatient"]>=2]["Readmitted30"].mean()*100,
            "Long Stay (>=7 days)":
                df[df["time_in_hospital"]>=7]["Readmitted30"].mean()*100,
            "Hospital Average":
                rate_30,
        }
        seg_df = pd.DataFrame(list(segments.items()), columns=["Segment","Rate"])

        fig = px.bar(seg_df.sort_values("Rate"), x="Rate", y="Segment",
                     orientation="h",
                     text=seg_df.sort_values("Rate")["Rate"].round(1).astype(str)+"%",
                     color="Rate",
                     color_continuous_scale=["#4db8a4","#f0a030","#e85d5d"],
                     title="High-Risk Patient Segments vs Hospital Average")
        fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                          font_color="#e0e2f0", coloraxis_showscale=False,
                          height=420, title_font_size=13)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""<div class="insight-box">
        🔴 <b>Elderly + Multiple admissions + Long stay = ~22% readmission rate (2× average)</b><br>
        These patients are identifiable AT ADMISSION using only 3 variables.
        Hospital can assign a case manager on day 1 — no extra clinical tests needed.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — ML MODEL
# ══════════════════════════════════════════════════════════════
elif page == "🤖 ML Model":
    st.markdown("# 🤖 Machine Learning Model")
    st.markdown("*XGBoost + SHAP Explainable AI*")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("🏆 Best Model",   "XGBoost")
    c2.metric("📈 ROC-AUC",      f"{auc_score:.4f}")
    c3.metric("📊 Training Size", f"{len(X_feat):,} patients")

    st.markdown("---")
    tab1, tab2 = st.tabs(["📊 Performance", "🔍 SHAP Insights"])

    with tab1:
        df_ml = df.copy()
        cat_c = df_ml.select_dtypes(include="object").columns.tolist()
        if "readmitted" in cat_c: cat_c.remove("readmitted")
        for c in cat_c:
            df_ml[c] = LabelEncoder().fit_transform(df_ml[c].astype(str))
        df_ml.drop(columns=["readmitted","DiagCategory","LOS_Band"],
                    errors="ignore", inplace=True)
        X2 = df_ml.drop(columns=["Readmitted30"])
        X2 = align_features_for_model(X2, feat_names)
        y2 = df_ml["Readmitted30"]
        _, X_te2, _, y_te2 = train_test_split(X2, y2, test_size=0.2,
                                               random_state=42, stratify=y2)
        X_te2 = align_features_for_model(X_te2, feat_names)
        y_prob2 = model.predict_proba(X_te2)[:,1]
        y_pred2 = model.predict(X_te2)
        fpr, tpr, _ = roc_curve(y_te2, y_prob2)

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, fill="tozeroy",
                                     line=dict(color=ACCENT, width=2.5),
                                     fillcolor="rgba(232,93,93,0.15)",
                                     name=f"AUC={auc_score:.3f}"))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1],
                                     line=dict(color="#555", dash="dash"),
                                     name="Baseline"))
            fig.update_layout(title="ROC Curve", paper_bgcolor="#0f1117",
                              plot_bgcolor="#1a1d27", font_color="#e0e2f0",
                              xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate",
                              legend=dict(bgcolor="#1a1d27"), height=380)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            cm = confusion_matrix(y_te2, y_pred2)
            fig2 = px.imshow(cm, text_auto=True,
                             x=["Not Readmitted","Readmitted"],
                             y=["Not Readmitted","Readmitted"],
                             color_continuous_scale=[[0,"#1a1d27"],[1,ACCENT]],
                             title="Confusion Matrix")
            fig2.update_layout(paper_bgcolor="#0f1117", font_color="#e0e2f0",
                               height=380, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.info("""
        **AUC Note:** 0.69 AUC is consistent with published academic benchmarks
        for this specific dataset. Healthcare administrative data is noisier than
        structured business data — biological and social factors not captured in
        records explain the ceiling. The model still correctly ranks high-risk
        patients above low-risk ones 69% of the time.
        """)

    with tab2:
        st.info("⏳ SHAP computing — ~20 seconds...")
        with st.spinner("Calculating SHAP values..."):
            df_shap = df.copy()
            cat_s = df_shap.select_dtypes(include="object").columns.tolist()
            if "readmitted" in cat_s: cat_s.remove("readmitted")
            for c in cat_s:
                df_shap[c] = LabelEncoder().fit_transform(df_shap[c].astype(str))
            df_shap.drop(columns=["readmitted","DiagCategory","LOS_Band"],
                          errors="ignore", inplace=True)
            X_shap = df_shap.drop(columns=["Readmitted30"])
            X_shap = align_features_for_model(X_shap, feat_names)
            explainer   = shap.TreeExplainer(model)
            shap_vals   = explainer.shap_values(X_shap.iloc[:1000])
            shap_mean   = np.abs(shap_vals).mean(axis=0)
            shap_df     = (pd.DataFrame({"Feature": X_shap.columns, "SHAP": shap_mean})
                            .sort_values("SHAP", ascending=False).head(12))

        fig = px.bar(shap_df.sort_values("SHAP"),
                     x="SHAP", y="Feature", orientation="h",
                     color="SHAP",
                     color_continuous_scale=["#4db8a4","#f0a030","#e85d5d"],
                     title="Top 12 Features — SHAP Importance")
        fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                          font_color="#e0e2f0", coloraxis_showscale=False, height=450)
        st.plotly_chart(fig, use_container_width=True)

        insights = {
            "number_inpatient":  "Each additional prior admission raises 30-day risk by ~3–4%. Frequent flyers need case management from day 1.",
            "time_in_hospital":  "Stays <2 days OR >14 days both signal above-average risk — extremes indicate incomplete recovery.",
            "num_medications":   "Patients on 15+ medications face higher post-discharge complication risk from complex regimens.",
            "num_lab_procedures":"Higher lab procedure counts correlate with case complexity → higher readmission risk.",
            "age_numeric":       "Every 10-year increase in age raises readmission risk by ~1.5 percentage points.",
            "diag_1":            "Primary diagnosis is the strongest clinical predictor — heart and respiratory conditions dominate.",
        }
        for feat in shap_df.head(6)["Feature"]:
            if feat in insights:
                st.markdown(f"""<div class="insight-box">
                <b>{feat}</b> — {insights[feat]}
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 4 — PATIENT RISK
# ══════════════════════════════════════════════════════════════
elif page == "🚨 Patient Risk":
    st.markdown("# 🚨 Patient Risk Dashboard")
    st.markdown("*Filter high-risk patients — ready for clinical team action*")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        diag_filter = st.multiselect("Diagnosis Category",
            options=df["DiagCategory"].unique().tolist(),
            default=df["DiagCategory"].unique().tolist())
    with c2:
        risk_filter = st.multiselect("Risk Flag",
            options=["High Risk","Medium Risk","Low Risk"],
            default=["High Risk"])
    with c3:
        min_risk = st.slider("Minimum Risk Score (%)", 0, 100, 60)

    filtered = df[
        (df["DiagCategory"].isin(diag_filter)) &
        (df["RiskFlag"].isin(risk_filter)) &
        (df["RiskScore"] >= min_risk)
    ]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients Shown",  len(filtered))
    c2.metric("High Risk (>60%)", (filtered["RiskScore"] > 60).sum())
    c3.metric("Very High (>80%)", (filtered["RiskScore"] > 80).sum())
    c4.metric("Avg Risk Score",   f"{filtered['RiskScore'].mean():.1f}%")

    st.markdown("---")

    c1, c2 = st.columns([3,2])
    with c1:
        fig = px.scatter(filtered.sample(min(3000, len(filtered))),
                         x="time_in_hospital", y="RiskScore",
                         color="RiskFlag",
                         color_discrete_map={
                             "High Risk": ACCENT,
                             "Medium Risk": AMBER,
                             "Low Risk": SAFE,
                         },
                         hover_data=["age","DiagCategory","number_inpatient"],
                         size="number_inpatient", size_max=14,
                         title="Risk Score vs Length of Stay (size = Prior Admissions)")
        fig.add_hline(y=60, line_dash="dash", line_color="white",
                      annotation_text="High Risk threshold")
        fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                          font_color="#e0e2f0", height=400,
                          legend=dict(bgcolor="#1a1d27"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.histogram(filtered, x="RiskScore", nbins=30,
                             color_discrete_sequence=[ACCENT],
                             title="Risk Score Distribution")
        fig2.add_vline(x=60, line_dash="dash", line_color="yellow",
                       annotation_text="60%")
        fig2.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                           font_color="#e0e2f0", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("**High Risk Patient List**")
    display_cols = ["age","DiagCategory","time_in_hospital",
                    "number_inpatient","num_medications",
                    "RiskScore","RiskFlag","readmitted"]
    table = (filtered[display_cols]
               .sort_values("RiskScore", ascending=False)
               .head(100)
               .reset_index(drop=True))
    st.dataframe(
        table.style
             .background_gradient(subset=["RiskScore"], cmap="RdYlGn_r")
             .format({"RiskScore": "{:.1f}%"}),
        use_container_width=True,
        height=400,
    )
    csv = table.to_csv(index=False)
    st.download_button("⬇️ Download High Risk Patient List",
                        csv, "high_risk_patients.csv", "text/csv")


# ══════════════════════════════════════════════════════════════
#  PAGE 5 — ROI CALCULATOR
# ══════════════════════════════════════════════════════════════
elif page == "💰 ROI Calculator":
    st.markdown("# 💰 ROI & Cost Impact Calculator")
    st.markdown("*Quantify the business value of early readmission prediction*")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Adjust Assumptions**")
        cost_per_readmit = st.slider("Cost per readmission ($)", 5000, 30000, 15000, 1000)
        reduction_pct    = st.slider("Model reduces readmissions by (%)", 5, 50, 20)
        total_patients   = st.slider("Annual patient volume", 10000, 200000, int(len(df)), 5000)

    readmit_count  = int(total_patients * (rate_30/100))
    total_cost     = readmit_count * cost_per_readmit
    saved_count    = int(readmit_count * reduction_pct/100)
    total_savings  = saved_count * cost_per_readmit
    new_cost       = total_cost - total_savings

    with c2:
        st.markdown("**Impact Summary**")
        st.metric("Annual Readmissions",     f"{readmit_count:,}")
        st.metric("Total Annual Cost",        f"${total_cost:,.0f}")
        st.metric("Readmissions Prevented",   f"{saved_count:,}")
        st.metric("💰 Annual Savings",         f"${total_savings:,.0f}",
                   f"↓ {reduction_pct}% reduction")

    st.markdown("---")
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute","relative","total"],
        x=["Current Annual Cost", f"−{reduction_pct}% Reduction", "New Annual Cost"],
        y=[total_cost, -total_savings, 0],
        text=[f"${total_cost/1e6:.1f}M", f"-${total_savings/1e6:.1f}M",
              f"${new_cost/1e6:.1f}M"],
        textposition="outside",
        connector={"line":{"color":"#2e3148"}},
        decreasing={"marker":{"color":SAFE}},
        increasing={"marker":{"color":ACCENT}},
        totals={"marker":{"color":AMBER}},
    ))
    fig.update_layout(
        title=f"Annual Readmission Cost — ${total_savings/1e6:.1f}M Potential Savings",
        paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
        font_color="#e0e2f0", height=420, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    monthly = total_savings / 12
    weekly  = total_savings / 52
    st.markdown(f"""
    <div style="background:#1a1d27; border:2px solid #4db8a4; border-radius:12px;
                padding:24px; text-align:center;">
        <div style="color:#888aaa; font-size:0.85rem; text-transform:uppercase; letter-spacing:2px;">
            This model saves
        </div>
        <div style="color:#4db8a4; font-size:3rem; font-weight:900; margin:12px 0;">
            ${total_savings:,.0f}
        </div>
        <div style="color:#c8cad8; font-size:1rem;">
            per year &nbsp;·&nbsp; <b>${monthly:,.0f}</b> per month
            &nbsp;·&nbsp; <b>${weekly:,.0f}</b> per week
        </div>
        <div style="color:#888aaa; font-size:0.85rem; margin-top:12px;">
            Based on {reduction_pct}% readmission reduction · ${cost_per_readmit:,} per episode
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Note: Cost estimates are approximations based on CMS average readmission costs. Actual savings depend on clinical implementation and patient population.")
