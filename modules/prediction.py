"""Module 3 - Predictive modeling: load saved pipeline, validate inputs, predict."""
import json
from datetime import datetime

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_processing import PREDICTORS

MODEL_PATH = "models/trained_model.joblib"
META_PATH = "models/model_metadata.json"
POSITIVE_RISK_CLASS = "Not Completed"

LIMITATIONS = [
    "Trained on a small synthetic dataset (420 rows); the midterm test accuracy was only about 61%.",
    "Uses only 5 submission-time predictors; no holidays, exam weeks, weather or user history.",
    "Post-event fields (Usage_Status, Actual_Attendance, Cancellation_Days_Before) are excluded to prevent leakage.",
    "A risk score is a decision-support signal. It must never automatically reject a reservation.",
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(META_PATH) as f:
        return json.load(f)


def build_input_df(facility_type, facility_name, user_category, lead_time_days, participants):
    """One-row DataFrame with the exact training feature names/types."""
    return pd.DataFrame([{
        "Facility_Type": str(facility_type), "Facility_Name": str(facility_name),
        "User_Category": str(user_category),
        "Lead_Time_Days": int(lead_time_days), "Participants": int(participants),
    }])[PREDICTORS]


def validate_inputs(lead_time_days, participants):
    errs = []
    if lead_time_days is None or lead_time_days < 0:
        errs.append("Lead time must be 0 or more days (event cannot be before booking).")
    if participants is None or participants <= 0:
        errs.append("Participants must be greater than 0.")
    return errs


def risk_tier(p):
    """Policy thresholds (adjustable by administrators)."""
    if p >= 0.60:
        return "HIGH"
    if p >= 0.40:
        return "MEDIUM"
    return "LOW"


def recommended_action(tier):
    return {"HIGH": "Require confirmation / deposit and verify with requester before approval.",
            "MEDIUM": "Flag for administrator review; send reminder before event.",
            "LOW": "Standard approval workflow."}[tier]


def risk_scores(model, X):
    idx = list(model.classes_).index(POSITIVE_RISK_CLASS)
    return model.predict_proba(X)[:, idx]


def render_prediction(role):
    model, meta = load_model(), load_metadata()
    st.caption(f"Model version: **{meta['model_version']}** | trained {meta['trained_on']} | "
               f"target: {meta['target']}")

    st.subheader("New reservation request")
    options = st.session_state.get("options", {})
    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        ftype = c1.selectbox("Facility type", options.get("Facility_Type", []))
        fname = c2.selectbox("Facility name", options.get("Facility_Name", []))
        ucat = c1.selectbox("User category", options.get("User_Category", []))
        lead = c2.number_input("Lead time (days between booking and event)", value=7, step=1)
        parts = c1.number_input("Expected participants", value=50, step=1)
        submitted = c2.form_submit_button("Predict risk", use_container_width=True)

    if submitted:
        errs = validate_inputs(lead, parts)
        if errs:
            for e in errs:
                st.error(e)
        else:
            X = build_input_df(ftype, fname, ucat, lead, parts)
            label = model.predict(X)[0]
            p_nc = float(risk_scores(model, X)[0])
            tier = risk_tier(p_nc)
            st.session_state["last_prediction"] = {
                "input": X.iloc[0].to_dict(), "label": label, "p_nc": p_nc, "tier": tier,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "version": meta["model_version"]}

    last = st.session_state.get("last_prediction")
    if last:
        st.subheader("Prediction result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted outcome", last["label"])
        c2.metric("Not-Completed probability", f"{last['p_nc']*100:.1f}%")
        c3.metric("Risk tier", last["tier"])
        st.write(f"**Recommended human follow-up:** {recommended_action(last['tier'])}")
        st.caption(f"Prediction timestamp: {last['time']} | Model: {last['version']} | "
                   "Human review required - this is not an automatic approval or rejection.")
        st.json(last["input"], expanded=False)

        if role == "Facility Administrator":
            with st.form("decision_form"):
                decision = st.selectbox("Final administrator decision",
                                        ["Approve", "Approve with confirmation required", "Flag for further review",
                                         "Reject (with documented reason)"])
                reason = st.text_input("Reason / notes (required)")
                if st.form_submit_button("Record decision"):
                    if not reason.strip():
                        st.error("A reason is required to record a decision.")
                    else:
                        st.session_state.setdefault("decision_log", []).append({
                            "time": last["time"], "source": "Manual prediction", **last["input"],
                            "risk_tier": last["tier"], "risk_score": round(last["p_nc"], 3),
                            "decision": decision, "reason": reason, "decided_by": role})
                        st.success("Decision recorded in the audit log (Dashboard page).")
        else:
            st.info("Only the Facility Administrator role can record final decisions.")

    st.divider()
    st.subheader("Model metrics (held-out 20% test set from the midterm)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{meta['accuracy']*100:.1f}%")
    m2.metric("Precision (Completed)", f"{meta['precision']:.3f}")
    m3.metric("Recall (Completed)", f"{meta['recall']:.3f}")
    m4.metric("F1 (Completed)", f"{meta['f1']:.3f}")
    cm = pd.DataFrame(meta["confusion_matrix"], index=["Actual: Completed", "Actual: Not Completed"],
                      columns=["Pred: Completed", "Pred: Not Completed"])
    st.dataframe(cm, use_container_width=True)
    fi = pd.Series(meta["feature_importances"]).sort_values(ascending=False).head(8).reset_index()
    fi.columns = ["Feature", "Importance"]
    st.plotly_chart(px.bar(fi, x="Importance", y="Feature", orientation="h", title="Top predictors"),
                    use_container_width=True)

    st.subheader("Limitations")
    for l in LIMITATIONS:
        st.markdown(f"- {l}")
