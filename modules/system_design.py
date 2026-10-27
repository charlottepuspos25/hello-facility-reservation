"""Module 5 - SAD integration."""
import os

import pandas as pd
import streamlit as st

FR = pd.DataFrame([
    ("FR-01", "Upload & validate reservation CSV; show schema errors, nulls and duplicates.", "Dataset page", "Part I validation / data quality (435 rows -> 15 duplicates)"),
    ("FR-02", "Require confirmation notice for 'Last Minute' bookings (0-2 days lead time).", "Prediction/Dashboard actions", "EDA Chart 3 - lead time vs outcome"),
    ("FR-03", "Flag reservations from high not-completed user categories for administrator review.", "Dashboard queue", "EDA Chart 2 - outcome by user category"),
    ("FR-04", "Show No-Show Risk score, tier, model version and timestamp at request entry.", "Prediction page", "Midterm Decision Tree (Part IV)"),
    ("FR-05", "Let the Facility Administrator record a final decision with a mandatory reason (audit log).", "Dashboard / Prediction", "Task 12(a): high risk must not auto-reject"),
], columns=["ID", "Functional requirement", "Implemented in", "Evidence"])

NFR = pd.DataFrame([
    ("NFR-01", "Performance", "Saved pipeline is loaded once (st.cache_resource); no retraining on refresh."),
    ("NFR-02", "Security & privacy", "Only synthetic/authorized data; no credentials in repo; role-restricted decisions."),
    ("NFR-03", "Usability & transparency", "Plain-language errors, visible limitations, model version, and human-review notice."),
], columns=["ID", "Category", "Requirement"])

ACTORS = pd.DataFrame([
    ("Requester (Student/Faculty/External)", "Submit request, view own risk indication", "No"),
    ("Facility Administrator", "View dashboard, review queue, record final decision", "Yes"),
    ("ML Decision Support Engine (system actor)", "Compute risk score on request", "No (advisory only)"),
], columns=["Actor", "Allowed actions", "Can record final decision?"])


def render_system_design():
    st.subheader("Functional requirements (evidence-based)")
    st.dataframe(FR, use_container_width=True, hide_index=True)
    st.subheader("Non-functional requirements")
    st.dataframe(NFR, use_container_width=True, hide_index=True)
    st.subheader("Actors and permissions")
    st.dataframe(ACTORS, use_container_width=True, hide_index=True)
    st.subheader("Updated use case diagram")
    if os.path.exists("assets/use_case_diagram.png"):
        st.image("assets/use_case_diagram.png", use_container_width=True)
    st.subheader("Decision workflow / analytical process flow")
    if os.path.exists("assets/decision_workflow.png"):
        st.image("assets/decision_workflow.png", use_container_width=True)
    st.info("Predictions are decision-support outputs. Only an authorized Facility Administrator records the final "
            "decision, and a high risk score alone never rejects a reservation.")
