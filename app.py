import streamlit as st

from modules.dashboard import render_dashboard
from modules.data_processing import (TARGET, clean_data, quality_summary, read_uploaded_csv, validate_schema)
from modules.exploratory_analysis import render_eda
from modules.prediction import load_metadata, render_prediction
from modules.system_design import render_system_design

st.set_page_config(page_title="Campus Facility Reservation - Risk Monitoring", layout="wide")

DEFAULT_CSV = "data/clean_facility_reservations.csv"


@st.cache_data
def load_default():
    import pandas as pd
    return clean_data(pd.read_csv(DEFAULT_CSV))


def set_active(df, name):
    st.session_state["data"] = df
    st.session_state["data_name"] = name
    st.session_state["options"] = {c: sorted(df[c].dropna().unique().tolist())
                                  for c in ["Facility_Type", "Facility_Name", "User_Category"]}


if "data" not in st.session_state:
    set_active(load_default(), "Bundled synthetic midterm dataset")

menu = st.sidebar.selectbox("Module", ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"])
role = st.sidebar.radio("Role (demo login)", ["Requester", "Facility Administrator"])
st.sidebar.caption(f"Active data: {st.session_state['data_name']}")
df = st.session_state["data"]

st.title("Campus Facility Reservation Monitoring")

if menu == "Home":
    meta = load_metadata()
    st.write("Decision-support system that predicts whether a facility reservation will be **Completed** or "
             "**Not Completed** (cancelled / no-show), reusing the Midterm Decision Tree model.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Records loaded", len(df))
    c2.metric("Model", meta["model_version"])
    c3.metric("Test accuracy", f"{meta['accuracy']*100:.1f}%")
    st.markdown("**Modules:** Dataset (upload/validate) → Data Analysis (EDA) → Prediction → Dashboard → System Design")
    st.warning("Predictions are decision-support outputs, not automatic authorization to reject a reservation. "
               "This app uses synthetic data only.")
    st.markdown("**Differences from the midterm:** the model is the same Decision Tree (max_depth=4, 80/20 stratified split, "
                "same 5 predictors); it is now saved as one scikit-learn Pipeline (one-hot encoding + tree) instead of "
                "retrained in the notebook. The midterm dashboard wireframe used placeholder numbers; this app shows the "
                "real values computed from the data.")

elif menu == "Dataset":
    st.subheader("Upload and validate CSV")
    up = st.file_uploader("Upload reservations CSV", type="csv")
    if up is not None:
        raw, err = read_uploaded_csv(up)
        if err:
            st.error(err)
        else:
            ok, msgs = validate_schema(raw)
            if not ok:
                for m in msgs:
                    st.error(m)
                st.caption("Required: Reservation_ID, Facility_Type, Facility_Name, User_Category, Lead_Time_Days, "
                           "Participants, and Usage_Status (or Reservation_Outcome).")
            else:
                q = quality_summary(raw)
                st.success("Schema valid.")
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows", q["rows"]); c2.metric("Columns", q["cols"]); c3.metric("Duplicate rows", q["duplicates"])
                st.write("Preview"); st.dataframe(raw.head(20), use_container_width=True)
                st.write("Null summary"); st.dataframe(q["nulls"], use_container_width=True)
                if st.button("Clean and use this dataset"):
                    set_active(clean_data(raw), up.name)
                    st.success("Dataset cleaned and activated for all modules.")
    else:
        st.info("No file uploaded - using the bundled dataset. Preview:")
        q = quality_summary(df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", q["rows"]); c2.metric("Columns", q["cols"]); c3.metric("Duplicate rows", q["duplicates"])
        st.dataframe(df.head(20), use_container_width=True)
        st.dataframe(q["nulls"], use_container_width=True)
    if st.button("Reset to bundled dataset"):
        set_active(load_default(), "Bundled synthetic midterm dataset")
        st.rerun()

elif menu == "Data Analysis":
    render_eda(df)
elif menu == "Prediction":
    render_prediction(role)
elif menu == "Dashboard":
    render_dashboard(df, role)
else:
    render_system_design()
