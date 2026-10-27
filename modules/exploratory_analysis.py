"""Module 2 - Exploratory Data Analysis (interactive Plotly charts)."""
import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_processing import LEAD_ORDER, TARGET

COLORS = {"Completed": "#2ca02c", "Not Completed": "#d62728"}


def render_eda(df):
    st.subheader("Descriptive statistics")
    num_cols = [c for c in ["Lead_Time_Days", "Participants", "Actual_Attendance", "Attendance_Ratio"] if c in df.columns]
    st.dataframe(df[num_cols].describe().round(2), use_container_width=True)

    if TARGET not in df.columns:
        st.warning("No outcome column available - class distribution and outcome charts are skipped.")
        return

    st.subheader("Class distribution (target)")
    dist = df[TARGET].value_counts().reset_index()
    dist.columns = ["Outcome", "Count"]
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.pie(dist, names="Outcome", values="Count", color="Outcome", color_discrete_map=COLORS,
                           title="Completed vs Not Completed"), use_container_width=True)
    c2.dataframe(dist, use_container_width=True, hide_index=True)
    c2.caption("Not Completed = Cancelled + No-Show. Mild imbalance: accuracy alone can mislead, so check recall/precision.")

    st.subheader("Interactive charts")
    # Chart 1 - volume by facility type
    fac = df["Facility_Type"].value_counts().reset_index()
    fac.columns = ["Facility_Type", "Reservations"]
    st.plotly_chart(px.bar(fac, x="Facility_Type", y="Reservations", text="Reservations",
                           title="Chart 1: Reservation volume by facility type"), use_container_width=True)

    # Chart 2 - outcome % by user category (filterable)
    group = st.selectbox("Chart 2 - group outcome by:", ["User_Category", "Facility_Type", "Facility_Name"])
    pct = (pd.crosstab(df[group], df[TARGET], normalize="index") * 100).round(1).reset_index()
    pct = pct.melt(id_vars=group, var_name="Outcome", value_name="Percent")
    st.plotly_chart(px.bar(pct, x=group, y="Percent", color="Outcome", barmode="stack", color_discrete_map=COLORS,
                           title=f"Chart 2: Outcome % by {group}"), use_container_width=True)

    # Chart 3 - outcome by lead-time category
    lt = (pd.crosstab(df["Lead_Time_Category"], df[TARGET], normalize="index") * 100).round(1)
    lt = lt.reindex([c for c in LEAD_ORDER if c in lt.index]).reset_index().melt(
        id_vars="Lead_Time_Category", var_name="Outcome", value_name="Percent")
    st.plotly_chart(px.bar(lt, x="Lead_Time_Category", y="Percent", color="Outcome", barmode="stack",
                           color_discrete_map=COLORS, title="Chart 3: Outcome % by lead-time category"),
                    use_container_width=True)

    # Chart 4 - participants distribution
    st.plotly_chart(px.histogram(df, x="Participants", color=TARGET, nbins=25, barmode="overlay", opacity=0.7,
                                 color_discrete_map=COLORS, title="Chart 4: Expected participants by outcome"),
                    use_container_width=True)

    # Chart 5 - heatmap of not-completed rate
    heat = df.assign(NC=(df[TARGET] == "Not Completed").astype(int)).pivot_table(
        index="Facility_Name", columns="User_Category", values="NC", aggfunc="mean") * 100
    st.plotly_chart(px.imshow(heat.round(1), text_auto=True, aspect="auto", color_continuous_scale="Reds",
                              title="Chart 5: Not-Completed rate (%) - venue x user category"),
                    use_container_width=True)

    st.info("Interpretation: use these charts as evidence for the functional requirements on the System Design page. "
            "Differences between small groups may be noise (see group sizes before acting).")
