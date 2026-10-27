"""Module 4 - Decision-support dashboard: KPIs, risk/review queue, interpretation, follow-up."""
import pandas as pd
import streamlit as st

from modules.data_processing import PREDICTORS, TARGET
from modules.prediction import load_model, recommended_action, risk_scores, risk_tier


def score_data(df):
    model = load_model()
    out = df.copy()
    out["Risk_Score"] = risk_scores(model, out[PREDICTORS]).round(3)
    out["Risk_Tier"] = out["Risk_Score"].apply(risk_tier)
    out["Recommended_Action"] = out["Risk_Tier"].apply(recommended_action)
    return out


def render_dashboard(df, role):
    scored = score_data(df)
    st.caption("Scores for existing records are in-sample (the model saw most of them in training); "
               "use this page to demonstrate the workflow, not as proof of accuracy.")

    # ---- KPIs
    total = len(scored)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total reservations", total)
    if TARGET in scored.columns:
        comp = (scored[TARGET] == "Completed").mean() * 100
        k2.metric("Completion rate", f"{comp:.1f}%")
        k3.metric("Cancel / No-show rate", f"{100-comp:.1f}%")
    k4.metric("High-risk reservations", int((scored["Risk_Tier"] == "HIGH").sum()))
    c1, c2 = st.columns(2)
    c1.metric("Most-booked venue", scored["Facility_Name"].value_counts().idxmax())
    if TARGET in scored.columns:
        rate = scored.assign(nc=(scored[TARGET] == "Not Completed")).groupby("User_Category")["nc"].mean()
        c2.metric("Highest not-completed user group", f"{rate.idxmax()} ({rate.max()*100:.1f}%)")

    # ---- Utilization
    st.subheader("Facility risk and utilization")
    util = scored.groupby("Facility_Name").agg(Reservations=("Reservation_ID", "count"),
                                               Avg_Risk=("Risk_Score", "mean"),
                                               High_Risk=("Risk_Tier", lambda s: int((s == "HIGH").sum())))
    util["Avg_Risk"] = util["Avg_Risk"].round(3)
    st.dataframe(util.sort_values("Reservations", ascending=False), use_container_width=True)

    # ---- Queue
    st.subheader("Risk / review queue")
    tiers = st.multiselect("Risk tiers to show", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM"])
    queue = scored[scored["Risk_Tier"].isin(tiers)].sort_values("Risk_Score", ascending=False)
    cols = ["Reservation_ID", "Facility_Type", "Facility_Name", "User_Category", "Lead_Time_Days",
            "Participants", "Risk_Score", "Risk_Tier", "Recommended_Action"]
    st.dataframe(queue[cols].head(100), use_container_width=True, hide_index=True)
    st.caption(f"Showing top {min(len(queue), 100)} of {len(queue)} records.")

    # ---- Interpretation
    st.subheader("Interpretation")
    n_high, n_med = int((scored.Risk_Tier == "HIGH").sum()), int((scored.Risk_Tier == "MEDIUM").sum())
    st.markdown(
        f"- **{n_high}** reservations are HIGH risk and **{n_med}** MEDIUM risk (thresholds: HIGH ≥ 0.60, MEDIUM ≥ 0.40).\n"
        "- A high score means *similar past requests were often cancelled/no-show*; it does not prove this request will fail.\n"
        "- Recommended follow-up: confirm with requester, request a deposit/check-in, or offer the slot to a waitlist - "
        "**never auto-reject**.")

    # ---- Human decision
    st.subheader("Human review: record decision")
    if role != "Facility Administrator":
        st.info("Switch the role in the sidebar to 'Facility Administrator' to record decisions.")
    elif queue.empty:
        st.write("Queue is empty.")
    else:
        with st.form("queue_decision"):
            rid = st.selectbox("Reservation", queue["Reservation_ID"].head(100))
            decision = st.selectbox("Decision", ["Approve", "Approve with confirmation required",
                                                 "Flag for further review", "Reject (with documented reason)"])
            reason = st.text_input("Reason / notes (required)")
            if st.form_submit_button("Record decision"):
                if not reason.strip():
                    st.error("A reason is required.")
                else:
                    row = queue[queue.Reservation_ID == rid].iloc[0]
                    st.session_state.setdefault("decision_log", []).append({
                        "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "source": "Review queue",
                        "Reservation_ID": rid, "risk_tier": row.Risk_Tier, "risk_score": row.Risk_Score,
                        "decision": decision, "reason": reason, "decided_by": role})
                    st.success(f"Decision recorded for {rid}.")

    log = st.session_state.get("decision_log", [])
    if log:
        st.subheader("Decision audit log (this session)")
        log_df = pd.DataFrame(log)
        st.dataframe(log_df, use_container_width=True, hide_index=True)
        st.download_button("Download log (CSV)", log_df.to_csv(index=False), "decision_log.csv", "text/csv")
