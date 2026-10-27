# Campus Facility Reservation Monitoring (Streamlit)

Deployment of the Midterm data-mining output (Section A): predicts whether a reservation will be
**Completed** or **Not Completed** (cancelled / no-show) using the midterm Decision Tree (max_depth=4).

## Modules
Home · Dataset (upload/validate) · Data Analysis (EDA, 5 interactive charts) · Prediction · Dashboard · System Design

## Files
- `app.py` – navigation and page routing
- `modules/` – data_processing, exploratory_analysis, prediction, dashboard, system_design
- `models/trained_model.joblib` – fitted preprocessing + model pipeline (`model_metadata.json` holds metrics)
- `train_model.py` – recreates the saved pipeline from `data/clean_facility_reservations.csv`
- `assets/` – use case diagram and decision workflow

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
## Deploy
Push to GitHub → https://share.streamlit.io → New app → pick repo/branch → main file `app.py`.

## Leakage prevention
Predictors: Facility_Type, Facility_Name, User_Category, Lead_Time_Days, Participants.
Excluded: Usage_Status, Actual_Attendance, Cancellation_Days_Before (post-event).

## Notes
- Synthetic data only. Predictions are decision-support; a human administrator makes the final decision.
- Midterm test accuracy: 60.7% (same split/seed as the notebook). Differences from midterm: model saved as a
  scikit-learn Pipeline; wireframe placeholder figures replaced by computed values.
- `scikit-learn` is pinned to the version used to save the model; keep it consistent.
