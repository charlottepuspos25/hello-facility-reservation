"""Data loading, schema validation, quality summary and cleaning.
Mirrors the cleaning + feature-engineering steps of the Midterm notebook."""
import numpy as np
import pandas as pd

PREDICTORS = ["Facility_Type", "Facility_Name", "User_Category", "Lead_Time_Days", "Participants"]
CATEGORICAL = ["Facility_Type", "Facility_Name", "User_Category"]
NUMERIC = ["Lead_Time_Days", "Participants"]
TARGET = "Reservation_Outcome"
# Columns an uploaded CSV must contain (plus Usage_Status OR Reservation_Outcome)
REQUIRED_COLUMNS = ["Reservation_ID"] + PREDICTORS
LEAD_ORDER = ["Last Minute", "Short Lead", "Medium Lead", "Early Booking"]

FACILITY_MAP = {
    "auditorium": "Auditorium", "conference room": "Conference Room", "conf room": "Conference Room",
    "laboratory": "Laboratory", "lab": "Laboratory", "gymnasium": "Gymnasium",
}
USER_MAP = {
    "student org": "Student Org", "student": "Student Org", "faculty": "Faculty",
    "external": "External", "admin": "Admin",
}


def read_uploaded_csv(file):
    """Return (df, error_message). Never raises for bad user files."""
    try:
        df = pd.read_csv(file)
    except pd.errors.EmptyDataError:
        return None, "The file is empty (no columns found)."
    except pd.errors.ParserError as e:
        return None, f"The file could not be parsed as CSV: {e}"
    except UnicodeDecodeError:
        return None, "The file is not a valid UTF-8 text CSV."
    except Exception as e:  # noqa
        return None, f"Could not read file: {e}"
    if df.empty:
        return None, "The CSV has headers but no data rows."
    return df, None


def validate_schema(df):
    """Return (is_valid, list_of_messages)."""
    msgs = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        msgs.append(f"Missing required column(s): {', '.join(missing)}")
    if "Usage_Status" not in df.columns and TARGET not in df.columns:
        msgs.append("Need either 'Usage_Status' or 'Reservation_Outcome' column (for EDA/KPIs).")
    return (len(msgs) == 0), msgs


def quality_summary(df):
    nulls = df.isna().sum().rename("Null Count").to_frame()
    nulls["Null %"] = (nulls["Null Count"] / len(df) * 100).round(2)
    return {"rows": len(df), "cols": df.shape[1],
            "duplicates": int(df.duplicated().sum()), "nulls": nulls}


def lead_time_category(days):
    if days <= 2:
        return "Last Minute"
    if days <= 7:
        return "Short Lead"
    if days <= 14:
        return "Medium Lead"
    return "Early Booking"


def clean_data(df):
    """Apply the Midterm cleaning rules. Returns a new DataFrame."""
    df = df.drop_duplicates().reset_index(drop=True).copy()

    df["Facility_Type"] = df["Facility_Type"].astype(str).str.strip().str.lower().map(FACILITY_MAP)
    df["User_Category"] = df["User_Category"].astype(str).str.strip().str.lower().map(USER_MAP)
    if df["Facility_Type"].notna().any():
        df["Facility_Type"] = df["Facility_Type"].fillna(df["Facility_Type"].mode()[0])
    if df["User_Category"].notna().any():
        df["User_Category"] = df["User_Category"].fillna(df["User_Category"].mode()[0])
    df["Facility_Name"] = df["Facility_Name"].astype(str).str.strip()

    df["Participants"] = pd.to_numeric(df["Participants"], errors="coerce")
    valid = df.loc[df["Participants"] > 0, "Participants"]
    median_p = valid.median() if len(valid) else 1
    df.loc[~(df["Participants"] > 0), "Participants"] = median_p
    df["Participants"] = df["Participants"].astype(int)

    df["Lead_Time_Days"] = pd.to_numeric(df["Lead_Time_Days"], errors="coerce").fillna(0).clip(lower=0).astype(int)

    if "Actual_Attendance" in df.columns:
        df["Actual_Attendance"] = pd.to_numeric(df["Actual_Attendance"], errors="coerce").fillna(0).clip(lower=0)
        df["Attendance_Ratio"] = np.where(df["Participants"] > 0,
                                          (df["Actual_Attendance"] / df["Participants"]).round(4), 0.0)

    df["Lead_Time_Category"] = df["Lead_Time_Days"].apply(lead_time_category)
    if "Usage_Status" in df.columns:
        df[TARGET] = df["Usage_Status"].apply(lambda x: "Completed" if str(x).strip() == "Used" else "Not Completed")
    return df
