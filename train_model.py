"""Re-creates the Midterm model (same predictors, split, DecisionTree settings) and
saves ONE fitted preprocessing+model pipeline with joblib. Run once, not on every app refresh:
    python train_model.py
"""
import json
from datetime import datetime

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from modules.data_processing import CATEGORICAL, NUMERIC, PREDICTORS, TARGET

df = pd.read_csv("data/clean_facility_reservations.csv")
X, y = df[PREDICTORS], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Same as pd.get_dummies(drop_first=True) in the midterm; unknown categories are ignored at inference.
pre = ColumnTransformer([
    ("num", "passthrough", NUMERIC),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CATEGORICAL),
])
pipe = Pipeline([("prep", pre), ("model", DecisionTreeClassifier(max_depth=4, criterion="gini", random_state=42))])
pipe.fit(X_train, y_train)

pred = pipe.predict(X_test)
meta = {
    "model_version": "v1.0-midterm-DT-depth4",
    "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "n_train": len(X_train), "n_test": len(X_test),
    "predictors": PREDICTORS, "target": TARGET, "classes": list(pipe.classes_),
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred, pos_label="Completed"),
    "recall": recall_score(y_test, pred, pos_label="Completed"),
    "f1": f1_score(y_test, pred, pos_label="Completed"),
    "confusion_matrix": confusion_matrix(y_test, pred, labels=["Completed", "Not Completed"]).tolist(),
    "feature_importances": dict(zip(pipe.named_steps["prep"].get_feature_names_out().tolist(),
                                    pipe.named_steps["model"].feature_importances_.round(4).tolist())),
}
joblib.dump(pipe, "models/trained_model.joblib")
json.dump(meta, open("models/model_metadata.json", "w"), indent=2)
print({k: v for k, v in meta.items() if k not in ("feature_importances",)})
