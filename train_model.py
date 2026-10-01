"""
train_model.py
--------------
Loads housing_price.csv, analyses the data, trains three regression models,
evaluates them, selects the best one, and saves it to model/house_price_model.pkl.

Run:
    python train_model.py
"""

import os
import warnings
import joblib
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose          import ColumnTransformer
from sklearn.pipeline         import Pipeline
from sklearn.preprocessing    import OneHotEncoder, StandardScaler
from sklearn.linear_model     import LinearRegression
from sklearn.tree             import DecisionTreeRegressor
from sklearn.ensemble         import RandomForestRegressor
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── 1. Load data ──────────────────────────────────────────────────────────────
DATA_PATH  = "housing_price.csv"
MODEL_DIR  = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "house_price_model.pkl")

print("=" * 60)
print("  HOUSE PRICE PREDICTION — MODEL TRAINING")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

# ── 2. Basic analysis ─────────────────────────────────────────────────────────
print(f"\nDataset loaded: {DATA_PATH}")
print(f"   Rows       : {df.shape[0]}")
print(f"   Columns    : {df.shape[1]}")
print(f"\n   Data types:\n{df.dtypes.to_string()}")
print(f"\n   Missing values:\n{df.isnull().sum().to_string()}")
print(f"\n   Duplicate rows: {df.duplicated().sum()}")
print(f"\n   Statistical summary:\n{df.describe().round(2).to_string()}")

# ── 3. Handle missing values (none expected, but be safe) ─────────────────────
df.dropna(inplace=True)

# ── 4. Features and target ────────────────────────────────────────────────────
NUMERIC_FEATURES     = ["area_sqft", "bedrooms", "bathrooms",
                         "stories",  "parking",  "age_years"]
CATEGORICAL_FEATURES = ["location"]
TARGET               = "price"

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET]

# ── 5. Train/test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\n   Train size : {len(X_train)}")
print(f"   Test  size : {len(X_test)}")

# ── 6. Preprocessing pipeline ────────────────────────────────────────────────
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer,     NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES),
])

# ── 7. Define models ──────────────────────────────────────────────────────────
models = {
    "Linear Regression":  LinearRegression(),
    "Decision Tree":      DecisionTreeRegressor(random_state=42),
    "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42),
}

# ── 8. Train and evaluate ─────────────────────────────────────────────────────
results = {}

print("\n" + "=" * 60)
print("  MODEL EVALUATION RESULTS")
print("=" * 60)
print(f"{'Model':<22} {'MAE':>12} {'RMSE':>14} {'R2 Score':>10}")
print("-" * 60)

best_model_name = None
best_r2         = -np.inf
best_pipeline   = None

for name, estimator in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model",        estimator),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "pipeline": pipeline}
    print(f"{name:<22} {mae:>12,.0f} {rmse:>14,.0f} {r2:>10.4f}")

    if r2 > best_r2:
        best_r2         = r2
        best_model_name = name
        best_pipeline   = pipeline

print("-" * 60)
print(f"\n[BEST] Model selected: {best_model_name}  (R2 = {best_r2:.4f})")

# ── 9. Save best model ────────────────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(best_pipeline, MODEL_PATH)
print(f"\n[SAVED] Model saved to '{MODEL_PATH}'")

# ── 10. Save evaluation results for use in Streamlit ─────────────────────────
eval_rows = [
    {"Model": name, "MAE": v["MAE"], "RMSE": v["RMSE"], "R2": v["R2"]}
    for name, v in results.items()
]
eval_df = pd.DataFrame(eval_rows)
eval_df.to_csv(os.path.join(MODEL_DIR, "eval_results.csv"), index=False)

# Save selected model name
with open(os.path.join(MODEL_DIR, "best_model.txt"), "w") as f:
    f.write(best_model_name)

print("\n[DONE] Training complete.")
print("=" * 60)
