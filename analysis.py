import os
import json
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe for server use
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# ─── Directories ────────────────────────────────────────────────────────────
os.makedirs("static/charts", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ─── Load dataset ───────────────────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\user\Downloads\50_Startups.csv")
df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nColumn information:")
df.info()
print("\nDescriptive statistics:")
print(df.describe(include="all"))
print("Missing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nState counts:")
print(df["State"].value_counts(dropna=False))

# ─── EDA Plots ──────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
sns.histplot(df["Profit"], kde=True)
plt.title("Distribution of Startup Profit")
plt.xlabel("Profit"); plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("static/charts/profit_distribution.png", dpi=120)
plt.close()

numeric_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("static/charts/correlation_matrix.png", dpi=120)
plt.close()

for col in ["R&D Spend", "Administration", "Marketing Spend"]:
    plt.figure(figsize=(7, 4))
    sns.regplot(data=df, x=col, y="Profit", ci=None, scatter_kws={"alpha": 0.8})
    plt.title(f"{col} vs. Profit")
    plt.tight_layout()
    safe_name = col.replace("&", "and").replace(" ", "_").lower()
    plt.savefig(f"static/charts/scatter_{safe_name}.png", dpi=120)
    plt.close()

# ─── Feature Definitions ────────────────────────────────────────────────────
target              = "Profit"
numerical_features  = ["R&D Spend", "Administration", "Marketing Spend"]
categorical_features= ["State"]
all_features        = numerical_features + categorical_features

X = df[all_features].copy()
y = df[target].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)
print("\nTraining observations:", len(X_train))
print("Testing observations:", len(X_test))

# ─── Preprocessing ──────────────────────────────────────────────────────────
numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Fit preprocessor on training data only
X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc  = preprocessor.transform(X_test)

# Reconstructed feature names after encoding
enc_feature_names = list(preprocessor.get_feature_names_out())
# Clean up prefix from ColumnTransformer
enc_feature_names_clean = [
    n.replace("num__", "").replace("cat__", "") for n in enc_feature_names
]

# ─── Full Pipeline (with State) ──────────────────────────────────────────────
full_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])
full_model.fit(X_train, y_train)

# ─── Reduced Pipeline (without State) ───────────────────────────────────────
reduced_model = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numerical_features)
    ])),
    ("regressor", LinearRegression())
])
reduced_model.fit(X_train[numerical_features], y_train)

pred_full    = full_model.predict(X_test)
pred_reduced = reduced_model.predict(X_test[numerical_features])


def calculate_metrics(y_true, y_pred):
    return {
        "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "R2":   round(float(r2_score(y_true, y_pred)), 4)
    }


comparison = pd.DataFrame({
    "Full model (with State)":    calculate_metrics(y_test, pred_full),
    "Reduced model (no State)":   calculate_metrics(y_test, pred_reduced)
}).T
print("\nModel Comparison:\n", comparison.round(3))

baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
pred_baseline    = baseline.predict(X_test)
final_predictions = full_model.predict(X_test)

evaluation = pd.DataFrame({
    "Baseline":                  calculate_metrics(y_test, pred_baseline),
    "Multiple Linear Regression": calculate_metrics(y_test, final_predictions)
}).T
print("\nEvaluation metrics:\n", evaluation.round(3))

results = pd.DataFrame({
    "Actual Profit":    y_test.to_numpy(),
    "Predicted Profit": final_predictions,
    "Residual":         y_test.to_numpy() - final_predictions
}, index=y_test.index)
print("\nActual vs. predicted:\n", results.round(2))

# Actual vs. predicted chart
plt.figure(figsize=(7, 6))
plt.scatter(y_test, final_predictions, alpha=0.8)
lower = min(y_test.min(), final_predictions.min())
upper = max(y_test.max(), final_predictions.max())
plt.plot([lower, upper], [lower, upper], linestyle="--", label="Perfect predictions")
plt.xlabel("Actual Profit"); plt.ylabel("Predicted Profit")
plt.title("Actual vs. Predicted Startup Profit")
plt.legend(); plt.tight_layout()
plt.savefig("static/charts/actual_vs_predicted.png", dpi=120)
plt.close()

# Residuals chart
plt.figure(figsize=(7, 5))
plt.scatter(final_predictions, results["Residual"], alpha=0.8)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Profit"); plt.ylabel("Residual")
plt.title("Residuals vs. Predicted Profit")
plt.tight_layout()
plt.savefig("static/charts/residuals.png", dpi=120)
plt.close()

# Residual distribution
plt.figure(figsize=(7, 5))
sns.histplot(results["Residual"], kde=True)
plt.xlabel("Residual"); plt.title("Distribution of Prediction Errors")
plt.tight_layout()
plt.savefig("static/charts/residual_distribution.png", dpi=120)
plt.close()

# Coefficients
fitted_preprocessor = full_model.named_steps["preprocessor"]
fitted_regressor    = full_model.named_steps["regressor"]
feature_names_out   = fitted_preprocessor.get_feature_names_out()
coefficients = pd.DataFrame({
    "Feature":     feature_names_out,
    "Coefficient": fitted_regressor.coef_
})
print("\nIntercept:", fitted_regressor.intercept_)
print("\nCoefficients:\n", coefficients.round(3).to_string(index=False))

# ════════════════════════════════════════════════════════════════════════════
# SECTION A — RMSE Comparison: 1 to 5 Features (Exhaustive Subsets by Size)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION A: RMSE Comparison (best subset of size 1..5)")
print("="*60)

n_enc = X_train_enc.shape[1]      # total encoded features (5 after OHE drop=first)
enc_names = enc_feature_names_clean

rmse_by_n = []   # [{n, best_subset, rmse, r2}]

for n in range(1, min(n_enc + 1, 6)):
    best_rmse   = np.inf
    best_subset = None
    for cols in itertools.combinations(range(n_enc), n):
        Xtr = X_train_enc[:, cols]
        Xte = X_test_enc[:, cols]
        reg = LinearRegression().fit(Xtr, y_train)
        preds = reg.predict(Xte)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        if rmse < best_rmse:
            best_rmse   = rmse
            best_subset = cols
    feat_labels = [enc_names[i] for i in best_subset]
    r2_val = r2_score(y_test, LinearRegression().fit(
        X_train_enc[:, best_subset], y_train).predict(X_test_enc[:, best_subset]))
    rmse_by_n.append({
        "n_features":   n,
        "best_subset":  feat_labels,
        "RMSE":         round(best_rmse, 2),
        "R2":           round(float(r2_val), 4)
    })
    print(f"  n={n}: RMSE={best_rmse:,.2f}  Features={feat_labels}")

rmse_df = pd.DataFrame(rmse_by_n)
rmse_df.to_csv("results/rmse_by_n_features.csv", index=False)

# Chart: RMSE vs. # features
plt.figure(figsize=(8, 5))
plt.plot(rmse_df["n_features"], rmse_df["RMSE"], marker="o", linewidth=2.5,
         color="#7c3aed", markersize=10)
for _, row in rmse_df.iterrows():
    plt.annotate(f"{row['RMSE']:,.0f}", (row["n_features"], row["RMSE"]),
                 textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
plt.title("Best-Subset RMSE vs. Number of Features")
plt.xlabel("Number of Features"); plt.ylabel("Test RMSE ($)")
plt.xticks(rmse_df["n_features"])
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("static/charts/rmse_by_n_features.png", dpi=120)
plt.close()
print("  → saved static/charts/rmse_by_n_features.png")

# ════════════════════════════════════════════════════════════════════════════
# SECTION B — Feature Selection Method Comparison
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION B: Feature Selection Method Comparison")
print("="*60)

selection_results = []


def eval_subset(col_indices, label):
    """Train LR on given (encoded) column indices; return metrics dict."""
    reg  = LinearRegression().fit(X_train_enc[:, col_indices], y_train)
    pred = reg.predict(X_test_enc[:, col_indices])
    names = [enc_names[i] for i in col_indices]
    return {
        "Method":           label,
        "Selected Features": names,
        "N Features":        len(names),
        "RMSE":              round(float(np.sqrt(mean_squared_error(y_test, pred))), 2),
        "MAE":               round(float(mean_absolute_error(y_test, pred)), 2),
        "R2":                round(float(r2_score(y_test, pred)), 4)
    }


# ── B1. LassoCV ─────────────────────────────────────────────────────────────
print("\n[1/5] LassoCV ...")
lasso = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000)
lasso.fit(X_train_enc, y_train)
lasso_mask  = lasso.coef_ != 0
lasso_cols  = np.where(lasso_mask)[0]
if len(lasso_cols) == 0:          # fallback if all zeroed out
    lasso_cols = np.array([np.argmax(np.abs(lasso.coef_))])
res = eval_subset(lasso_cols, "LassoCV")
print(f"   α={lasso.alpha_:.4f}  |  selected: {res['Selected Features']}")
selection_results.append(res)

# ── B2. RFECV ────────────────────────────────────────────────────────────────
print("[2/5] RFECV ...")
rfecv = RFECV(
    estimator=LinearRegression(),
    cv=5,
    scoring="neg_root_mean_squared_error",
    min_features_to_select=1
)
rfecv.fit(X_train_enc, y_train)
rfecv_cols = np.where(rfecv.support_)[0]
res = eval_subset(rfecv_cols, "RFECV")
print(f"   optimal n={rfecv.n_features_}  |  selected: {res['Selected Features']}")
selection_results.append(res)

# ── B3. Sequential Feature Selector (SFS) ───────────────────────────────────
print("[3/5] SFS ...")
sfs = SequentialFeatureSelector(
    LinearRegression(),
    n_features_to_select="auto",
    tol=1e-3,
    direction="forward",
    cv=5,
    scoring="neg_root_mean_squared_error"
)
sfs.fit(X_train_enc, y_train)
sfs_cols = np.where(sfs.get_support())[0]
res = eval_subset(sfs_cols, "SFS (Forward)")
print(f"   selected: {res['Selected Features']}")
selection_results.append(res)

# ── B4. ElasticNetCV ─────────────────────────────────────────────────────────
print("[4/5] ElasticNetCV ...")
enet = ElasticNetCV(
    cv=5,
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
    random_state=RANDOM_STATE,
    max_iter=10000
)
enet.fit(X_train_enc, y_train)
enet_mask = enet.coef_ != 0
enet_cols  = np.where(enet_mask)[0]
if len(enet_cols) == 0:
    enet_cols = np.array([np.argmax(np.abs(enet.coef_))])
res = eval_subset(enet_cols, "ElasticNetCV")
print(f"   α={enet.alpha_:.4f}  l1={enet.l1_ratio_:.2f}  |  selected: {res['Selected Features']}")
selection_results.append(res)

# ── B5. Exhaustive Feature Selection (best CV-RMSE across all subsets) ───────
print("[5/5] Exhaustive Feature Selection ...")
best_cv_rmse = np.inf
best_efs_cols = None
for r in range(1, n_enc + 1):
    for cols in itertools.combinations(range(n_enc), r):
        cols = list(cols)
        cv_scores = cross_val_score(
            LinearRegression(),
            X_train_enc[:, cols],
            y_train,
            cv=5,
            scoring="neg_root_mean_squared_error"
        )
        cv_rmse = -cv_scores.mean()
        if cv_rmse < best_cv_rmse:
            best_cv_rmse = cv_rmse
            best_efs_cols = cols
res = eval_subset(best_efs_cols, "Exhaustive FS")
print(f"   best CV-RMSE={best_cv_rmse:,.2f}  |  selected: {res['Selected Features']}")
selection_results.append(res)

# ── Summary ──────────────────────────────────────────────────────────────────
sel_df = pd.DataFrame(selection_results)
print("\nFeature Selection Comparison:")
print(sel_df[["Method", "N Features", "RMSE", "MAE", "R2",
              "Selected Features"]].to_string(index=False))
sel_df.to_csv("results/feature_selection_comparison.csv", index=False)

# Chart: RMSE per selection method
plt.figure(figsize=(10, 5))
colors = ["#7c3aed", "#0891b2", "#059669", "#d97706", "#dc2626"]
bars = plt.bar(sel_df["Method"], sel_df["RMSE"], color=colors, edgecolor="white", width=0.6)
for bar, val in zip(bars, sel_df["RMSE"]):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
             f"{val:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.title("RMSE by Feature Selection Method")
plt.xlabel("Method"); plt.ylabel("Test RMSE ($)")
plt.xticks(rotation=15, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("static/charts/feature_selection_rmse.png", dpi=120)
plt.close()
print("  → saved static/charts/feature_selection_rmse.png")

# Chart: N Features per method
plt.figure(figsize=(10, 4))
plt.bar(sel_df["Method"], sel_df["N Features"], color=colors, edgecolor="white", width=0.6)
plt.title("Number of Features Selected by Each Method")
plt.xlabel("Method"); plt.ylabel("# Features Selected")
plt.xticks(rotation=15, ha="right")
plt.yticks(range(0, n_enc + 2))
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("static/charts/feature_selection_n.png", dpi=120)
plt.close()

# ─── Save all result summaries as JSON for Flask ─────────────────────────────
with open("results/rmse_by_n.json", "w") as f:
    json.dump(rmse_by_n, f, indent=2)

sel_json = sel_df.copy()
sel_json["Selected Features"] = sel_json["Selected Features"].apply(lambda x: ", ".join(x))
sel_json.to_json("results/feature_selection.json", orient="records", indent=2)

# Save base model metrics for web dashboard
base_metrics = {
    "full_model":    calculate_metrics(y_test, pred_full),
    "reduced_model": calculate_metrics(y_test, pred_reduced),
    "baseline":      calculate_metrics(y_test, pred_baseline),

    "intercept":     round(float(fitted_regressor.intercept_), 2),
    "coefficients":  {
        str(n): round(float(c), 4)
        for n, c in zip(feature_names_out, fitted_regressor.coef_)
    }
}
with open("results/base_metrics.json", "w") as f:
    json.dump(base_metrics, f, indent=2)

print("\n[OK] All results saved to results/ and static/charts/")
print("[OK] Run  python app.py  then open  http://127.0.0.1:5000")