import itertools
import json
import os
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)
BASE_DIR = os.path.dirname(__file__)
RANDOM_STATE = 42

DATASETS = {
    "startups": {
        "label": "50 Startups",
        "source": os.path.join(BASE_DIR, "data", "50_Startups.csv"),
    },
    "boston": {
        "label": "Boston Housing",
        "source": os.path.join(BASE_DIR, "data", "BostonHousing.csv"),
    },
}


def read_dataset(dataset_key, custom_url=None):
    if dataset_key == "custom":
        parsed = urlparse(custom_url or "")
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Custom dataset must be a valid http(s) raw CSV URL.")
        source = custom_url
        label = "Custom dataset"
    else:
        if dataset_key not in DATASETS:
            raise ValueError("Unknown dataset.")
        source = DATASETS[dataset_key]["source"]
        label = DATASETS[dataset_key]["label"]

    df = pd.read_csv(source)
    df.columns = [str(column).strip() for column in df.columns]
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if len(df) < 10:
        raise ValueError("The dataset must contain at least 10 rows.")
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_columns) < 2:
        raise ValueError("The dataset must contain at least two numeric columns.")
    target = numeric_columns[-1]
    predictors = [column for column in df.columns if column != target]
    if not predictors:
        raise ValueError("The dataset needs at least one predictor column.")
    return df, label, target, predictors


def make_model(df, target, predictors):
    numeric = [column for column in predictors
               if pd.api.types.is_numeric_dtype(df[column])]
    categorical = [column for column in predictors if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore",
                                     sparse_output=False)),
        ]), categorical))
    preprocessor = ColumnTransformer(transformers)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])


def metrics(y_true, predictions):
    return {
        "MAE": round(float(mean_absolute_error(y_true, predictions)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, predictions))), 2),
        "R2": round(float(r2_score(y_true, predictions)), 4),
    }


def analyze_dataset(dataset_key, custom_url=None):
    df, label, target, predictors = read_dataset(dataset_key, custom_url)
    X = df[predictors]
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    model = make_model(df, target, predictors)
    model.fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    predictions = model.predict(X_test)
    full_metrics = metrics(y_test, predictions)
    baseline_predictions = np.full(len(y_test), float(y_train.mean()))
    baseline_metrics = metrics(y_test, baseline_predictions)

    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]
    encoded_train = np.asarray(preprocessor.transform(X_train), dtype=float)
    encoded_test = np.asarray(preprocessor.transform(X_test), dtype=float)
    names = [name.replace("num__", "").replace("cat__", "")
             for name in preprocessor.get_feature_names_out()]
    coefficients = dict(zip(names, regressor.coef_.tolist()))
    all_estimates = np.r_[regressor.intercept_, regressor.coef_]
    design = np.column_stack([np.ones(len(encoded_train)), encoded_train])
    dof = max(len(y_train) - design.shape[1], 1)
    variance = np.sum((y_train.to_numpy() - train_predictions) ** 2) / dof
    covariance = variance * np.linalg.pinv(design.T @ design)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    t_stats = np.divide(all_estimates, standard_errors,
                        out=np.zeros_like(standard_errors),
                        where=standard_errors > 0)
    correlation = np.corrcoef(encoded_train, rowvar=False)
    vif = np.diag(np.linalg.pinv(np.atleast_2d(correlation)))
    coefficient_rows = []
    for index, (name, estimate) in enumerate(
            zip(["Intercept"] + names, all_estimates)):
        margin = 1.96 * standard_errors[index]
        coefficient_rows.append({
            "feature": name,
            "estimate": round(float(estimate), 4),
            "std_error": round(float(standard_errors[index]), 4),
            "t_stat": round(float(t_stats[index]), 3),
            "p_value": "< 0.0001" if abs(t_stats[index]) >= 4 else "n.s.",
            "confidence": "[{:,.2f}, {:,.2f}]".format(
                estimate - margin, estimate + margin),
            "vif": None if index == 0 else round(float(vif[index - 1]), 2),
        })

    residuals = y_test.to_numpy() - predictions
    ordered = sorted(zip(y_test.index, y_test, predictions, residuals),
                     key=lambda row: abs(row[3]), reverse=True)
    diagnostic_rows = []
    for rank, (index, actual, predicted, residual) in enumerate(ordered, 1):
        percentage = abs(float(residual)) / max(abs(float(actual)), 1e-12) * 100
        diagnostic_rows.append({
            "rank": rank, "sample_id": int(index) + 1,
            "actual": round(float(actual), 2),
            "predicted": round(float(predicted), 2),
            "residual": round(float(residual), 2),
            "abs_error": round(abs(float(residual)), 2),
            "percent_error": round(percentage, 2),
            "status": "High Residual" if percentage >= 10 else "In Range",
        })

    def evaluate_subset(indices, method):
        indices = list(indices)
        selected_model = LinearRegression().fit(
            encoded_train[:, indices], y_train
        )
        selected_predictions = selected_model.predict(encoded_test[:, indices])
        result = metrics(y_test, selected_predictions)
        return {
            "Method": method,
            "Selected Features": ", ".join(names[i] for i in indices),
            "N Features": len(indices),
            **result,
        }

    selection_results = []

    lasso = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000)
    lasso.fit(encoded_train, y_train)
    lasso_indices = np.flatnonzero(lasso.coef_)
    if len(lasso_indices) == 0:
        lasso_indices = np.array([np.argmax(np.abs(lasso.coef_))])
    selection_results.append(evaluate_subset(lasso_indices, "LassoCV"))

    rfecv = RFECV(
        estimator=LinearRegression(), cv=5,
        scoring="neg_root_mean_squared_error", min_features_to_select=1,
    )
    rfecv.fit(encoded_train, y_train)
    selection_results.append(
        evaluate_subset(np.flatnonzero(rfecv.support_), "RFECV")
    )

    sfs = SequentialFeatureSelector(
        LinearRegression(), n_features_to_select="auto", tol=1e-3,
        direction="forward", cv=5, scoring="neg_root_mean_squared_error",
    )
    sfs.fit(encoded_train, y_train)
    selection_results.append(
        evaluate_subset(np.flatnonzero(sfs.get_support()), "SFS (Forward)")
    )

    elastic_net = ElasticNetCV(
        cv=5, l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
        random_state=RANDOM_STATE, max_iter=10000,
    )
    elastic_net.fit(encoded_train, y_train)
    elastic_indices = np.flatnonzero(elastic_net.coef_)
    if len(elastic_indices) == 0:
        elastic_indices = np.array([np.argmax(np.abs(elastic_net.coef_))])
    selection_results.append(
        evaluate_subset(elastic_indices, "ElasticNetCV")
    )

    # Exhaustive search is bounded for wide datasets so a custom CSV cannot
    # make a request take exponentially long.
    exhaustive_candidates = np.arange(len(names))
    if len(exhaustive_candidates) > 10:
        correlations = np.abs(np.corrcoef(
            np.column_stack([encoded_train, y_train.to_numpy()]), rowvar=False
        )[:-1, -1])
        exhaustive_candidates = np.argsort(correlations)[-10:]
    best_cv_rmse = np.inf
    best_indices = [int(exhaustive_candidates[0])]
    for count in range(1, len(exhaustive_candidates) + 1):
        for subset in itertools.combinations(exhaustive_candidates, count):
            cv_rmse = -cross_val_score(
                LinearRegression(), encoded_train[:, subset], y_train,
                cv=5, scoring="neg_root_mean_squared_error",
            ).mean()
            if cv_rmse < best_cv_rmse:
                best_cv_rmse = cv_rmse
                best_indices = list(subset)
    selection_results.append(evaluate_subset(best_indices, "Exhaustive FS"))

    subset_rows = []
    encoded_names = names
    max_features = min(5, len(encoded_names))
    for count in range(1, max_features + 1):
        best = None
        for subset in itertools.combinations(range(len(encoded_names)), count):
            reg = LinearRegression().fit(encoded_train[:, subset], y_train)
            prediction = reg.predict(encoded_test[:, subset])
            result = metrics(y_test, prediction)
            if best is None or result["RMSE"] < best["RMSE"]:
                best = {**result, "n_features": count,
                        "best_subset": [encoded_names[i] for i in subset]}
        subset_rows.append(best)

    method_rows = selection_results

    equation = " = ".join([target, "{:,.2f}".format(regressor.intercept_)])
    equation += "".join(" {:+,.4f} × ({})".format(coef, name)
                        for name, coef in zip(names, regressor.coef_))
    return {
        "dataset_label": label, "target": target, "predictors": predictors,
        "feature_names": names, "row_count": len(df),
        "base_metrics": {"full_model": full_metrics, "baseline": baseline_metrics,
                         "intercept": float(regressor.intercept_)},
        "coefficients": coefficients, "coefficient_rows": coefficient_rows,
        "diagnostics": diagnostic_rows, "equation": equation,
        "rmse_by_n": subset_rows, "feature_sel": method_rows,
        "eda": {
            "target": target,
            "target_values": [round(float(value), 4) for value in y.tolist()],
            "actual": [round(float(value), 4) for value in y_test.tolist()],
            "predicted": [round(float(value), 4) for value in predictions.tolist()],
        },
    }


def selected_inputs():
    return request.args.get("dataset", "startups"), request.args.get("custom_url")


@app.route("/")
def index():
    dataset, custom_url = selected_inputs()
    try:
        dashboard = analyze_dataset(dataset, custom_url)
        error = None
    except Exception as exc:
        dashboard = analyze_dataset("startups")
        error = str(exc)
    return render_template("index.html", dashboard=dashboard,
                           base_metrics=dashboard["base_metrics"],
                           rmse_by_n=dashboard["rmse_by_n"],
                           feature_sel=dashboard["feature_sel"],
                           diagnostics={
                               "equation": dashboard["equation"],
                               "coefficients_table": dashboard["coefficient_rows"],
                               "diagnostics": dashboard["diagnostics"],
                           },
                           selected_dataset=dataset, custom_url=custom_url or "",
                           datasets=DATASETS, error=error)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        dataset = data.get("dataset", "startups")
        custom_url = data.get("custom_url")
        df, _, target, predictors = read_dataset(dataset, custom_url)
        model = make_model(df, target, predictors)
        model.fit(df[predictors], pd.to_numeric(df[target], errors="coerce"))
        sample = {}
        for column in predictors:
            value = data.get("features", {}).get(column, "")
            if pd.api.types.is_numeric_dtype(df[column]):
                sample[column] = float(value or 0)
            else:
                sample[column] = str(value)
        prediction = float(model.predict(pd.DataFrame([sample]))[0])
        return jsonify({"success": True, "target": target,
                        "profit": round(prediction, 2)})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/dashboard")
def api_dashboard():
    dataset, custom_url = selected_inputs()
    try:
        return jsonify(analyze_dataset(dataset, custom_url))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
