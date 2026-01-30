import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from math import sqrt
from scipy.stats import t
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge


def mean_std_ci(x, confidence=0.95):
    x = np.asarray(x, dtype=float)
    n = x.size
    mean = float(x.mean())
    std = float(x.std(ddof=1))
    alpha = 1.0 - confidence
    tcrit = float(t.ppf(1.0 - alpha / 2.0, df=n - 1))
    half = tcrit * std / sqrt(n)
    return mean, std, mean - half, mean + half


def get_baseline_models(random_state=42):
    models = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, solver="lsqr"))
        ]),

        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            random_state=random_state,
            n_jobs=-1,
            max_features="sqrt",
        ),

        "GBDT": HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.1,
            max_iter=300,
            random_state=random_state
        )
    }

    return models


def cv_eval_models(X, y, groups, k_folds=5, confidence=0.95, feature_tag="Topo+Electro"):

    gkf = GroupKFold(n_splits=k_folds)
    models = get_baseline_models()
    y_var = np.var(y, ddof=1)

    rows = []
    for model_name, mdl in models.items():
        print(f"\n--- Model: {model_name} ---")
        fold_metrics = []  # [mse_scaled, mape, r2]

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
            t0 = time.time()

            print(f"  Fold {fold}/{k_folds}: fitting...")
            mdl.fit(X[tr_idx], y[tr_idx])

            print(f"  Fold {fold}/{k_folds}: predicting...")
            y_hat = mdl.predict(X[va_idx])

            mse = mean_squared_error(y[va_idx], y_hat)
            mse_scaled = mse / y_var
            mape = mean_absolute_percentage_error(y[va_idx], y_hat)
            r2 = r2_score(y[va_idx], y_hat)

            dt = time.time() - t0
            print(f"  Fold {fold}/{k_folds} done in {dt:.1f}s | MSE_scaled={mse_scaled:.4g} MAPE={mape:.4g} R2={r2:.4g}", flush=True)

            fold_metrics.append([mse_scaled, mape, r2])

        fold_metrics = np.asarray(fold_metrics, dtype=float)
        metric_names = ["MSE_scaled", "MAPE", "R2"]

        for j, mname in enumerate(metric_names):
            m, s, lo, hi = mean_std_ci(fold_metrics[:, j], confidence=confidence)
            rows.append([feature_tag, model_name, mname, m, s, lo, hi])

    return pd.DataFrame(rows, columns=["FeatureSet", "Model", "Metric", "Mean", "Std", "CI Low", "CI High"])


def fit_and_test_models(X_train, y_train, X_test, y_test, feature_tag="Topo+Electro"):

    models = get_baseline_models()
    rows = []
    y_var = np.var(y_train, ddof=1)
    for model_name, mdl in models.items():
        mdl.fit(X_train, y_train)
        pred = mdl.predict(X_test)

        mse = mean_squared_error(y_test, pred)
        mse_scaled = mse / y_var
        mape = mean_absolute_percentage_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        rows.append([feature_tag, model_name, mse_scaled, mape, r2])

    return pd.DataFrame(rows, columns=["FeatureSet", "Model", "Test_MSE_scaled", "Test_MAPE", "Test_R2"])


def run_baselines(p, L, k_folds):

    # ========== Load Topological features ==========
    with open("TDA_17k_MTF_features.pkl", "rb") as f:
        features = []
        while True:
            try:
                features.append(pickle.load(f))
            except EOFError:
                break

    X_topological = np.asarray(features[0])
    print("Loaded X_topological:", X_topological.shape)

    # Flatten topo
    X_topological_reshaped = X_topological.reshape(X_topological.shape[0], -1).astype(np.float32)
    print("Topo flattened shape:", X_topological_reshaped.shape)

    # ====== Load Electrostatic features ======
    features_path = f"X/X_electrostatic_p{p}_L{L}.csv"
    X_electrostatic_df = pd.read_csv(features_path)
    X_electrostatic_df = X_electrostatic_df.drop(columns=X_electrostatic_df.columns[0])
    X_electrostatic = X_electrostatic_df.drop(["PDB_IDs"], axis=1).to_numpy(dtype=np.float32)
    print("Electrostatic feature shape:", X_electrostatic.shape)

    # ====== Combine features (Topo + Electro) ======
    X_combined = np.concatenate([X_topological_reshaped, X_electrostatic], axis=1).astype(np.float32)
    print("Combined feature shape:", X_combined.shape)

    # ====== Load ids + labels ======
    comps_df = pd.read_csv("comp_17k_list.txt", sep=r"\s+", header=None, names=["PDB_IDs"])
    labels_df = pd.read_csv("17k_CE_labels.txt", sep=",", header=None, names=["PDB_IDs", "CE"])
    labels_df = labels_df.drop_duplicates(subset=["PDB_IDs"]).reset_index(drop=True)

    df = pd.merge(comps_df, labels_df, on="PDB_IDs", how="inner").copy()

    # ====== Load group keys ======
    group_df = pd.read_csv("pdb_group_key.csv")
    df = pd.merge(df, group_df, left_on="PDB_IDs", right_on="pdb_id", how="left")

    # Reindex df to comps_df order
    df = pd.merge(
        comps_df.assign(_idx=np.arange(len(comps_df))),
        df,
        on="PDB_IDs",
        how="inner",
    ).sort_values("_idx").reset_index(drop=True)

    idx = df["_idx"].to_numpy()
    X_both = X_combined[idx]
    y = df["CE"].to_numpy(dtype=np.float32)
    groups = df["group_key"].to_numpy(dtype=object)
    
    print("\nFinal aligned shapes:")
    print("  X_both     :", X_both.shape)
    print("  y          :", y.shape)
    print("  groups     :", groups.shape)
    print("N unique groups:", len(set(groups)))

    # ====== Group Train/Test split ======
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_both, y, groups=groups))

    groups_train = groups[train_idx]
    groups_test = groups[test_idx]
    overlap = set(groups_train).intersection(set(groups_test))
    print("\nGroup overlap train/test:", len(overlap))

    y_train, y_test = y[train_idx], y[test_idx]
    X_train_both, X_test_both = X_both[train_idx], X_both[test_idx]
    cv_both = cv_eval_models(X_train_both, y_train, groups_train, k_folds=k_folds, feature_tag="Topo+Electro")

    os.makedirs("evals/baselines", exist_ok=True)
    cv_both.to_csv(f"evals/baselines/baseline_cv_both_p{p}_L{L}_mean_std_ci.tsv", sep="\t", index=False)

    test_both = fit_and_test_models(X_train_both, y_train, X_test_both, y_test, feature_tag="Topo+Electro")
    test_both.to_csv(f"evals/baselines/baseline_eval_metrics_both_p{p}_L{L}.tsv", sep="\t", index=False)


    return cv_both, test_both


if __name__ == "__main__":
    p = int(sys.argv[1])
    L = int(sys.argv[2])
    k = int(sys.argv[3]) 

    cv_df, test_df = run_baselines(p, L, k_folds=k)
