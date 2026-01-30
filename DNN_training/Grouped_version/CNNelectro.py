import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import os
import pickle
import joblib
import matplotlib.pyplot as plt
import sys
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv1D, AveragePooling1D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from math import sqrt
from scipy.stats import t


def model(p, L, k_folds):

    # ========== Load Electrostatic features ==========
    comps_df = pd.read_csv('comp_17k_list.txt', sep=r'\s+', header=None, names=['PDB_IDs'])

    features_path = f"X/X_electrostatic_p{p}_L{L}.csv"
    X_electrostatic_df = pd.read_csv(features_path)
    X_electrostatic_df = X_electrostatic_df.drop(columns=X_electrostatic_df.columns[0])
    X_electrostatic_df = pd.merge(comps_df, X_electrostatic_df, on="PDB_IDs", how="left")
    X_electrostatic_df = X_electrostatic_df.drop(['PDB_IDs'], axis = 1).to_numpy(dtype=np.float32)
    print("Electrostatic feature shape:", X_electrostatic_df.shape)


    # ========== Load ids + labels ==========
    labels_df = pd.read_csv('17k_CE_labels.txt', sep=',', header=None, names=['PDB_IDs', 'CE'])
    labels_df = labels_df.drop_duplicates(subset=['PDB_IDs']).reset_index(drop=True)

    df = pd.merge(comps_df, labels_df, on='PDB_IDs', how='inner').copy()

    # ========== Load group keys ==========
    group_df = pd.read_csv("pdb_group_key.csv")
    df = pd.merge(df, group_df, left_on="PDB_IDs", right_on="pdb_id", how="left")

    # Reindex df to comps_df order 
    df = pd.merge(comps_df.assign(_idx=np.arange(len(comps_df))),
                  df, on="PDB_IDs", how="inner").sort_values("_idx").reset_index(drop=True)

    idx = df["_idx"].to_numpy()
    X = X_electrostatic_df[idx]
    y = df["CE"].to_numpy(dtype=np.float32)
    groups = df["group_key"].to_numpy(dtype=object)

    print("Final aligned shapes -> X:", X.shape, "y:", y.shape, "groups:", groups.shape)
    print("N unique groups:", len(set(groups)))

    # ========== Group Train/Test split ==========

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]
    groups_test  = groups[test_idx]

    # Check leakage
    overlap = set(groups_train).intersection(set(groups_test))
    print("Group overlap train/test:", len(overlap))
    gkf = GroupKFold(n_splits=k_folds)

    histories = []
    fold_metrics = []

    best_mse_scaled = float('inf')
    best_model_filename = None
    counter = 1

    for train_idx, val_idx in gkf.split(X_train, y_train, groups=groups_train):
        X_train_fold, X_val = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val = y_train[train_idx], y_train[val_idx]

        print(X_train_fold.shape)
        print(X_val.shape)
        print(y_train_fold.shape)
        print(y_val.shape)
 
        # --- Scale fold data ---
        scaler_X_fold = StandardScaler()
        X_train_fold_scaled = scaler_X_fold.fit_transform(X_train_fold)
        X_val_scaled = scaler_X_fold.transform(X_val)

        scaler_y_fold = StandardScaler()
        y_train_fold_scaled = scaler_y_fold.fit_transform(y_train_fold.reshape(-1, 1)).flatten()
        y_val_scaled = scaler_y_fold.transform(y_val.reshape(-1, 1)).flatten()

        # --- Sequential Model ---
        model = Sequential()
        model.add(Input(shape=(X.shape[1],)))
        model.add(layers.Dense(128, activation="relu",kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.15))
        model.add(layers.Dense(64))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(Dropout(0.15))
        model.add(layers.Dense(32))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(Dropout(0.15))
        model.add(layers.Dense(16))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(Dropout(0.15))
        model.add(layers.Dense(8))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(Dropout(0.15))
        model.add(layers.Dense(4))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(Dropout(0.15))
        model.add(layers.Dense(1))
        
        adam = Adam(learning_rate = 0.0001)

        model.compile(loss = 'mean_squared_error', optimizer = adam, metrics=['mean_squared_error'])   
                
        history = model.fit(X_train_fold_scaled, y_train_fold_scaled, 
                            batch_size = 32, epochs=100, 
                            validation_data = (X_val_scaled, y_val_scaled), verbose=1)
                
        histories.append(history)

        y_val_pred_scaled = model.predict(X_val_scaled).ravel()
        y_val_pred = scaler_y_fold.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).ravel()

        # --- Metrics ---
        mse_scaled = mean_squared_error(y_val_scaled, y_val_pred_scaled)
        mape = mean_absolute_percentage_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)

        print(f"\nFold {counter} validation:")
        print(f" Mean Squared Error (MSE) scaled: {mse_scaled}")
        print(f" Mean Absolute Percentage Error (MAPE):  {mape}")
        print(f" R-squared (R2) Score:   {r2}")


        if mse_scaled < best_mse_scaled:
            best_mse_scaled = mse_scaled
            base_name = f"best_model_p{p}_L{L}_fold_{counter}"

            # Save the model
            os.makedirs("models", exist_ok=True)
            model_filename = os.path.join("models", f"{base_name}.keras")
            model.save(model_filename)
            joblib.dump(scaler_X_fold, os.path.join("models", f"{base_name}_X_scaler.pkl"))
            joblib.dump(scaler_y_fold, os.path.join("models", f"{base_name}_y_scaler.pkl"))

            print(f"\n New best model saved with scaled MSE: {best_mse_scaled} at fold {counter} \n")

            best_model_filename = model_filename
            best_scaler_X = os.path.join("models", f"{base_name}_X_scaler.pkl")
            best_scaler_y = os.path.join("models", f"{base_name}_y_scaler.pkl")

        fold_metrics.append([mse_scaled, mape, r2])
        counter += 1

    # ========== CV Summary ==========
    fold_metrics = np.array(fold_metrics, dtype=float)
    names = ["MSE_scaled", "MAPE", "R2"]

    cv_mean = fold_metrics.mean(axis=0)
    cv_std  = fold_metrics.std(axis=0, ddof=1) 

    print("\nCV summary across folds:")
    rows = []
    for j, name in enumerate(names):
        m, s, lo, hi = mean_std_ci(fold_metrics[:, j], confidence=0.95)
        print(f"  {name}: mean={m}, std={s}, CI=[{lo}, {hi}]")
        rows.append([name, m, s, lo, hi])

    cv_summary = pd.DataFrame(rows, columns=["Metric", "Mean", "Std", "CI Low", "CI High"])
    os.makedirs("evals/electro", exist_ok=True)
    cv_summary.to_csv(f"evals/electro/cv_metrics_electro_p{p}_L{L}_mean_std_ci.csv", sep="\t", index=False)

    print(f"\nBest fold model: {best_model_filename} (best val MSE_scaled={best_mse_scaled})\n")

    # ========== Evaluate best model on GROUPED test set ==========

    best_model = load_model(best_model_filename)
    best_model.compile(optimizer=Adam(learning_rate=0.0001),
                       loss='mean_squared_error',       
                       metrics=['mse', 'mape']) 
    scaler_X_best = joblib.load(best_scaler_X)
    scaler_y_best = joblib.load(best_scaler_y)

    X_test_scaled = scaler_X_best.transform(X_test)
    y_pred_scaled = best_model.predict(X_test_scaled, verbose=0)
    y_pred_inv = scaler_y_best.inverse_transform(y_pred_scaled)
    y_test_scaled = scaler_y_best.transform(y_test.reshape(-1, 1)).ravel()
    y_test = y_test.reshape(-1, 1).ravel()
    y_pred = y_pred_inv.ravel()

    test_mse_scaled = mean_squared_error(y_test_scaled, y_pred_scaled)
    test_mape = mean_absolute_percentage_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    return histories, y_test, y_pred, cv_summary, test_mse_scaled, test_mape, test_r2

def mean_std_ci(x, confidence=0.95):

    x = np.asarray(x, dtype=float)
    n = x.size
    mean = float(x.mean())
    std = float(x.std(ddof=1)) 

    alpha = 1.0 - confidence
    tcrit = float(t.ppf(1.0 - alpha/2.0, df=n - 1))
    half = tcrit * std / sqrt(n)
    return mean, std, mean - half, mean + half

def plot_CV_loss(histories, title="Loss (Mean ± Std)"):

    train_loss = np.array([h.history["loss"]     for h in histories], dtype=float)
    val_loss   = np.array([h.history["val_loss"] for h in histories], dtype=float)

    min_len = min(train_loss.shape[1], val_loss.shape[1])
    train_loss = train_loss[:, :min_len]
    val_loss   = val_loss[:, :min_len]
    epochs = np.arange(1, min_len + 1)

    mean_train = train_loss.mean(axis=0)
    std_train  = train_loss.std(axis=0, ddof=1) 

    mean_val = val_loss.mean(axis=0)
    std_val  = val_loss.std(axis=0, ddof=1) 

    # tr_lo = np.maximum(mean_train - std_train, 0.0)
    # va_lo = np.maximum(mean_val - std_val, 0.0)
    tr_lo = mean_train - std_train
    tr_hi = mean_train + std_train
    va_lo = mean_val - std_val
    va_hi = mean_val + std_val

    os.makedirs('plots/electro', exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, mean_train, label="Train Loss (mean)", linewidth=2.2)
    plt.plot(epochs, mean_val,   label="Val Loss (mean)",   linewidth=2.2)
    plt.fill_between(epochs, tr_lo, tr_hi, alpha=0.25)
    plt.fill_between(epochs, va_lo, va_hi, alpha=0.25)
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'plots/electro/loss_electro_p{p}_L{L}.png')

def plot_scatter(y_true, y_pred):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    os.makedirs('plots/electro', exist_ok=True)
    plt.figure(figsize=(9, 6))
    plt.scatter(y_true, y_pred, marker='o', facecolors='none', edgecolors='b')
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], color='red')
    plt.xlabel('Reference Values', fontsize=20)
    plt.ylabel('Predicted Values', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(f'plots/electro/scatter_electro_p{p}_L{L}.png')


if __name__ == "__main__":
    # Extract command-line arguments
    p = int(sys.argv[1])
    L = int(sys.argv[2])
    k = int(sys.argv[3])

    (histories, y_true, y_pred, cv_summary, test_mse_scaled, test_mape, test_r2) = model(p, L, k)

    plot_CV_loss(histories)
    plot_scatter(y_true, y_pred)

    os.makedirs('outputs', exist_ok=True)
    os.makedirs("evals/electro", exist_ok=True)
    pd.DataFrame(y_true).to_csv(f'outputs/y_true_electro_p{p}_L{L}.csv', index=False)
    pd.DataFrame(y_pred).to_csv(f'outputs/y_pred_electro_p{p}_L{L}.csv', index=False)

    # Save test metrics
    metrics_data = {
         "Metric": [
              "Mean Squared Error (MSE) scaled",
              "Mean Absolute Percentage Error (MAPE)",
              "R-squared (R2) Score"],
        "Value": [
             test_mse_scaled,
             test_mape,
             test_r2]
    }
    
    dfm = pd.DataFrame(metrics_data).set_index("Metric").T
    print(dfm)

    dfm.to_csv(f"evals/electro/evaluation_metrics_electro_p{p}_L{L}_CE.txt", sep='\t', index=False)


