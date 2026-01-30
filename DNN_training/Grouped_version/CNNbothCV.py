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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from math import sqrt
from scipy.stats import t


def model(p, L, k_folds):

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

    # ========== Load Electrostatic features ==========
    comps_df = pd.read_csv('comp_17k_list.txt', sep=r'\s+', header=None, names=['PDB_IDs'])

    features_path = f"X/X_electrostatic_p{p}_L{L}.csv"
    X_electrostatic_df = pd.read_csv(features_path)
    X_electrostatic_df = X_electrostatic_df.drop(columns=X_electrostatic_df.columns[0])
    X_electrostatic_df = pd.merge(comps_df, X_electrostatic_df, on="PDB_IDs", how="left")
    X_electrostatic_df = X_electrostatic_df.drop(['PDB_IDs'], axis = 1).to_numpy(dtype=np.float32)
    print("Electrostatic feature shape:", X_electrostatic_df.shape)

    X_topological_reshaped = X_topological.reshape(X_topological.shape[0], -1) # Flatten the image features
    X_combined = np.concatenate((X_topological_reshaped, np.array(X_electrostatic_df)), axis=1)

    print("Combined feature shape:", X_combined.shape)


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
    X = X_combined[idx]
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

        # Get topological features from fold data
        n_img_feats = X_topological_reshaped.shape[1]

        X_train_images_fold = X_train_fold[:, :n_img_feats].reshape(-1, 200, 12, 1) 
        X_val_images = X_val[:, :n_img_feats].reshape(-1, 200, 12, 1) 
        X_test_images = X_test[:, :n_img_feats].reshape(-1, 200, 12, 1)
 
        
        # Center train images
        datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
        datagen.fit(X_train_images_fold)
        iterator = datagen.flow(X_train_images_fold, batch_size=len(X_train_images_fold), shuffle=False)
        X_train_centered = iterator.__next__()
        print(X_train_centered.shape, X_train_centered.mean(), X_train_centered.std())

        # Center val images
        iterator = datagen.flow(X_val_images, batch_size=len(X_val_images), shuffle=False)
        X_val_centered = iterator.__next__()
        print(X_val_centered.shape, X_val_centered.mean(), X_val_centered.std())

        # Center test images
        iterator = datagen.flow(X_test_images, batch_size=len(X_test_images), shuffle=False)
        X_test_centered = iterator.__next__()
        print(X_test_centered.shape, X_test_centered.mean(), X_test_centered.std())

        # Reshape 
        X_train_centered = X_train_centered.reshape((X_train_centered.shape[0], X_train_centered.shape[1], X_train_centered.shape[2]))
        X_test_centered = X_test_centered.reshape((X_test_centered.shape[0], X_test_centered.shape[1], X_test_centered.shape[2]))
        X_val_centered = X_val_centered.reshape((X_val_centered.shape[0], X_val_centered.shape[1], X_val_centered.shape[2]))

        scaler_X_fold = StandardScaler()
        X_train_ef_scaled = scaler_X_fold.fit_transform(X_train_fold[:, n_img_feats:])
        X_test_ef_scaled = scaler_X_fold.transform(X_test[:, n_img_feats:])
        X_val_ef_scaled = scaler_X_fold.transform(X_val[:, n_img_feats:])

        # SCALE y
        scaler_y_fold = StandardScaler()
        y_train_scaled = scaler_y_fold.fit_transform(np.array(y_train_fold).reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y_fold.transform(np.array(y_test).reshape(-1, 1)).flatten()
        y_val_scaled = scaler_y_fold.transform(np.array(y_val).reshape(-1, 1)).flatten()

        inputCNN = keras.Input(shape = X_train_centered.shape[1:])
        inputEF = keras.Input(shape = (X_train_ef_scaled.shape[1],)) 

        # CNN model
        x = Conv1D(filters=128, kernel_size=3, activation='tanh', kernel_regularizer=l2(0.02))(inputCNN)
        x = AveragePooling1D(pool_size=2)(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.02))(x)
        x = AveragePooling1D(pool_size=2)(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters=64, kernel_size=3, activation='tanh', kernel_regularizer=l2(0.02))(x)
        x = AveragePooling1D(pool_size=2)(x)
        x = Dropout(0.3)(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        modelCNN = keras.Model(inputs=inputCNN, outputs=x)

        # EF model
        y = Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01))(inputEF)
        y = Dropout(0.15)(y)
        y = BatchNormalization()(y)
        y = Dense(64, activation='relu', kernel_regularizer=l2(0.02))(y)
        modelEF = keras.Model(inputs=inputEF, outputs=y)

        # Merged model
        mergedOutput = concatenate([modelCNN.output, modelEF.output])  
        z = Dense(32, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(mergedOutput)
        z = Dropout(0.15)(z)
        z = BatchNormalization()(z)
        z = Dense(16, activation='relu', kernel_regularizer=l2(0.02))(z)
        z = Dropout(0.15)(z)
        z = BatchNormalization()(z)
        z = Dense(8, activation='relu', kernel_regularizer=l2(0.02))(z)
        finalOutput = Dense(1)(z)
        model_merged = keras.Model(inputs=[modelCNN.input, modelEF.input], outputs=finalOutput)

        adam = Adam(learning_rate = 0.0001)
        model_merged.compile(loss = 'mean_squared_error',
                        optimizer = adam,
                        metrics=['mean_squared_error'])

        history = model_merged.fit(
            [X_train_centered, X_train_ef_scaled], y_train_scaled, 
            batch_size = 16, epochs=100, 
            validation_data = ([X_val_centered, X_val_ef_scaled], y_val_scaled),
            verbose=1)

        histories.append(history)

        y_val_pred_scaled = model_merged.predict([X_val_centered, X_val_ef_scaled]).ravel()
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

            base_name = f"best_model_fold_{counter}"
            os.makedirs("models", exist_ok=True)

            model_filename = os.path.join("models", f"{base_name}.keras")
            model_merged.save(model_filename)
            best_model_filename = model_filename

            joblib.dump(
                {
                    "scaler_X": scaler_X_fold,
                    "scaler_y": scaler_y_fold,
                    "img_mean": datagen.mean,
                    "img_std": datagen.std,
                    "n_img_feats": n_img_feats,
                },
                os.path.join("models", f"{base_name}_preproc.pkl")
            )
            best_preproc_filename = os.path.join("models", f"{base_name}_preproc.pkl")

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
    os.makedirs("evals/both", exist_ok=True)
    cv_summary.to_csv(f"evals/both/cv_metrics_both_p{p}_L{L}_mean_std_ci.csv", sep="\t", index=False)

    print(f"\nBest fold model: {best_model_filename} (best val MSE_scaled={best_mse_scaled})\n")

    # ========== Evaluate best model on GROUPED test set ==========

    best_model = load_model(best_model_filename)
    pp = joblib.load(best_preproc_filename)

    n_img_feats = pp["n_img_feats"]
    X_test_images = X_test[:, :n_img_feats].reshape(-1, 200, 12, 1)
    X_test_centered = (X_test_images - pp["img_mean"]) / (pp["img_std"] + 1e-8)
    X_test_centered = X_test_centered.reshape(X_test_centered.shape[0], 200, 12)
    X_test_ef_scaled = pp["scaler_X"].transform(X_test[:, n_img_feats:])
    y_pred_scaled = best_model.predict([X_test_centered, X_test_ef_scaled]).ravel()
    y_pred = pp["scaler_y"].inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
    y_test = y_test.reshape(-1, 1).ravel()

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

    tr_lo = np.maximum(mean_train - std_train, 0.0)
    va_lo = np.maximum(mean_val - std_val, 0.0)
    # tr_lo = mean_train - std_train
    tr_hi = mean_train + std_train
    # va_lo = mean_val - std_val
    va_hi = mean_val + std_val

    os.makedirs('plots/both', exist_ok=True)
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
    plt.savefig(f'plots/both/loss_both_p{p}_L{L}.png')

def plot_scatter(y_true, y_pred):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    os.makedirs('plots/both', exist_ok=True)
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
    plt.savefig(f'plots/both/scatter_both_p{p}_L{L}.png')


if __name__ == "__main__":
    # Extract command-line arguments
    p = int(sys.argv[1])
    L = int(sys.argv[2])
    k = int(sys.argv[3])

    (histories, y_true, y_pred, cv_summary, test_mse_scaled, test_mape, test_r2) = model(p, L, k)

    plot_CV_loss(histories)
    plot_scatter(y_true, y_pred)

    os.makedirs('outputs', exist_ok=True)
    os.makedirs("evals/both", exist_ok=True)
    pd.DataFrame(y_true).to_csv(f'outputs/y_true_both_p{p}_L{L}.csv', index=False)
    pd.DataFrame(y_pred).to_csv(f'outputs/y_pred_both_p{p}_L{L}.csv', index=False)

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

    dfm.to_csv(f"evals/both/evaluation_metrics_both_p{p}_L{L}_CE.txt", sep='\t', index=False)














