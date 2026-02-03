# === Import Required Libraries ===
import os
import re
import sys
import joblib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from numpy import savetxt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, Reshape, BatchNormalization 
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, AveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.initializers import GlorotUniform
from keras import layers, regularizers

print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))


def model(p, L, k):
    
    # ---------- Load Electrostatic features ----------
    features_path = f"X/X_electrostatic_p{p}_L{L}.csv"
    X_electrostatic_df = pd.read_csv(features_path)
    X_electrostatic_df = X_electrostatic_df.drop(columns=X_electrostatic_df.columns[0])

    # ---------- Load labels ----------
    comps_df = pd.read_csv('comp_17k_list.txt', sep='\s+', header = None, names = ['PDB_IDs'])
    print(comps_df)
    New_Set_labels = pd.read_csv('17k_CE_labels.txt', sep=',', header = None, names = ['PDB_IDs', 'CE'])
    New_Set_labels = New_Set_labels.drop_duplicates(subset=['PDB_IDs'])
    print(New_Set_labels)

    merged = pd.merge(X_electrostatic_df, New_Set_labels, on='PDB_IDs', how='inner')

    # Clean CE
    merged = merged.dropna(subset=['CE'])
    merged = merged[~merged['CE'].isin([np.inf, -np.inf])]

    print("Merged shape:", merged.shape)
    print("Example rows:")
    print(merged[['PDB_IDs', 'CE']].head())

    X = merged.drop(columns=['PDB_IDs', 'CE']).to_numpy(dtype=np.float32)
    y = merged['CE'].to_numpy(dtype=np.float32)

    print("Electrostatic feature shape:", X.shape)
    print("Number of elements in labels:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)          
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    histories = []
    y_true = []
    y_pred = []
    evaluation_metrics = []

    best_mse = float('inf')  
    filename = None 
    best_scaler_X = None
    best_scaler_y = None 

    counter = 1
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val = X_train[train_index], X_train[val_index]
        y_train_fold, y_val = y_train[train_index], y_train[val_index]

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
                
        history = model.fit(
                            X_train_fold_scaled, y_train_fold_scaled, 
                            batch_size = 32, epochs=300, 
                            validation_data = (X_val_scaled, y_val_scaled), verbose=1
                            )
                
        histories.append(history)

        y_val_pred_scaled = model.predict(X_val_scaled)
        y_val_pred = scaler_y_fold.inverse_transform(y_val_pred_scaled)

        y_true.append(y_val)
        y_pred.append(y_val_pred)

        # Metrics
        mse_scaled = mean_squared_error(y_val_scaled, y_val_pred_scaled)
        mape = mean_absolute_percentage_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)
        cc = np.corrcoef(y_val.squeeze(), y_val_pred.squeeze())[0, 1]
        
        print("Evaluation Metrics for Validation Data for k = ", counter)
        print(f"Mean Squared Error (MSE): {mse_scaled}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape}")
        print(f"R-squared (R2) Score: {r2}")
        print(f"Correlation Coefficient: {cc}")

        if mse_scaled < best_mse:
            best_mse = mse_scaled
            base_name = f"best_model_p{p}_L{L}_fold_{counter}"

            os.makedirs("models", exist_ok=True)
            model_filename = os.path.join("models", f"{base_name}.keras")
            model.save(model_filename)
            joblib.dump(scaler_X_fold, os.path.join("models", f"{base_name}_X_scaler.pkl"))
            joblib.dump(scaler_y_fold, os.path.join("models", f"{base_name}_y_scaler.pkl"))

            print(f"\n New best model saved with scaled MSE: {best_mse} at fold {counter} \n")

            filename = model_filename
            best_scaler_X = os.path.join("models", f"{base_name}_X_scaler.pkl")
            best_scaler_y = os.path.join("models", f"{base_name}_y_scaler.pkl")

        evaluation_metrics.append((mse_scaled, mape, r2, cc))
        counter = counter + 1

    # Calculate mean evaluation metrics across all folds
    mean_metrics = np.mean(evaluation_metrics, axis=0)
    print("mean evaluation metrics: ", mean_metrics)
    print(f"\n Best Model scaled MSE: {best_mse}, saved as {filename}\n ")

    # ---------- Evaluate best model on TEST ----------
    os.makedirs("plots/electro", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("evals/electro", exist_ok=True)

    best_model = load_model(filename)
    best_model.compile(optimizer=Adam(learning_rate=0.0001),
                       loss='mean_squared_error',       
                       metrics=['mse', 'mape']) 
    scaler_X_best = joblib.load(best_scaler_X)
    scaler_y_best = joblib.load(best_scaler_y)

    X_test_scaled = scaler_X_best.transform(X_test)
    y_pred_scaled = best_model.predict(X_test_scaled, verbose=0)
    y_pred_inv = scaler_y_best.inverse_transform(y_pred_scaled)
    y_test_scaled = scaler_y_best.transform(y_test.reshape(-1, 1)).ravel()

    test_mse_scaled = mean_squared_error(y_test_scaled, y_pred_scaled.ravel())
    test_rmse_scaled = np.sqrt(test_mse_scaled)
    test_mape = mean_absolute_percentage_error(y_test, y_pred_inv.ravel())
    test_cc = np.corrcoef(y_test.squeeze(), y_pred_inv.ravel().squeeze())[0, 1]
    test_r2 = r2_score(y_test, y_pred_inv.ravel())
    

    return histories, y_test, y_pred_inv, mean_metrics, test_mse_scaled, test_rmse_scaled, test_mape, test_cc, test_r2


def plot_mean_loss(histories, p, L):
    mean_train_loss = np.mean([h.history['loss'] for h in histories], axis=0)
    mean_val_loss = np.mean([h.history['val_loss'] for h in histories], axis=0)

    plt.figure(figsize=(8,5))
    plt.plot(mean_train_loss, label='Mean Training Loss')
    plt.plot(mean_val_loss, label='Mean Validation Loss')
    plt.xlabel('Epochs', fontsize = 22)
    plt.ylabel('Loss', fontsize = 22)
    plt.legend(fontsize=18)  
    plt.xticks(fontsize=14)  
    plt.yticks(fontsize=14) 
    plt.tight_layout()
    plt.savefig(f'plots/electro/loss_electro_p{p}_L{L}.png')

def plot_scatter(y_true, y_pred, p, L):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred) 
    plt.figure(figsize=(8,5))
    plt.scatter(y_true, y_pred, marker='o', facecolors='none', edgecolors='b')
    lo = min(y_true.min(), y_pred.min()); hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], color='red')
    plt.xlabel('Reference Values',fontsize=22)
    plt.ylabel('Predicted Values',fontsize = 22)
    plt.xticks(fontsize=14)  
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(f'plots/electro/scatter_electro_p{p}_L{L}.png')
  
if __name__ == "__main__":
    # Extract command-line arguments
    p = int(sys.argv[1])
    L = int(sys.argv[2])
    k = int(sys.argv[3])

    histories, y_true, y_pred, mean_metrics, test_mse_scaled, test_rmse_scaled, test_mape, test_cc, test_r2 = model(p, L, k)
    
    plot_mean_loss(histories, p, L)
    plot_scatter(y_true, y_pred, p, L)

    os.makedirs("outputs", exist_ok=True)  
    pd.DataFrame(y_true).to_csv(f'outputs/y_true_electro_p{p}_L{L}.csv', index=False)
    pd.DataFrame(y_pred).to_csv(f'outputs/y_pred_electro_p{p}_L{L}.csv', index=False)

    
    metrics_data = {
         "Metric": [
              "p",
              "L",
              "Mean Squared Error (MSE) Scaled",
              "Root Mean Squared Error (RMSE) Scaled",
              "Mean Absolute Percentage Error (MAPE)",
              "Correlation coefficient",
              "R-squared (R2) Score"],

        "Value": [
             p,
             L,
             test_mse_scaled,
             test_rmse_scaled,
             test_mape,
             test_cc,
             test_r2]}


    # Create the DataFrame
    df = pd.DataFrame(metrics_data).set_index("Metric").T
    print(df)

    df.to_csv("evals/electro/evaluation_metrics_p" + str(p) + "_L" + str(L) + "_CE.txt", sep='\t', index=False)




