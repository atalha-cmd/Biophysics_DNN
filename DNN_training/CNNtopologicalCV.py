import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import os 
import re
import pickle
import joblib
import matplotlib.pyplot as plt
import sys
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from numpy import savetxt
from keras import regularizers
from keras.initializers import GlorotUniform
from keras import regularizers
from keras import layers
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.layers import Reshape, concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, AveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, AveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error 
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA  
from sklearn.model_selection import KFold


def model(k):
    
    pickle_file = open("TDA_17k_MTF_features.pkl", "rb")
    features = []
    while True:
        try:
            features.append(pickle.load(pickle_file))
        except EOFError:
            break
    pickle_file.close()

    X_topological = np.asarray(features[0])

    # ---------- labels ----------
    comps_df = pd.read_csv('comp_17k_list.txt', sep='\s+', header = None, names = ['PDB_IDs'])
    print(comps_df)
    New_Set_labels = pd.read_csv('17k_CE_labels.txt', sep=',', header = None, names = ['PDB_IDs', 'CE'])
    New_Set_labels = New_Set_labels.drop_duplicates(subset=['PDB_IDs']).reset_index(drop=True)
    print(New_Set_labels)

    y_df = pd.merge(comps_df, New_Set_labels, on='PDB_IDs', how='inner')

    print(y_df.shape)
    print("number of elements in labels:",len(y_df))
    print("Shape X: ", X_topological.shape)

    y_df = y_df.drop(['PDB_IDs'], axis = 1)
    y = y_df['CE'].to_numpy().astype(np.float32)

    histories = []
    y_true = []
    y_pred = []
    evaluation_metrics = []

    X_train, X_test, y_train, y_test = train_test_split(X_topological, y, test_size=0.2, random_state=42)
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    best_mse = float('inf')  
    filename = None   
    counter = 1

    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val = X_train[train_index], X_train[val_index]
        y_train_fold, y_val = y_train[train_index], y_train[val_index]

        width, height, channels = X_train_fold.shape[1], X_train_fold.shape[2], 1
        X_train_fold = X_train_fold.reshape((X_train_fold.shape[0], width, height, channels))
        X_val = X_val.reshape((X_val.shape[0], width, height, channels))
        
        datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
        datagen.fit(X_train_fold)

        train_iterator = datagen.flow(X_train_fold, batch_size=len(X_train_fold), shuffle=False)
        X_train_centered = train_iterator.__next__()

        val_iterator = datagen.flow(X_val, batch_size=len(X_val), shuffle=False)
        X_val_centered = val_iterator.__next__()

        print(X_train_centered.shape, X_train_centered.mean(), X_train_centered.std())
        print(X_val_centered.shape, X_val_centered.mean(), X_val_centered.std())

        X_train_centered = X_train_centered.reshape((X_train_centered.shape[0], width, height))
        X_val_centered = X_val_centered.reshape((X_val_centered.shape[0], width, height))

        # SCALE y
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(np.array(y_train_fold).reshape(-1, 1)).flatten()
        y_val_scaled = scaler_y.transform(np.array(y_val).reshape(-1, 1)).flatten()

        inputCNN = keras.Input(shape = X_train_centered.shape[1:])

        # -------------- CNN --------------
        x = Conv1D(filters=128, kernel_size=3, activation='tanh', kernel_regularizer=l2(0.02))(inputCNN)
        x = AveragePooling1D(pool_size=2)(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)

        x = Conv1D(filters=256, kernel_size=3, activation='relu', kernel_regularizer=l2(0.02))(x)
        x = AveragePooling1D(pool_size=2)(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)

        x = Conv1D(filters=128, kernel_size=3, activation='tanh', kernel_regularizer=l2(0.02))(x)
        x = AveragePooling1D(pool_size=2)(x)
        x = Dropout(0.3)(x)

        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.15)(x)
        out = Dense(1)(x)

        modelCNN = keras.Model(inputs=inputCNN, outputs=out)

        adam = Adam(learning_rate = 0.0001)
        modelCNN.compile(loss = 'mean_squared_error', optimizer = adam, metrics=['mean_squared_error'])

        history = modelCNN.fit(
            X_train_centered, y_train_scaled, 
            batch_size = 16, epochs=300, 
            validation_data = (X_val_centered, y_val_scaled),
            verbose=1
        )
        histories.append(history)
       
        y_val_pred_scaled = modelCNN.predict(X_val_centered)
        y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled)
        
        y_true.append(y_val)
        y_pred.append(y_val_pred)

        mse = mean_squared_error(y_val_scaled, y_val_pred_scaled.ravel())
        mape = mean_absolute_percentage_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)
        cc = np.corrcoef(np.array(y_val).squeeze(), np.array(y_val_pred).squeeze())[0, 1]
        
        print("Evaluation Metrics for Validation Data for k = ", counter)
        print(f"Mean Squared Error (MSE) scaled: {mse}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape}")
        print(f"R-squared (R2) Score: {r2}")
        print(f"Correlation Coefficient: {cc}")

        if mse < best_mse:
            best_mse = mse
            base_name = f"best_model_fold_{counter}"

            model_filename = f"{base_name}.keras"
            modelCNN.save(model_filename)
            print(f"\n New best model saved with MSE: {best_mse} at fold {counter} \n")

            np.savez(f"{base_name}_norm_stats.npz", mean=datagen.mean, std=datagen.std)
            joblib.dump(scaler_y, f"{base_name}_y_scaler.pkl")
            filename = model_filename

        evaluation_metrics.append((mse, mape, r2, cc))
        counter = counter + 1 

    mean_metrics = np.mean(evaluation_metrics, axis=0)
    print("\n mean evaluation metrics: ", mean_metrics)
    print(f"\n Best Model MSE: {best_mse}, saved as {filename}\n ")

    # Evaluate best model on test
    best_model = load_model(filename)
    best_model.compile(optimizer=Adam(learning_rate=0.0001),
                       loss='mean_squared_error',       
                       metrics=['mse', 'mape'])    
    
    f_name = filename.rsplit('.', 1)[0]  
    stats = np.load(f"{f_name}_norm_stats.npz")
    mean, std = stats["mean"], stats["std"] + 1e-7

    X_test_4d = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
    X_test_4d = (X_test_4d - mean) / std
    X_test_centered = X_test_4d.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    y_pred_scaled = best_model.predict(X_test_centered)

    scaler_y = joblib.load(f"{f_name}_y_scaler.pkl")
    y_pred_inv = scaler_y.inverse_transform(y_pred_scaled)
    y_test = np.array(y_test)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    test_mse_scaled = mean_squared_error(y_test_scaled, y_pred_scaled.ravel())
    test_rmse_scaled = np.sqrt(test_mse_scaled)
    test_mape = mean_absolute_percentage_error(y_test, y_pred_inv)
    test_cc = np.corrcoef(np.array(y_test).squeeze(), np.array(y_pred_inv).squeeze())[0, 1]
    test_r2 = r2_score(y_test, y_pred_inv)

    print("y_test:", y_test[0:10].ravel())
    print("y_pred_inv:", y_pred_inv[0:10].ravel())

    return histories, y_test, y_pred_inv, mean_metrics, test_mse_scaled, test_rmse_scaled, test_mape, test_cc, test_r2


def plot_mean_loss(histories):
    mean_train_loss = np.mean([h.history['loss'] for h in histories], axis=0)
    mean_val_loss = np.mean([h.history['val_loss'] for h in histories], axis=0)

    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.plot(mean_train_loss, label='Mean Training Loss')
    plt.plot(mean_val_loss, label='Mean Validation Loss')
    plt.xlabel('Epochs', fontsize = 22)
    plt.ylabel('Loss', fontsize = 22)
    plt.legend(fontsize=18)  
    plt.xticks(fontsize=14)  
    plt.yticks(fontsize=14)  
    plt.tight_layout()
    plt.savefig('plots/loss_topological.png')

def plot_scatter(y_true, y_pred):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.scatter(y_true, y_pred, marker='o', facecolors='none', edgecolors='b')
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], color='red')
    plt.xlabel('Reference Values', fontsize=22)
    plt.ylabel('Predicted Values', fontsize = 22)
    plt.xticks(fontsize=14)  
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/scatter_topological.png')
  
if __name__ == "__main__":
    k = int(sys.argv[1])

    histories, y_true, y_pred, mean_metrics,  test_mse_scaled,  test_rmse_scaled,  test_mape, test_cc, test_r2 = model(k)
    
    plot_mean_loss(histories)
    plot_scatter(y_true, y_pred)

    os.makedirs('outputs', exist_ok=True)
    os.makedirs('evals', exist_ok=True)

    pd.DataFrame(y_true).to_csv('outputs/y_true_topological.csv', index=False)
    pd.DataFrame(y_pred).to_csv('outputs/y_pred_topological.csv', index=False)

    metrics_data = {
         "Metric": [
              "Mean Squared Error (MSE) scaled",
              "Root Mean Squared Error (RMSE) scaled",
              "Mean Absolute Percentage Error (MAPE)",
              "Correlation coefficient",
              "R-squared (R2) Score"],
        "Value": [
             test_mse_scaled,
             test_rmse_scaled,
             test_mape,
             test_cc,
             test_r2]
     }
    df = pd.DataFrame(metrics_data).set_index("Metric").T
    print(df)

    df.to_csv("evals/evaluation_metrics_topological.txt", sep='\t', index=False)
