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
from tensorflow.keras.layers import Input, concatenate, Dense, Dropout, Activation, Flatten, Reshape, BatchNormalization 
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, AveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.initializers import GlorotUniform
from keras import layers, regularizers


def model(p, L, k):

    # ---------- Load labels ----------
    comps_df = pd.read_csv('comp_17k_list.txt', sep='\s+', header = None, names = ['PDB_IDs'])
    print(comps_df)
    New_Set_labels = pd.read_csv('17k_CE_labels.txt', sep=',', header = None, names = ['PDB_IDs', 'CE'])
    New_Set_labels = New_Set_labels.drop_duplicates(subset=['PDB_IDs']).reset_index(drop=True)
    print(New_Set_labels)

    y_df = pd.merge(comps_df, New_Set_labels, on='PDB_IDs', how='inner')
    y_df = y_df.drop(['PDB_IDs'], axis = 1)
    y = y_df['CE'].to_numpy().astype(np.float32)
    print("Number of elements in labels:", y.shape)

    # ---------- Load Topological features ----------
    pickle_file = open("TDA_17k_MTF_features.pkl", "rb")
    features = []
    while True:
        try:
            features.append(pickle.load(pickle_file))
        except EOFError:
            break
    pickle_file.close()

    X_topological = np.asarray(features[0])
    print("Topological feature shape: ", X_topological.shape)

    # ---------- Load Electrostatic features ----------
    features_path = f"X/X_electrostatic_p{p}_L{L}.csv"
    X_electrostatic_df = pd.read_csv(features_path)
    X_electrostatic_df = X_electrostatic_df.drop(columns=X_electrostatic_df.columns[0])
    X_electrostatic_df = X_electrostatic_df.drop(['PDB_IDs'], axis = 1).to_numpy(dtype=np.float32)
    print("Electrostatic feature shape:", X_electrostatic_df.shape)

    X_topological_reshaped = X_topological.reshape(X_topological.shape[0], -1) # Flatten the image features
    X_combined = np.concatenate((X_topological_reshaped, np.array(X_electrostatic_df)), axis=1)

    print("Combined feature shape:", X_combined.shape)

    histories = []
    y_true = []
    y_pred = []
    evaluation_metrics = []
    
    X_train_combined, X_test_combined, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    best_mse = float('inf')  
    filename = None 
    best_scaler_X = None
    best_scaler_y = None 

    counter = 1
    
    for train_index, val_index in kf.split(X_train_combined):

        X_train_fold, X_val = X_train_combined[train_index], X_train_combined[val_index]
        y_train_fold, y_val = y_train[train_index], y_train[val_index]

        print(X_train_fold.shape)
        print(X_val.shape)
        print(y_train_fold.shape)
        print(y_val.shape)

        # Get topological features from fold data
        n_img_feats = X_topological_reshaped.shape[1]

        X_train_images_fold = X_train_fold[:, :n_img_feats].reshape(-1, 200, 12, 1) 
        X_val_images = X_val[:, :n_img_feats].reshape(-1, 200, 12, 1) 
        X_test_images = X_test_combined[:, :n_img_feats].reshape(-1, 200, 12, 1)
 
        
        # Center train images
        datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
        datagen.fit(X_train_images_fold)
        iterator = datagen.flow(X_train_images_fold, batch_size=len(X_train_images_fold), shuffle=False)
        X_train_centered = iterator.__next__()
        print(X_train_centered.shape, X_train_centered.mean(), X_train_centered.std())

        # Center val images
        datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
        datagen.fit(X_val_images)
        iterator = datagen.flow(X_val_images, batch_size=len(X_val_images), shuffle=False)
        X_val_centered = iterator.__next__()
        print(X_val_centered.shape, X_val_centered.mean(), X_val_centered.std())

        # Center test images
        datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
        datagen.fit(X_test_images)
        iterator = datagen.flow(X_test_images, batch_size=len(X_test_images), shuffle=False)
        X_test_centered = iterator.__next__()
        print(X_test_centered.shape, X_test_centered.mean(), X_test_centered.std())

        # Reshape 
        X_train_centered = X_train_centered.reshape((X_train_centered.shape[0], X_train_centered.shape[1], X_train_centered.shape[2]))
        X_test_centered = X_test_centered.reshape((X_test_centered.shape[0], X_test_centered.shape[1], X_test_centered.shape[2]))
        X_val_centered = X_val_centered.reshape((X_val_centered.shape[0], X_val_centered.shape[1], X_val_centered.shape[2]))

        scaler_X_fold = StandardScaler()
        X_train_ef_scaled = scaler_X_fold.fit_transform(X_train_fold[:, n_img_feats:])
        X_test_ef_scaled = scaler_X_fold.transform(X_test_combined[:, n_img_feats:])
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
            batch_size = 16, epochs=500, 
            validation_data = ([X_val_centered, X_val_ef_scaled], y_val_scaled),
            verbose=1)

        histories.append(history)

        y_val_pred_scaled = model_merged.predict([X_val_centered, X_val_ef_scaled])
        y_val_pred = scaler_y_fold.inverse_transform(y_val_pred_scaled)

        y_true.append(y_val)
        y_pred.append(y_val_pred)

        # Calculate evaluation metrics
        mse_scaled = mean_squared_error(y_val_scaled, y_val_pred_scaled)
        mape = mean_absolute_percentage_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)
        cc = np.corrcoef(np.array(y_val).squeeze(), np.array(y_val_pred).squeeze())[0, 1]
        
        print("Evaluation Metrics for Validation Data for k = ", counter)
        print(f"Scaled Mean Squared Error (MSE): {mse_scaled}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape}")
        print(f"R-squared (R2) Score: {r2}")
        print(f"Correlation Coefficient: {cc}")

        if mse_scaled < best_mse:
            best_mse = mse_scaled
            base_name = f"best_model_p{p}_L{L}_fold_{counter}"

            # Save the model
            os.makedirs("models", exist_ok=True)
            model_filename = os.path.join("models", f"{base_name}.keras")
            model_merged.save(model_filename)

            print(f"\n New best model saved with scaled MSE: {best_mse} at fold {counter} \n")

            filename = model_filename

        evaluation_metrics.append((mse_scaled, mape, r2, cc))
        counter = counter + 1
    
    # Calculate mean evaluation metrics across all folds
    mean_metrics = np.mean(evaluation_metrics, axis=0)
    print("mean evaluation metrics: ", mean_metrics)
    print(f"\n Best Model scaled MSE: {best_mse}, saved as {filename}\n ")

    # ---------- Evaluate best model on TEST ----------
    os.makedirs("plots/both", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("evals/both", exist_ok=True)

    best_model = load_model(filename)
    best_model.compile(optimizer=Adam(learning_rate=0.0001),
                       loss='mean_squared_error',       
                       metrics=['mse', 'mape']) 

    y_pred_scaled = best_model.predict([X_test_centered, X_test_ef_scaled])
    y_pred_inv = scaler_y_fold.inverse_transform(y_pred_scaled)
    y_test = np.array(y_test)

    test_mse_scaled = mean_squared_error(y_test_scaled, y_pred_scaled)
    test_rmse_scaled = np.sqrt(test_mse_scaled)
    test_mape = mean_absolute_percentage_error(y_test, y_pred_inv)
    test_cc = np.corrcoef(np.array(y_test).squeeze(), np.array(y_pred_inv).squeeze())[0, 1]
    test_r2 = r2_score(y_test, y_pred_inv)

    return histories, y_test, y_pred_inv, mean_metrics, test_mse_scaled, test_rmse_scaled, test_mape, test_cc, test_r2


def plot_mean_loss(histories):
    mean_train_loss = np.mean([h.history['loss'] for h in histories], axis=0)
    mean_val_loss = np.mean([h.history['val_loss'] for h in histories], axis=0)

    plt.figure(figsize=(9,6))
    plt.plot(mean_train_loss, label='Mean Training Loss')
    plt.plot(mean_val_loss, label='Mean Validation Loss')
    plt.xlabel('Epochs', fontsize = 20)
    plt.ylabel('Loss', fontsize = 20)
    plt.legend(fontsize=18)  
    plt.xticks(fontsize=14)  
    plt.yticks(fontsize=14)      
    plt.tight_layout()
    plt.savefig(f'plots/both/loss_both_p{p}_L{L}.png')

def plot_scatter(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred) 
    plt.figure(figsize=(9,6))
    plt.scatter(y_true, y_pred, marker='o', facecolors='none', edgecolors='b')
    lo = min(y_true.min(), y_pred.min()); hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], color='red')    
    plt.xlabel('Reference Values', fontsize=20)
    plt.ylabel('Predicted Values',fontsize = 20)
    plt.xticks(fontsize=14)  
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(f'plots/both/scatter_both_p{p}_L{L}.png')

  
if __name__ == "__main__":
    # Extract command-line arguments
    p = int(sys.argv[1])
    L = int(sys.argv[2])
    k = int(sys.argv[3])

    histories, y_true, y_pred, mean_metrics, test_mse_scaled, test_rmse_scaled, test_mape, test_cc, test_r2 = model(p, L, k)
    
    plot_mean_loss(histories)
    plot_scatter(y_true, y_pred)

    #  Save output data  
    os.makedirs("outputs", exist_ok=True)  
    pd.DataFrame(y_true).to_csv(f'outputs/y_true_both_p{p}_L{L}.csv', index=False)
    pd.DataFrame(y_pred).to_csv(f'outputs/y_pred_both_p{p}_L{L}.csv', index=False)

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

    # Create a DataFrame
    df = pd.DataFrame(metrics_data).set_index("Metric").T
    print(df)

    # Save the DataFrame to a text file
    df.to_csv("evals/both/evaluation_metrics_both_p" + str(p) + "_L" + str(L) + "_CE.txt", sep='\t', index=False)




