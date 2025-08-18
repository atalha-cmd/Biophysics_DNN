import numpy as np
import pandas as pd
import shutil
import os 
import glob
import sys

def get_features(p, L):    

    comps = pd.read_csv("comp_test_protein.txt", header=None, names = ['PDB_IDs'])

    comps['FilePath'] = '/users/atalha/Biophysics_ML/Data/test_protein/' + comps['PDB_IDs'] + "/" + comps['PDB_IDs']+'.pqr'

    # Make list of file paths and protein IDs 
    comps["e_features_commands"] = './a.out ' + comps["FilePath"] + ' 0.0 ' + str(p) + ' ' + str(L)

    features = []
    dir = "/users/atalha/Biophysics_ML/Features/electrical/e-features_geng"
    os.chdir(dir)
    os.system('make clean')
    os.system('make')

    for command in comps["e_features_commands"]:
        os.system(command)
        print(command)
        # Read feature from efeature.txt
        feature  = np.loadtxt('efeatrue.txt')
        features.append(feature)
    os.system('make clean')
    dir = "/users/atalha/Biophysics_ML/Features/electrical"
    os.chdir(dir)

    # Convert features to numpy array
    X_electrostatic = np.vstack(features)
    X_electrostatic_df = pd.DataFrame(X_electrostatic)  
    X_electrostatic_df.reset_index(drop=True, inplace=True)
    print("X_df shape: ", X_electrostatic_df.shape)

    # Save protein IDs
    protein_IDs = comps['PDB_IDs']
    protein_IDs.reset_index(drop=True, inplace=True)
    print("protein_IDs shape:", protein_IDs.shape)
    
    X_electrostatic_df = pd.concat([protein_IDs, X_electrostatic_df], axis=1)
    print("X_df shape: ", X_electrostatic_df.shape)

    X_electrostatic_df.to_csv('X_electrostatic_p' + str(p) + '_L' + str(L) + '.csv')
    print(X_electrostatic_df.shape)

if __name__ == "__main__":
    # Extract command-line arguments
    p = int(sys.argv[1])
    L = int(sys.argv[2])
    
    # Call the function
    get_features(p, L)
