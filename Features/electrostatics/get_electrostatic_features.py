import numpy as np
import pandas as pd
import os 
import sys

def get_features(p, L):    


    csv_filename = "comp_17k_list.txt"
    # csv_filename = "comp_test_protein.txt"

    comps = pd.read_csv(csv_filename, header=None, names = ['PDB_IDs'])

    # Save protein IDs
    protein_IDs = comps['PDB_IDs']
    protein_IDs.reset_index(drop=True, inplace=True)
    print(f"\nprotein_IDs shape:{protein_IDs.shape} \n")

    comps['FilePath'] = '/users/atalha/Home/BioPhysics_DNN/v2020_pdb_pqr_dataSet/' + comps['PDB_IDs'] + "/" + comps['PDB_IDs']+'.pqr'

    # Make list of file paths and protein IDs 
    comps["e_features_commands"] = './a.out ' + comps["FilePath"] + ' 0.0 ' + str(p) + ' ' + str(L)

    features = []
    work_dir = "/users/atalha/Home/BioPhysics_DNN/Features/pro/e-features_geng"
    os.chdir(work_dir)
    os.system('make clean')
    os.system('make')

    features = []
    for cmd in comps["e_features_commands"]:
        print(f"Running....: {cmd}")
        os.system(cmd)
        feature  = np.loadtxt('efeature.txt')
        features.append(feature)
    
    # Clean build files and go back
    os.system('make clean')
    os.chdir("/users/atalha/Home/BioPhysics_DNN/Features/pro")

    # Convert features to numpy array
    X_electrostatic = np.vstack(features)
    X_electrostatic_df = pd.DataFrame(X_electrostatic)
    print("Raw feature shape:", X_electrostatic_df.shape) 

    X_electrostatic_df.insert(0, 'PDB_IDs', comps['PDB_IDs'])
    print("Feature matrix shape:", X_electrostatic_df.shape)  
    
    os.makedirs('X', exist_ok=True)
    X_electrostatic_df.to_csv('X/X_electrostatic_p' + str(p) + '_L' + str(L) + '.csv')
    print("Saved feature file successfully!")

if __name__ == "__main__":
    p = int(sys.argv[1])
    L = int(sys.argv[2])
    get_features(p, L)