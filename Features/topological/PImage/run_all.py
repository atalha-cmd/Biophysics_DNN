# THIS CODE IS USED TO GENERATE TOPOLOGICAL FEATURES
import time
from Get_structure import Get_structure
from Run_alpha import Run_alpha
from Run_alpha_hydro import Run_alpha_hydro
from PrepareImage import PrepareImage
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import re
import pickle

csv_filename = "comp_test_protein.txt"

df = pd.read_csv(csv_filename, header=None, names=['PDB_IDs'])
print(df)
print(df['PDB_IDs'].shape)

new_set_features = []   
new_set_diagrams = []   
new_total_time = 0
counter = 0

for comp in df['PDB_IDs']:
    start_time = time.time()
    print(counter + 1)
    print(comp)

    work_dir = "/Users/atalha/Desktop/Topology/PImage/PD_PI/test_dataSet/" + comp

    os.chdir(work_dir)  # change to directory

    pdb_file = [x for x in os.listdir(work_dir) if x.endswith('.pdb')][0]

    Get_structure(pdb_file, 'complex.npz')
    Run_alpha('complex.npz', 'protein.PH')
    Run_alpha_hydro('complex.npz', 'protein_C-C.PH')

    feature, diagrams = PrepareImage('protein.PH', 'protein_C-C.PH', 'complex_digit.npy')

    print("feature size", feature.shape)
    new_set_features.append(feature)
    new_set_diagrams.append(diagrams)

    for f in ['complex.npz', 'protein.PH', 'protein_C-C.PH', 'complex_digit.npy']:
        if os.path.exists(f):
            os.remove(f)

    end_time = time.time()
    total_time = (end_time - start_time)
    print("total time: ", total_time, " seconds")
    new_total_time = new_total_time + total_time
    counter = counter + 1

print("test_total_time:", new_total_time)
print("Length of final shape:", len(new_set_features))

save_dir = "/Users/atalha/Desktop/Topology/PImage/PD_PI"
os.chdir(save_dir)  

# PImages
with open("TDA_test_PImage_features.pkl", "wb") as fp:   
    pickle.dump(new_set_features, fp)

with open("TDA_test_PDiagram_features.pkl", "wb") as fp:
    pickle.dump(new_set_diagrams, fp)
