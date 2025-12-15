# THIS CODE IS USED TO GENERATE TOPOLOGICAL FEATURES

import time
from Get_structure import Get_structure
from Run_alpha import Run_alpha
from Run_alpha_hydro import Run_alpha_hydro
from PrepareData import PrepareData
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os 
import re
import pickle


# Read data from the CSV file into a DataFrame
# csv_filename = "comp_17k_list.txt"
csv_filename = "comp_test_protein.txt"

df = pd.read_csv(csv_filename, header = None, names = ['PDB_IDs'])
print(df)
print(df['PDB_IDs'].shape)

# list of output files
new_set_features = []
new_total_time = 0
counter = 0


for comp in df['PDB_IDs']:
  start_time = time.time()
  print(counter + 1)
  print(comp)

  # dir = "/users/atalha/Home/BioPhysics_DNN/v2020_pdb_pqr_dataSet/" + comp 
  dir = "/users/atalha/Home/BioPhysics_DNN/test_dataSet/" + comp

  os.chdir(dir) # change to directory
  pdb_file = [x for x in os.listdir(dir) if x.endswith('.pdb')][0]

  # Generate PH and MTF
  Get_structure(pdb_file,'complex.npz')
  Run_alpha('complex.npz', 'protein.PH')
  Run_alpha_hydro('complex.npz', 'protein_C-C.PH')
  PrepareData('protein.PH', 'protein_C-C.PH', 'complex_digit.npy')
  
  # Load and append feature
  feature = np.load('complex_digit.npy') 
  print("feature size", feature.shape)
  new_set_features.append(feature)

  # Delete temporary files
  for f in ['complex.npz', 'protein.PH', 'protein_C-C.PH', 'complex_digit.npy']:
      if os.path.exists(f):
          os.remove(f)

  # Computing Time
  end_time = time.time()
  total_time = (end_time - start_time)
  print("total time: ", total_time, " seconds")
  new_total_time = new_total_time + total_time
  counter = counter + 1

print("test_total_time:", new_total_time)
print("final shape:", len(new_set_features ))

# now save 
save_dir = "/users/atalha/Home/BioPhysics_DNN/Features/pro"
os.chdir(save_dir) # change to directory
with open("TDA_test_MTF_features.pkl", "wb") as fp:   #Pickling
    pickle.dump(new_set_features, fp)

