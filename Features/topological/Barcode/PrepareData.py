def PrepareData(pro_PH_filepath, pro_hydro_PH_filepath, npy_filepath):
    import os
    import numpy as np
    import math
    import sys

    rs = 0.25
    thr = 50.0
    lth = int(np.rint(thr/rs))
    small = 0.0001

    ProPHFile = pro_PH_filepath
    ProPHHydroFile = pro_hydro_PH_filepath
    OutFile = npy_filepath

    # 1. Count of birth in each interval
    # 2. Count of death in each interval
    # 3. Count of bars that span over each interval
    # Orders: Pro Betti-1 + Pro Betti-2 

    feature_i_1_2 = np.zeros([lth,6], float)

    # Process the ProPHHydro file
    with open(ProPHHydroFile, 'r') as infile:
        lines = infile.read().splitlines()
        for line in lines:
            dim, b, d = line.split()
            dim = int(dim)
            b = float(b)
            d = float(d)

            # Filter out irrelevant data
            if d - b < small or dim == 0 or b > thr or d > thr:
                continue

            # Calculate birth interval index and update birth count
            intid = int(math.floor(b / rs))
            fid = (dim - 1) * 3  # Column index for birth counts
            feature_i_1_2[intid, fid] += 1.0
            
            # Calculate death interval index and update death count
            intid = int(math.floor(d / rs))
            fid = (dim - 1) * 3 + 1  # Column index for death counts
            feature_i_1_2[intid, fid] += 1.0
            
            # Calculate bar count for intervals spanning birth to death
            bintid = int(math.floor(b / rs))
            dintid = int(math.floor(d / rs))
            fid = (dim - 1) * 3 + 2  # Column index for bar counts
            feature_i_1_2[bintid:dintid + 1, fid] += 1.0


    feature_all_i_1_2 = np.zeros([lth,6], float)

    # Process the ProPH file
    with open(ProPHFile, 'r') as infile:
        lines = infile.read().splitlines()
        for line in lines:
            dim, b, d = line.split()
            dim = int(dim)
            b = float(b)
            d = float(d)

            # Filter out irrelevant data
            if d - b < small or dim == 0 or b > thr or d > thr:
                continue

            # Calculate birth interval index and update birth count
            intid = int(math.floor(b / rs))
            fid = (dim - 1) * 3  # Column index for birth counts
            feature_all_i_1_2[intid, fid] += 1.0
            
            # Calculate death interval index and update death count
            intid = int(math.floor(d / rs))
            fid = (dim - 1) * 3 + 1  # Column index for death counts
            feature_all_i_1_2[intid, fid] += 1.0
            
            # Calculate bar count for intervals spanning birth to death
            bintid = int(math.floor(b / rs))
            dintid = int(math.floor(d / rs))
            fid = (dim - 1) * 3 + 2  # Column index for bar counts
            feature_all_i_1_2[bintid:dintid + 1, fid] += 1.0

    feature = np.concatenate((feature_i_1_2, feature_all_i_1_2), axis=1)
    outfile = open(OutFile, 'wb')
    np.save(outfile, feature)
    outfile.close()
