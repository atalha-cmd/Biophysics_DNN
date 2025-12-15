def Run_alpha(npz_filepath, pro_PH_filepath): 
    import os, sys, gudhi
    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt 

    # Documentation: https://gudhi.inria.fr/python/latest/simplex_tree_user.html 

    InputFile = npz_filepath
    ProOutFile = pro_PH_filepath

    structfile = open(InputFile)
    # print("hello")
    # print('structfile',structfile)
    structfile = "complex.npz"
    data = np.load(structfile)
    PRO = data['PRO']

    pro_pt = []
    for atom in PRO:
    	pos = []
    	pos.extend([atom['pos'][0], atom['pos'][1], atom['pos'][2]])
    	pro_pt.append(pos)

    print("Length of Protein:",len(pro_pt))

    # Protein (alpha compelx is Cech complex)

    alpha_complex = gudhi.AlphaComplex(points=pro_pt)
    simplex_tree = alpha_complex.create_simplex_tree()
    PH = simplex_tree.persistence()

    # Write results to the output file
    with open(ProOutFile, 'w') as pro_out_file:
        for simplex in PH:
            # print("simplex")
            # print(simplex)
            dim, b, d = int(simplex[0]), float(simplex[1][0]), float(simplex[1][1])
            if d-b >= 0.1:
                pro_out_file.write(f'{dim} {b} {d}\n')


