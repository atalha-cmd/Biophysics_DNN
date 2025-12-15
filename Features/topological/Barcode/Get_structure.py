def Get_structure(pdb_filepath, npz_filepath): # (input, input, output)	
	import os
	import sys
	import numpy as np
	import scipy as sp
	import sys

	typatm = np.dtype([('typ', 'S2'), ('pos', float, (3,)), ('rad', float), ('id', int)])

	# element specific persisent homolgy 

	pro_ele_list = ['C','N','O','S']
	aa_list = ['ALA','ARG','ASN','ASP','CYS','GLU','GLN','GLY','HIS','HSE','HSD','SEC',
	           'ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','PYL']


	def gettyp( rawtyp ):
		if not rawtyp[1].isalpha():
			typ = ' '+rawtyp[0]
		else:
			typ = rawtyp
		return typ


	propath = pdb_filepath
	outpath = npz_filepath

	profile = open(propath)
	pronum = 0
	lines = profile.read().splitlines()
	for line in lines:
		if line[0:4] == 'ATOM' and line[17:20] in aa_list:
			typ = line[12:14].replace(" ","")
			x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])

			if typ in pro_ele_list:
				pronum += 1
				
	PRO = np.zeros([pronum], dtype = typatm)

	j = 0
	for line in lines:
		if line[0:4] == 'ATOM' and line[17:20] in aa_list:
			typ = line[12:14]
			x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])

			if typ.replace(" ", "") in pro_ele_list:
				
				PRO[j]['typ'] = typ; PRO[j]['pos'][:] = np.array([x,y,z]);
				PRO[j]['id'] = -1; j+=1
				
	print("Length of Protein: ",len(PRO))

	outfile = open(outpath, 'wb')
	np.savez(outfile,PRO=PRO)
	outfile.close()
