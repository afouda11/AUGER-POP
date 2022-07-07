import numpy as np
from io import StringIO
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import multiprocessing
import itertools
from functools import partial	

def split(word):
    return [char for char in word]

def gaussian1D(yo,xo,x,d):
    return yo * (np.exp(-1 * (((x - xo) / d) ** 2)))

def linefit(x, y, fwhm, norm):
    
	sig = fwhm / (2 * np.sqrt(2*np.log(2)))
	xmin  = int(x.min())-1
	xmax  = int(x.max())+1
	#xmin  = int(x.min())
	#xmax  = int(x.max())
	xbase = np.linspace(xmin, xmax, 1001)
#print(xbase)
	yfit  = np.zeros(len(xbase))
	for i,val in enumerate(y):
		yfit += gaussian1D(val, x[i], xbase, sig[i]) 
	if norm == True:
		yfit = yfit / yfit.max()
	return np.column_stack((xbase, yfit))

def ci_vec_read(final_init, CI, log_lines, n_states):

	CI_start_index = []
	CI_end_index   = []
	for index,line in enumerate(log_lines):
		if "has been made, which may change the order of the CSFs." in line:
			CI_start_index.append(index+2)
		if "Natural orbitals and occupation numbers for root  1" in line:
			CI_end_index.append(index)

	index_vec = []
	root_vec = []
	count = 0
	if final_init == "init":#no intial state selectivity, need to edit log file
		lines = log_lines[CI_start_index[1]:CI_end_index[1]]
	if final_init == "final":
		lines = log_lines[CI_start_index[2]:CI_end_index[2]]

	for index,line in enumerate(lines):
		if 'printout of CI-coefficients larger than' in line:
			index_vec.append(index)
			count += 1
			root_vec.append(count)
	length_vec = []
	for i in range(len(index_vec)-1):
		length_vec.append(index_vec[i+1] - index_vec[i])
	length_vec.append(len(lines) - max(index_vec))
	civec = {}
	csf   = {}
	for n,val in enumerate(index_vec):
		civec[n] = []
		csf[n]   = []
		for i in range(val+3,val+length_vec[n]-1):      
			c = StringIO(lines[i])
			if CI == "coef":
				civec[n].append(np.loadtxt(c, usecols = 2))
			if CI == "weight":
				civec[n].append(np.loadtxt(c, usecols = 3))
			csf[n].append(lines[i].split()[1])
	return csf, civec, root_vec
	

def auger_calc(name, hole, n_init_states, n_final_states, 
			   atom_col, n_core_init, n_core_final, final_state_spin, 
			   CI, diag, mull_print_long, nprocs):

	log_file = open("inputs/"+str(hole)+"_"+str(name)+".log", 'r')
	log_lines = log_file.readlines()

	csf_init,  civec_init,  root_vec_init  = ci_vec_read("init",  CI, log_lines, n_init_states)
	csf_final, civec_final, root_vec_final = ci_vec_read("final", CI, log_lines, n_final_states)
	
# 	mull_pop_file = open(mullpop_file_name, 'r')
# 	lines = mull_pop_file.readlines()   
	pop_list = []
	grid_it  = []
	for index,line in enumerate(log_lines):
		if "Charges per occupied MO" in line:
			grid_it.append(index)	
	#print(grid_it)
	for index,line in enumerate(log_lines[grid_it[0]:]):#just need to start from first grid_it calc
		if mull_print_long == False:
			if "Mulliken charges per centre" in line:#core-hole atom has to be in the first 12 in the xyz
				total_mo_line = log_lines[grid_it[0]+index+4]
				c = StringIO(total_mo_line)
				pop_list.append(np.loadtxt(c, usecols = atom_col))
		if mull_print_long == True:
			if "Total  " in line or "total  " in line:#only works for less than 12 atoms in the molecule
				total_mo_line = log_lines[grid_it[0]+index]
				c = StringIO(total_mo_line)
				pop_list.append(np.loadtxt(c, usecols = atom_col))
	if name == "teoe":
		for i,val in enumerate(pop_list):
			pop_list[i] = pop_list[i][0] + pop_list[i][1]
	mull_pop  = np.hsplit(np.array(pop_list),n_init_states)

	#paramlist = list(itertools.product(range(n_init_states),range(len(csf_final))))
	paramlist = list(itertools.product(range(n_init_states),range(n_final_states)))
	#print(name)
	#print(hole)
	pool = multiprocessing.Pool(nprocs)
	funcpool = partial(intensity_cal, n_init_states, csf_init, csf_final, n_core_init,
					   n_core_final, final_state_spin, mull_pop, civec_init, civec_final, diag)

	I = pool.map(funcpool,paramlist)
	I_dict = {}
	index = 0
	for n in range(n_init_states):
		I_dict[n] = np.zeros(n_final_states)
		for i in range(n_final_states):
			I_dict[n][i] = I[index]
			index += 1

	#print(I)

	return I_dict, root_vec_final

def intensity_cal(n_init_states, csf_init, csf_final, n_core_init, n_core_final,
				final_state_spin, mull_pop, civec_init, civec_final, DIAG, params):

	n = params[0]
	i = params[1]
	Ini = 0.0
	t = []
	C = []
	for m in range(len(csf_init[n])):
		orbs_init = split(csf_init[n][m])
		for j in range(len(csf_final[i])):
			orbs_final = split(csf_final[i][j])
			wv = []
			count = 0
			for k,orb_init in enumerate(orbs_init[n_core_init:]):
				orb_final = orbs_final[k+n_core_final]
				if orb_init != orb_final:
		
					if orb_init == "2" and (orb_final == "u" or orb_final == "d"):
						wv.append(k)
						count += 1
					if orb_init == "2" and orb_final == "0":
						wv.append(k) 
						count =3
					if (orb_init == "u" or orb_init == "d") and orb_final == "0":
						wv.append(k)
						count += 1
					#if the number electrons increases then we ignore it
					if (orb_init == "u" or orb_init == "d") and orb_final == "2":
						count=5
					if (orb_init == "0") and (orb_final == "u" or orb_final == "d"):
						count=5
					if (orb_init == "0") and orb_final == "2":
						count=5
				else:
					continue
			if count == 2:
			#w != v
				if final_state_spin[i] == "s": 
					t.append(np.sqrt(0.5) * (mull_pop[n][wv[0]] + mull_pop[n][wv[1]]))
				if final_state_spin[i] == "d":
					t.append(np.sqrt(0.5) * (mull_pop[n][wv[0]] + mull_pop[n][wv[1]]))
					#t.append(np.sqrt(1.5) * (mull_pop[n][wv[0]] - mull_pop[n][wv[1]]))
				if final_state_spin[i] == "t":
					t.append(np.sqrt(1.5) * (mull_pop[n][wv[0]] - mull_pop[n][wv[1]]))
				C.append(civec_final[i][j] * civec_init[n][m])
			if count == 3:
			#w = v
				t.append(mull_pop[n][wv[0]])
				C.append(civec_final[i][j] * civec_init[n][m])
			#not valid
			if count == 5:
				continue
	if DIAG == True:	
		for diag in range(len(t)):

			Ini += (( np.absolute(t[diag])**2 ) * ( np.absolute(C[diag]) ** 2))
# 			for offdiag in range(len(t)):
# 				Ini += ((t[diag]*t[offdiag]) * (C[diag]*C[offdiag]))
		
		Ini =  (2 * np.pi * (Ini))
	
	if DIAG == False:
		for diag in range(len(t)):
 			Ini += ( np.absolute(t[diag]) * np.absolute(C[diag]) )

		Ini =  (2 * np.pi * (Ini**2))
	return Ini

