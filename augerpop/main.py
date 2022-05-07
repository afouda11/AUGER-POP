import numpy as np
from io import StringIO
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import augerpop.utils as ut

def auger_run(steps, core_orb, initial_states, initial_index, final_states, atom_col, min_mo_init, lumo_index_init, min_mo_final, lumo_index_final, final_state_spin, CI, E, DIAG, e_type, E_shift=0.0, dofit=True, norm=None, fwhm=None, prev_step=None, nprocs=1):

	for i,ival in enumerate(steps):

		E_init     = dict.fromkeys(core_orb, [])
		E_final    = dict.fromkeys(core_orb, []) 
		T          = dict.fromkeys(core_orb, {}) 
		T_all      = dict.fromkeys(core_orb, []) 
		E_x        = dict.fromkeys(core_orb, []) 
		spec       = dict.fromkeys(core_orb, []) 
		spec_nofit = dict.fromkeys(core_orb, []) 
		ordered    = dict.fromkeys(core_orb, []) 

		cwd = os.getcwd()
		if os.path.isdir(str(cwd)+"/outputs/amplitudes") == False:
			os.mkdir(str(cwd)+"/outputs/amplitudes")
		if os.path.isdir(str(cwd)+"/outputs/roots") == False:
			os.mkdir(str(cwd)+"/outputs/roots")

		for j,jval in enumerate(core_orb):

			T[jval], roots = ut.auger_calc("inputs/CI_vecs_"+str(jval)+"_"+str(ival),
			"inputs/"+str(jval)+"_"+str(ival)+"_mull_pop.log", jval, ival, initial_states[i][j], final_states[i][j],
			atom_col[i][j],min_mo_init[i][j], lumo_index_init[i][j], min_mo_final[i][j], lumo_index_final[i][j],final_state_spin[i][j], CI[i][j], DIAG[i][j], nprocs)    

			for n in range(initial_states[i][j]):
				np.savetxt("outputs/amplitudes/T_"+str(jval)+"_"+str(ival)+"_spec_"+str(n)+".txt",T[jval][n])
			np.savetxt("outputs/roots/root_"+str(ival)+"_spec.txt",roots)

			if prev_step[i][j] != None: # normalize each dch_2 to the dch_1 intensities	
				T_prev = np.loadtxt("outputs/amplitudes/T_"+str(jval)+"_"+str(prev_step[i][j])+"_spec_"+str(0)+".txt")
				for n in range(initial_states[i][j]):
					print(n)
					Norm = np.sum(T[jval][n]) / T_prev[n]
					T[jval][n] /= Norm

			GE = get_energies(ival, jval, "ground", e_type[i][j])
			E_init[jval]  = get_energies(ival, jval, "init",  e_type[i][j], initial_index[i][j], final_states[i][j])
			E_final[jval] = get_energies(ival, jval, "final", e_type[i][j], initial_index[i][j], final_states[i][j])
	
			for n in range(initial_states[i][j]):
				T_all[jval] = np.append([T_all[jval]], [T[jval][n]])
				if E == "KE":
					E_x[jval] = np.append([E_x[jval]], [(-E_final[jval] + E_init[jval][n])*27.2114])
					#E_x[jval] = np.append([E_x[jval]], [E_final[jval]])
				if E == "BE":
					E_x[jval] = np.append([E_x[jval]], [((-1 * GE) + E_final[jval])*27.2114])

			E_x[jval] += E_shift[i][j]

			ordered[jval]  = np.column_stack((np.repeat(roots, initial_states[i][j]), E_x[jval], T_all[jval]))
			ordered[jval]  = ordered[jval][ordered[jval][:, 2].argsort()[::-1]]
			np.savetxt("outputs/roots/max_roots_"+str(jval)+"_"+str(ival)+".txt", ordered[jval])
		
			spec_nofit[jval] = np.column_stack((E_x[jval], T_all[jval]))

			if norm[i][j] == True:
				spec_nofit[jval][:,1] /= max(spec_nofit[jval][:,1])
			if DIAG[i][j] == False and norm[i][j] == False:
				np.savetxt("outputs/spec_nofit_"+str(jval)+"_"+str(ival)+"_full.txt", spec_nofit[jval])
			if DIAG[i][j] == True and norm[i] == False:
				np.savetxt("outputs/spec_nofit_"+str(jval)+"_"+str(ival)+"_diag.txt", spec_nofit[jval])
			if DIAG[i][j] == False and norm[i][j] == True:
				np.savetxt("outputs/spec_nofit_"+str(jval)+"_"+str(ival)+"_full_norm.txt", spec_nofit[jval])
			if DIAG[i][j] == True and norm[i][j] == True:
				np.savetxt("outputs/spec_nofit_"+str(jval)+"_"+str(ival)+"_diag_norm.txt", spec_nofit[jval])

			if dofit == True:
				spec[jval] = ut.linefit(spec_nofit[jval][:,0], spec_nofit[jval][:,1], fwhm[i][j], norm[i][j])
				if DIAG[i][j] == False and norm[i][j] == False:
					np.savetxt("outputs/spec_"+str(jval)+"_"+str(ival)+"_full.txt" ,spec[jval])	
				if DIAG[i][j] == True and norm[i][j] == False:
					np.savetxt("outputs/spec_"+str(jval)+"_"+str(ival)+"_diag.txt" ,spec[jval])	
				if DIAG[i][j] == False and norm[i][j] == True:
					np.savetxt("outputs/spec_"+str(jval)+"_"+str(ival)+"_full_norm.txt" ,spec[jval])	
				if DIAG[i][j] == True and norm[i][j] == True:
					np.savetxt("outputs/spec_"+str(jval)+"_"+str(ival)+"_diag_norm.txt" ,spec[jval])	
		
	return

def get_energies(ival, jval, ground_init_final, e_type, inital_index=None, final_states=None):
	
	log_file = open("inputs/"+str(jval)+"_"+str(ival)+".log", 'r')
	lines = log_file.readlines()
	E_line_index = []
	for index,line in enumerate(lines):
		if e_type == "rasscf":
			if "Final state energy(ies):" in line:
				E_line_index.append(index)
		if e_type == "raspt2":
			if "Total CASPT2 energies:" in line:
				E_line_index.append(index)

	E = []
	if ground_init_final == "ground":
		E_line = lines[E_line_index[0]+1]
		c = StringIO(E_line)
		if e_type == "raspt2":
			E = np.loadtxt(c, usecols = 6)
		if e_type == "rasscf":
			E = np.loadtxt(c, usecols = 7)

	if ground_init_final == "init":
		for i,val in enumerate(inital_index):
			E_line = lines[E_line_index[1]+1+val]
			c = StringIO(E_line)
			if e_type == "raspt2":
				E.append(np.loadtxt(c, usecols = 6))
			if e_type == "rasscf":
				E.append(np.loadtxt(c, usecols = 7))
		E = np.array(E, ndmin=1)

	if ground_init_final == "final":
		for i in range(final_states):
			E_line = lines[E_line_index[2]+1+i]
			c = StringIO(E_line)
			if e_type == "raspt2":
				E.append(np.loadtxt(c, usecols = 6))
			if e_type == "rasscf":
				E.append(np.loadtxt(c, usecols = 7))
		E = np.array(E)
	return E


