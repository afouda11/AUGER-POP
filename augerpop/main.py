import numpy as np
from io import StringIO
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import augerpop.utils as ut

def auger_run(ar_options):

	names 	 		 = ar_options["names"]
	core_orb 		 = ar_options["core_orb"]
	initial_states 	 = ar_options["initial_states"] 
	initial_index  	 = ar_options["initial_index"]
	final_states     = ar_options["final_states"] 
	atom_col         = ar_options["atom_col"]
	n_core_init      = ar_options["n_core_init"] 
	n_core_final     = ar_options["n_core_final"] 
	final_state_spin = ar_options["final_state_spin"]
	DIAG 			 = ar_options["DIAG"]
	mull_print_long  = ar_options["mull_print_long"]
	e_type 			 = ar_options["e_type"]
	E_shift 		 = ar_options["E_shift"]
	norm 			 = ar_options["norm"]
	fwhm 			 = ar_options["fwhm"]
	prev_step 		 = ar_options["prev_step"]
	n_prev_step 	 = ar_options["n_prev_step"]
	E 				 = ar_options["E"]
	dofit 			 = ar_options["dofit"]
	nprocs 			 = ar_options["nprocs"]

	for i,ival in enumerate(names):

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
			print(ival)
			print(jval)
			T[jval], roots = ut.auger_calc(ival, jval, initial_states[i][j], final_states[i][j],
			atom_col[i][j], n_core_init[i][j], n_core_final[i][j], final_state_spin[i][j], DIAG[i][j], 
			mull_print_long[i][j], nprocs)    

			for n in range(initial_states[i][j]):
				np.savetxt("outputs/amplitudes/T_"+str(jval)+"_"+str(ival)+"_spec_"+str(n)+".txt",T[jval][n])
			np.savetxt("outputs/roots/root_"+str(ival)+"_spec.txt",roots)

			if prev_step[i][j] != None: # normalize each dch_2 to the dch_1 intensities	
				T_prev = np.loadtxt("outputs/amplitudes/T_"+str(prev_step[i][j])+"_spec_"+str(0)+".txt")
				for n in range(initial_states[i][j]):
#					print(n)
					Norm = np.sum(T[jval][n]) / T_prev[n]
					T[jval][n] /= Norm
					T[jval][n] *= n_prev_step[i][j] #currently just uses the first DCH to normalize and x N_{DCH}
#					print(T_prev[n])
#					print(np.sum(T[jval][n]))
# 			if prev_step[i][j] != None: # normalize each dch_2 to the dch_1 intensities
# 				T_prev = []
# 				#T_norm = np.zeros(final_states[iV][j])
# 				T_norm = []
# 				for n in range(initial_states[i][j]):
# 					for k in range(n_prev_step[i][j]):
# 						#normalize with respect to each initial state of the previous step and sum
# 						T_prev = np.loadtxt("outputs/amplitudes/T_"+str(prev_step[i][j])+"_spec_"+str(k)+".txt")
# 						Norm = np.sum(T[jval][n]) / T_prev[n]
# 						T_norm.append(T[jval][n] / Norm)
# 					T[jval][n] = T_norm

			E_init[jval]  = get_energies(ival, jval, "init",  e_type[i][j], initial_index[i][j], 
							initial_states[i][j], final_states[i][j])
			E_final[jval] = get_energies(ival, jval, "final",  e_type[i][j], initial_index[i][j], 
	                        initial_states[i][j], final_states[i][j])

# 			if prev_step[i][j] != None: # normalize each dch_2 to the dch_1 intensities
# 				tmp = E_final[jval]
# 				E_final[jval] = np.repeat(tmp, n_prev_step[i][j])

			for n in range(initial_states[i][j]):
				T_all[jval] = np.append([T_all[jval]], [T[jval][n]])
				if E == "KE":
					E_x[jval] = np.append([E_x[jval]], [(-E_final[jval] + E_init[jval][n])*27.2114])
					#E_x[jval] = np.append([E_x[jval]], [E_final[jval]])
				if E == "BE":
					GE = get_energies(ival, jval, "ground", e_type[i][j])
					E_x[jval] = np.append([E_x[jval]], [((-1 * GE) + E_final[jval])*27.2114])

			E_x[jval] += E_shift[i][j]

			if prev_step[i][j] == None: # normalize each dch_2 to the dch_1 intensities	
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

def get_energies(ival, jval, ground_init_final, e_type, inital_index=None, initial_states=None, final_states=None):
	
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
		if e_type == "raspt2":
			E_line = lines[E_line_index[0]+1]
			c = StringIO(E_line)
			E = np.loadtxt(c, usecols = 6)
		if e_type == "rasscf":
			E_line = lines[E_line_index[0]+3]
			c = StringIO(E_line)
			E = np.loadtxt(c, usecols = 7)

# 	if ground_init_final == "init":
# 		if inital_index != None:
# 			for i,val in enumerate(inital_index):
# 				E_line = lines[E_line_index[1]+1+val]
# 				c = StringIO(E_line)
# 				if e_type == "raspt2":
# 					E.append(np.loadtxt(c, usecols = 6))
# 				if e_type == "rasscf":
# 					E.append(np.loadtxt(c, usecols = 7))
# 		if inital_index == None:
# 			for i in range(initial_states):
# 				E_line = lines[E_line_index[1]+1+i]
# 				print("hey")
# 				print(E_line)
# 				c = StringIO(E_line)
# 				if e_type == "raspt2":
# 					E.append(np.loadtxt(c, usecols = 6))
# 				if e_type == "rasscf":
# 					E.append(np.loadtxt(c, usecols = 7))
# 		E = np.array(E, ndmin=1)
	if ground_init_final == "init":
		for i,val in enumerate(inital_index):
			if i < 99:
				if e_type == "raspt2":
					E_line = lines[E_line_index[1]+1+val]
					c = StringIO(E_line)
					E.append(np.loadtxt(c, usecols = 6))
				if e_type == "rasscf":
					E_line = lines[E_line_index[1]+3+val]
					c = StringIO(E_line)
					E.append(np.loadtxt(c, usecols = 7))
			if i >= 99: 
				if e_type == "raspt2":
					E_line = lines[E_line_index[1]+1+val]
					c = StringIO(E_line)
					E.append(np.loadtxt(c, usecols = 5))
				if e_type == "rasscf":
					E_line = lines[E_line_index[1]+3+val]
					c = StringIO(E_line)
					E.append(np.loadtxt(c, usecols = 6))
		E = np.array(E, ndmin=1)

	if ground_init_final == "final":
		for i in range(final_states):
			if i < 99:
				if e_type == "raspt2":
					E_line = lines[E_line_index[2]+1+i]
					c = StringIO(E_line)
					E.append(np.loadtxt(c, usecols = 6))
				if e_type == "rasscf":
					E_line = lines[E_line_index[2]+3+i]
					c = StringIO(E_line)
					E.append(np.loadtxt(c, usecols = 7))
			if i >= 99: 
				if e_type == "raspt2":
					E_line = lines[E_line_index[2]+1+i]
					c = StringIO(E_line)
					E.append(np.loadtxt(c, usecols = 5))
				if e_type == "rasscf":
					E_line = lines[E_line_index[2]+3+i]
					c = StringIO(E_line)
					E.append(np.loadtxt(c, usecols = 6))
		E = np.array(E)
	return E


