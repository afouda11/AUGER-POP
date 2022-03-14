import numpy as np
from io import StringIO
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import augerpop.utils as ut

def auger_run(steps, core_orb, ras_nas, initial_states, atom_col, min_mo_init, 
			  lumo_index_init, min_mo_final, lumo_index_final, final_state_spin,
			  GE, E, E_shift, dofit, norm=None, fwhm=None, prev_step=None):

	for i,ival in enumerate(steps):

		E_init     = dict.fromkeys(core_orb, [])
		E_final    = dict.fromkeys(core_orb, []) 
		T          = dict.fromkeys(core_orb, {}) 
		T_all      = dict.fromkeys(core_orb, []) 
		E_x        = dict.fromkeys(core_orb, []) 
		spec       = dict.fromkeys(core_orb, []) 
		spec_nofit = dict.fromkeys(core_orb, []) 
		ordered    = dict.fromkeys(core_orb, []) 

		for j,jval in enumerate(core_orb):

			T[jval], roots = ut.auger_calc("inputs/CI_vecs_"+str(jval)+"_"+str(ival),
			"inputs/mull_pops_"+str(jval)+"_"+str(ival)+".txt", jval, ival, ras_nas[i], initial_states[i],
			atom_col[j],min_mo_init[i], lumo_index_init[i], min_mo_final[i], lumo_index_final[i], final_state_spin)    

			for n in range(initial_states[i]):
				np.savetxt("outputs/T_"+str(jval)+"_"+str(ival)+"_spec_"+str(i)+".txt",T[jval][n])
			np.savetxt("outputs/root_"+str(ival)+"_spec.txt",roots)

			if prev_step[i] != None: # normalize each dch_2 to the dch_1 intensities	
				T_prev = np.loadtxt("outputs/T_"+str(jval)+"_"+str(prev_step[i])+"_spec_"+str(0)+".txt")
				for n in range(initial_states[i]):
					norm = np.sum(T[jval][n]) / T_prev[n]
					T[jval][n] /= norm

			E_init[jval]  = np.array(np.loadtxt("inputs/energies_"+str(jval)+"_"+str(ival)+"_init.txt"), ndmin=1)
			E_final[jval] = np.loadtxt("inputs/energies_"+str(jval)+"_"+str(ival)+"_final.txt")
			for n in range(initial_states[i]):
				T_all[jval] = np.append([T_all[jval]], [T[jval][n]])
				if E == "KE":
					E_x[jval] = np.append([E_x[jval]], [(-E_final[jval] + E_init[jval][n])*27.2114])
					#E_x[jval] = np.append([E_x[jval]], [E_final[jval]])
				if E == "BE":
					E_x[jval] = np.append([E_x[jval]], [(-GE + E_final[jval])*27.2114])

			E_x[jval] += E_shift
			ordered[jval]  = np.column_stack((np.repeat(roots, initial_states[i]), E_x[jval], T_all[jval]))
			ordered[jval]  = ordered[jval][ordered[jval][:, 2].argsort()[::-1]]
			np.savetxt("outputs/max_roots_"+str(jval)+"_"+str(ival)+".txt", ordered[jval])
		
			spec_nofit[jval] = np.column_stack((E_x[jval], T_all[jval]))
			if norm[i] == True:
				spec_nofit[jval][:,1] /= max(spec_nofit[jval][:,1])
			np.savetxt("outputs/spec_nofit_"+str(jval)+"_"+str(ival)+".txt", spec_nofit[jval])

			if dofit == True:
				spec[jval] = ut.linefit(spec_nofit[jval][:,0], spec_nofit[jval][:,1], fwhm[i], norm[i])
				np.savetxt("outputs/spec_"+str(jval)+"_"+str(ival)+".txt" ,spec[jval])	
		
	return
