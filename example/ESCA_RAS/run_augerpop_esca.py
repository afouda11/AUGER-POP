import augerpop
import augerpop.main
import numpy as np	

steps 		   	 = ["ground_opt"]
core_orb 	   	 = ["cf3", "coo"]
initial_states 	 = [[1,1]]
initial_index 	 = [[[0],[0]]]
final_states 	 = [[50,50]]
atom_col 	   	 = [[3,3]]
min_mo_init    	 = [[1,1]] 
max_mo_init    	 = [[19,19]]
min_mo_final 	 = [[1,1]] 
final_state_spin = [[np.repeat("d", 50),np.repeat("d", 50)]]
CI 				 = [["coef", "coef"]]
E 				 = "BE"
DIAG			 = [[False, False]]
e_type			 = [["raspt2", "raspt2"]]
E_shift 		 = [[0.0, 0.0]]
dofit 			 = True
norm  			 = [[False, False]]
fwhm  			 = [[np.repeat(1.5, 50), np.repeat(1.5, 50)]] 
prev_steps 		 = [[None,None]]
nprocs = 1

augerpop.main.auger_run(steps, core_orb, initial_states, initial_index, 
						final_states, atom_col, min_mo_init, max_mo_init, 
						min_mo_final, final_state_spin, CI, E, DIAG, e_type, 
						E_shift, dofit, norm, fwhm, prev_steps, nprocs)
