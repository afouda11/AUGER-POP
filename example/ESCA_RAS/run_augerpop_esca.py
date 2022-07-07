import augerpop
import augerpop.main
import numpy as np	

ar_options = {}
ar_options["names"]		   	 	= ["ground_opt"]
ar_options["core_orb"] 	   	 	= ["cf3", "coo"]
ar_options["initial_states"] 	= [[1,1]]
ar_options["initial_index"] 	= [[[0],[0]]]
ar_options["final_states"] 	 	= [[50,50]]
ar_options["atom_col"] 	   	 	= [[3,3]]
ar_options["n_core_init"]    	= [[1,1]] 
ar_options["n_core_final"]    	= [[1,1]] 
ar_options["final_state_spin"] 	= [[np.repeat("d", 50),np.repeat("d", 50)]]
ar_options["E"] 				= "BE"
ar_options["DIAG"]			 	= [[False, False]]
ar_options["mull_print_long"]   = [[False,False]]
ar_options["e_type"]			= [["raspt2", "raspt2"]]
ar_options["E_shift"] 		 	= [[0.0, 0.0]]
ar_options["dofit"] 			= True
ar_options["norm"]  			= [[False, False]]
ar_options["fwhm"]  			= [[np.repeat(1.5, 50), np.repeat(1.5, 50)]] 
ar_options["prev_step"] 		= [[None,None]]
ar_options["n_prev_step"] 		= [[None,None]]
ar_options["nprocs"]			= 1

augerpop.main.auger_run(ar_options)
