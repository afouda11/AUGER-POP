import numpy as np
from io import StringIO
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import multiprocessing as mp

def split(word):
    return [char for char in word]

def gaussian1D(yo,xo,x,d):
    return yo * (np.exp(-1 * (((x - xo) / d) ** 2)))

def pool_func(x, y, xbase, sig):
	return gaussian1D(x, y, xbase, sig) 

# def pool_func(*args):
# 	return gaussian1D(args[0], args[1], args[2], args[3])

def linefit(x, y, fwhm, norm):
    
	sig = fwhm / (2 * np.sqrt(2*np.log(2)))
	xmin  = int(x.min())-3
	xmax  = int(x.max())+3
	#xmin  = int(x.min())
	#xmax  = int(x.max())
	xbase = np.linspace(xmin, xmax, 101)
#print(xbase)
	yfit  = np.zeros(len(xbase))
	for i,val in enumerate(y):
		yfit += gaussian1D(val, x[i], xbase, sig) 
	if norm == True:
		yfit = yfit / yfit.max()
	return np.column_stack((xbase, yfit))

def ci_vec_read(civec_file_name, final_initial):
	if final_initial == "final":
		civec_file = open(str(civec_file_name)+"_final.txt", 'r')
	if final_initial == "initial":
		civec_file  = open(str(civec_file_name)+"_init.txt", 'r')
	lines = civec_file.readlines()
	index_vec = []
	root_vec = []
	count = 0
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
			civec[n].append(np.loadtxt(c, usecols = 3))
			csf[n].append(lines[i].split()[1])
	return csf, civec, root_vec
	

def auger_calc(civec_file_name, mullpop_file_name, hole, step, ras_nas, n_init_states, atom_col, min_mo_init,
lumo_index_init, min_mo_final, lumo_index_final, final_state_spin):

	csf_final, civec_final, root_vec_final = ci_vec_read(civec_file_name, "final")
	csf_init,  civec_init,  root_vec_init  = ci_vec_read(civec_file_name, "initial")
	
	mull_pop_file = open(mullpop_file_name, 'r')
	lines = mull_pop_file.readlines()   
	pop_list = []
	for index,line in enumerate(lines):
		if "Total   " in line or "total   " in line:
			c = StringIO(line)
			pop_list.append(np.loadtxt(c, usecols = atom_col))
	if step == "teoe":
		for i,val in enumerate(pop_list):
			pop_list[i] = pop_list[i][0] + pop_list[i][1]
	mull_pop  = np.hsplit(np.array(pop_list),n_init_states)

	I = {}
	for n in range(n_init_states):
		I[n] = np.zeros(len(csf_final))
		for i in range(len(csf_final)):
			t = []
			C = []
			#print("FINAL STATE: "+str(i))
			for m in range(len(csf_init[n])):
				orbs_init = split(csf_init[n][m])
				#print(orbs_init)
				for j in range(len(csf_final[i])):
					orbs_final = split(csf_final[i][j])
					#print(orbs_final)
					wv = []
					count = 0
					if ras_nas == "nas":
						for k,orb_init in enumerate(orbs_init[min_mo_init:lumo_index_init+1]):

							if orb_init != orbs_final[k+min_mo_final]:
					
								if orb_init == "2" and (orbs_final[k+min_mo_final] == "u" or orbs_final[k+min_mo_final] == "d"):
									wv.append(k)
									count += 1
								if orb_init == "2" and orbs_final[k+min_mo_final] == "0":
									wv.append(k) 
									count =3
								if (orb_init == "u" or orb_init == "d") and orbs_final[k+min_mo_final] == "0":
									wv.append(k)
									count += 1
								#if the number electrons increases then we ignore it
								if (orb_init == "u" or orb_init == "d") and orbs_final[k+min_mo_final] == "2":
									count=5
								if (orb_init == "0") and (orbs_final[k+min_mo_final] == "u" or orbs_final[k+min_mo_final] == "d"):
									count=5
								if (orb_init == "0") and orbs_final[k+min_mo_final] == "2":
									count=5
							else:
								continue
						#print(count)
						if count == 2:
						#w != v
							if final_state_spin[i] == "s": 
								#t = np.sqrt(0.5) * (mull_pop[n][wv[0]] + mull_pop[n][wv[1]])
								t.append(np.sqrt(0.5) * (mull_pop[n][wv[0]] + mull_pop[n][wv[1]]))
							if final_state_spin[i] == "d":
								#t = mull_pop[n][wv[0]] + mull_pop[n][wv[1]]
								t.append(mull_pop[n][wv[0]] + mull_pop[n][wv[1]])
							if final_state_spin[i] == "t":
								#t = np.sqrt(1.5) * (mull_pop[n][wv[0]] + mull_pop[n][wv[1]])
								t.append(np.sqrt(1.5) * (mull_pop[n][wv[0]] - mull_pop[n][wv[1]]))

							#I[n][i] += ( np.absolute(t)**2 ) * ( np.absolute(civec_final[i][j] * civec_init[n][m]) ** 2)
							C.append(civec_final[i][j] * civec_init[n][m])
						if count == 3:
						#w = v
							#t = mull_pop[n][wv[0]]
							t.append(mull_pop[n][wv[0]])
							C.append(civec_final[i][j] * civec_init[n][m])
				
							#I[n][i] += ( np.absolute(t)**2 ) * ( np.absolute(civec_final[i][j] * civec_init[n][m]) ** 2)
						#not valid
						if count == 5:
							continue
			#print(t)
			#print(C)
			for diag in range(len(t)):
				I[n][i] += (( np.absolute(t[diag])**2 ) * ( np.absolute(C[diag]) ** 2))
				for offdiag in range(len(t)):
					I[n][i] += ((t[diag]*t[offdiag]) * (C[diag]*C[offdiag]))
# 					for k,orb_init in enumerate(orbs_init[min_mo_init:lumo_index_init]):
# 
# 						if orb_init != orbs_final[k+min_mo_init]:
# 				
# 							if orb_init == "2" and (orbs_final[k+min_mo_init] == "u" or orbs_final[k+min_mo_init] == "d"):
# 								wv.append(k)
# 								count += 1
# 							if orb_init == "2" and orbs_final[k+min_mo_init] == "0":
# 								wv.append(k) 
# 								count =3
# 							if (orb_init == "u" or orb_init == "d") and orbs_final[k+min_mo_init] == "0":
# 								wv.append(k)
# 								count += 1
# 							if (orb_init == "u" or orb_init == "d") and orbs_final[k+min_mo_init] == "2":
# 								count=5
# 						else:
# 							continue
# 
# 					if count == 2:
# 					#spectator
# 						t = np.sqrt(0.5) * (mull_pop[n][wv[0]] + mull_pop[n][wv[1]])
# 					if count == 3:
# 					#spectator w = v
# 						t =  mull_pop[n][wv[0]]
# 					if count == 1:
# 					#participator
# 						t = np.sqrt(0.5) * (mull_pop[n][wv[0]] + mull_pop[n][lumo_index_final-1])
# 					if count == 0:
# 					#double participator
# 						t = np.sqrt(0.5) * (2 * mull_pop[n][lumo_index_final-1])
# 					#not valid
# 					if count == 5:
# 						t = 0.0	
# 					I[n][i] += ( np.absolute(t)**2 ) * ( np.absolute(civec_final[i][j] * civec_init[n][m]) ** 2)
# 
			I[n][i] *=  (2 * np.pi)
			#print(" ")

	return I, root_vec_final
