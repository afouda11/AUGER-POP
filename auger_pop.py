import numpy as np
from io import StringIO
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def split(word):
    return [char for char in word]

def gaussian1D(yo,xo,x,d):
    return yo * (np.exp(-1 * (((x - xo) / d) ** 2)))

def linefit(x, y, sig, norm):
        
    xmin  = int(x.min())-3
    xmax  = int(x.max())+3
    xdiff = ((xmax - xmin) * 1000) + 1
    xbase = np.linspace(xmin, xmax, xdiff)
    yfit  = np.zeros(len(xbase))
        
    for i,val in enumerate(y):
        xcent = x[i]
        line  = gaussian1D(val, xcent, xbase, sig)
            
        yfit = line + yfit
            
    if norm == True:
        yfit = yfit / yfit.max()
    
    return np.column_stack((xbase, yfit))

def auger_calc(civec_file_name, mullpop_file_name, method, hole, type_, n_init_states):
    civec_file = open(civec_file_name, 'r')
    lines = civec_file.readlines()
    index_vec = []
    root_vec = []
    count = 0
    for index,line in enumerate(lines):
        if method == 'rasscf':
            if 'printout of CI-coefficients larger than' in line:
                index_vec.append(index)
                count += 1
                root_vec.append(count)
        if method == 'mspt2':
            if 'The CI coefficients for the MIXED state' in line:
                index_vec.append(index)
                count += 1
                root_vec.append(count)
    length_vec = []
    for i in range(len(index_vec)-1):
        length_vec.append(index_vec[i+1] - index_vec[i])
    if method == 'rasscf' and hole == 'nc' and type_ == 'sch':     
        length_vec.append(12) # big hack for the last one
    if method == 'rasscf' and hole == 'nt' and type_ == 'sch':     
        length_vec.append(18) # big hack for the last one
    if method == 'mspt2' and hole == 'nc' and type_ == 'sch':     
        length_vec.append(34) # big hack for the last one
    if method == 'mspt2' and hole == 'nt' and type_ == 'sch':     
        length_vec.append(30) # big hack for the last one
    if method == 'rasscf' and hole == 'nt' and type_ == 'dch_1':     
        length_vec.append(47) # big hack for the last one
    if method == 'rasscf' and hole == 'nc' and type_ == 'dch_1':     
        length_vec.append(23) # big hack for the last one
    if method == 'rasscf' and hole == 'nt' and type_ == 'dch_2':     
        length_vec.append(70) # big hack for the last one
    if method == 'rasscf' and hole == 'nc' and type_ == 'dch_2':     
        length_vec.append(37) # big hack for the last one

    civec = {}
    csf   = {}
    for n,val in enumerate(index_vec):
        civec[n] = []
        csf[n]   = []
        if method == 'rasscf':
            for i in range(val+3,val+length_vec[n]-1):      
                c = StringIO(lines[i])
                civec[n].append(np.loadtxt(c, usecols = 2))
                csf[n].append(lines[i].split()[1])
        if method == 'mspt2':
            for i in range(val+7,val+length_vec[n]-1):       
                c = StringIO(lines[i])
                civec[n].append(np.loadtxt(c, usecols = 6))
                csf[n].append(lines[i].split()[5])

    mull_pop_file = open(mullpop_file_name, 'r')
    lines = mull_pop_file.readlines()   
    pop_list = []
    for index,line in enumerate(lines):
        if "Total   " in line:
            c = StringIO(line)
            if hole == 'nc':
                pop_list.append(np.loadtxt(c, usecols = 2))
            if hole == 'nt':
                pop_list.append(np.loadtxt(c, usecols = 1))

    mull_pop  = np.hsplit(np.array(pop_list),n_init_states)            
    T = {}
    for n in range(n_init_states):
        T[n] = np.zeros(len(csf))
        for i in range(len(csf)):
            for j in range(len(csf[i])):
                orbs = split(csf[i][j])

                wv = []
                count = 0
                if type_ == 'sch':
                    for k,orb in enumerate(orbs[1:9]):

                        if orb == "u" or orb == "d":
                            wv.append(k)
                            count += 1
                        elif orb == "0":
                            wv.append(k)
                            count =3

                    if count == 2:
#            print("spectator")
                        T[n][i] += ( (mull_pop[n][3+wv[0]] + mull_pop[n][3+wv[1]]) ** 2 ) * ( civec[i][j] ** 2 )
                    if count == 3:
#            print("spectator w = v")
                        T[n][i] += ( 2 * (mull_pop[n][3+wv[0]]) ** 2 ) * ( civec[i][j] ** 2 )
                    if count == 1:
#            print("participator")
                        T[n][i] += ( (mull_pop[n][3+wv[0]] + mull_pop[n][10]) ** 2 ) * ( civec[i][j] ** 2 )

                if type_ == 'dch_1':
                    for k,orb in enumerate(orbs[1:9]):
                        
                        if orb == "u" or orb == "d":
                            wv.append(k) 
                            count += 1
                        elif orb == "0":
                            wv.append(k)
                            count = 3

                    if count == 2:
#            print("spectator")
                        T[n][i] += ( (mull_pop[n][3+wv[0]] + mull_pop[n][3+wv[1]]) ** 2 ) * ( civec[i][j] ** 2 )
                    if count == 3:
#            print("spectator w = v")
                        T[n][i] += ( 2 * (mull_pop[n][3+wv[0]]) ** 2 ) * ( civec[i][j] ** 2 )
                    if count == 1:
#            print("participator")
                        T[n][i] += ( (mull_pop[n][3+wv[0]] + mull_pop[n][11]) ** 2 ) * ( civec[i][j] ** 2 )
                    if count == 0:
#            print("double participator")
                        T[n][i] += ( 2* (mull_pop[n][11]) ** 2 ) * ( civec[i][j] ** 2 )

                if type_ == 'dch_2':
                    t = 0
                    for k,orb in enumerate(orbs[1:9]):
                        
                        if orb == "2":
                            dum = 0        
                        if orb == "u" or orb == "d":
                            dum = 1
                        if orb == "0":
                            dum = 2

                        t += dum * mull_pop[n][3+k]
                    for k,orb in enumerate(orbs[9:11]):
                        
                        if orb == "0":
                            count += 0
                        if orb == "1":
                            count += 1
                        if orb == "2":
                            count += 2
                    if count == 2:
#            print("double spectator")
                        continue
                    if count == 1:
#            print("1 participator 1 spectator")
                        t += 1 * mull_pop[n][3+k]
                    if count == 0:
#            print("double participator")
                        t += 2 * mull_pop[n][3+k]
                    
                    T[n][i] += (t ** 2) * ( civec[i][j] ** 2 )
    return T, root_vec

#load_or_calc = "calc"
load_or_calc = "load"
# sch
sch_n_init_states = 1
if load_or_calc == "calc": 
    T_nc_sch, roots_sch = auger_calc('inputs/CI_vecs_nc_sch_final.txt', 'inputs/mo_mull_pops_nc_sch.txt', 'rasscf', 'nc', 'sch', sch_n_init_states)    
    T_nt_sch, roots_sch = auger_calc('inputs/CI_vecs_nt_sch_final.txt', 'inputs/mo_mull_pops_nt_sch.txt', 'rasscf', 'nt', 'sch', sch_n_init_states)    
    for i in range(sch_n_init_states):
        np.savetxt("outputs/T_nc_sch_spec_"+str(i)+".txt",T_nc_sch[i])
        np.savetxt("outputs/T_nt_sch_spec_"+str(i)+".txt",T_nt_sch[i])
    np.savetxt("outputs/root_sch_spec.txt",roots_sch)
if load_or_calc == "load":
    T_nc_sch = {} 
    T_nt_sch = {} 
    for i in range(sch_n_init_states):
        T_nc_sch[i] = np.loadtxt("outputs/T_nc_sch_spec_"+str(i)+".txt")
        T_nt_sch[i] = np.loadtxt("outputs/T_nt_sch_spec_"+str(i)+".txt")
    roots_sch   = np.loadtxt("outputs/root_sch_spec.txt")

energies_nc_sch_final = np.loadtxt('inputs/energies_nc_sch_final.txt')
energies_nt_sch_final = np.loadtxt('inputs/energies_nt_sch_final.txt')

energies_nc_sch_init = np.loadtxt('inputs/energies_nc_sch_init.txt')
energies_nt_sch_init = np.loadtxt('inputs/energies_nt_sch_init.txt')

KE_nc_sch = ((-1*energies_nc_sch_final) + energies_nc_sch_init)*27.2114
KE_nt_sch = ((-1*energies_nt_sch_final) + energies_nt_sch_init)*27.2114
spec_nc_sch = linefit(KE_nc_sch, T_nc_sch[0], 1.0, False)
spec_nt_sch = linefit(KE_nt_sch, T_nt_sch[0], 1.0, False)

max_roots_nc_sch = np.column_stack((roots_sch, KE_nc_sch, T_nc_sch[0]))
max_roots_nc_sch = max_roots_nc_sch[max_roots_nc_sch[:, 2].argsort()[::-1]]
np.savetxt("outputs/max_roots_nc_sch.txt", max_roots_nc_sch)
max_roots_nt_sch = np.column_stack((roots_sch, KE_nt_sch, T_nt_sch[0]))
max_roots_nt_sch = max_roots_nt_sch[max_roots_nt_sch[:, 2].argsort()[::-1]]
np.savetxt("outputs/max_roots_nt_sch.txt", max_roots_nt_sch)

# dch_1
dch_1_n_init_states = 1
if load_or_calc == "calc": 
    T_nc_dch_1, roots_dch_1 = auger_calc('inputs/CI_vecs_nt_dch_1_final.txt', 'inputs/mo_mull_pops_dch_1.txt', 'rasscf', 'nc', 'dch_1', dch_1_n_init_states)    
    T_nt_dch_1, roots_dch_1 = auger_calc('inputs/CI_vecs_nc_dch_1_final.txt', 'inputs/mo_mull_pops_dch_1.txt', 'rasscf', 'nt', 'dch_1', dch_1_n_init_states)    
    for i in range(dch_1_n_init_states):
        np.savetxt("outputs/T_nc_dch_1_spec_"+str(i)+".txt",T_nc_dch_1[i])
        np.savetxt("outputs/T_nt_dch_1_spec_"+str(i)+".txt",T_nt_dch_1[i])
    np.savetxt("outputs/roots_dch_1.txt",roots_dch_1)
if load_or_calc == "load":
    T_nc_dch_1 = {} 
    T_nt_dch_1 = {} 
    for i in range(dch_1_n_init_states):
        T_nc_dch_1[i] = np.loadtxt("outputs/T_nc_dch_1_spec_"+str(i)+".txt")
        T_nt_dch_1[i] = np.loadtxt("outputs/T_nt_dch_1_spec_"+str(i)+".txt")
    roots_dch_1 = np.loadtxt("outputs/roots_dch_1.txt")

energies_nc_dch_1_final = np.loadtxt('inputs/energies_nt_dch_1_final.txt')
energies_nt_dch_1_final = np.loadtxt('inputs/energies_nc_dch_1_final.txt')
energies_dch_1_init = np.loadtxt('inputs/energies_dch_1_init.txt')
KE_nc_dch_1   = {}
KE_nt_dch_1   = {}
spec_nc_dch_1 = {}
spec_nt_dch_1 = {}
for i in range(dch_1_n_init_states):
    KE_nc_dch_1[i]   = ((-1*energies_nc_dch_1_final) + energies_dch_1_init[i])*27.2114
    KE_nt_dch_1[i]   = ((-1*energies_nt_dch_1_final) + energies_dch_1_init[i])*27.2114
    spec_nc_dch_1[i] = linefit(KE_nc_dch_1[i], T_nc_dch_1[i], 1.0, False)
    spec_nt_dch_1[i] = linefit(KE_nt_dch_1[i], T_nt_dch_1[i], 1.0, False)

max_roots_nc_dch_1 = np.column_stack((roots_dch_1, KE_nc_dch_1[0], T_nc_dch_1[0]))
max_roots_nc_dch_1 = max_roots_nc_dch_1[max_roots_nc_dch_1[:, 2].argsort()[::-1]]
np.savetxt("outputs/max_roots_nc_dch_1.txt", max_roots_nc_dch_1)
max_roots_nt_dch_1 = np.column_stack((roots_dch_1, KE_nt_dch_1[0], T_nt_dch_1[0]))
max_roots_nt_dch_1 = max_roots_nt_dch_1[max_roots_nt_dch_1[:, 2].argsort()[::-1]]
np.savetxt("outputs/max_roots_nt_dch_1.txt", max_roots_nt_dch_1)

# dch_2
dch_2_n_init_states = 317
if load_or_calc == "calc": 
    T_nc_dch_2 = auger_calc('inputs/CI_vecs_nc_dch_2_final.txt', 'inputs/mo_mull_pops_nc_dch_2.txt', 'rasscf', 'nc', 'dch_2', dch_2_n_init_states)    
    T_nt_dch_2 = auger_calc('inputs/CI_vecs_nt_dch_2_final.txt', 'inputs/mo_mull_pops_nt_dch_2.txt', 'rasscf', 'nt', 'dch_2', dch_2_n_init_states)
    for i in range(dch_2_n_init_states):
        np.savetxt("outputs/T_nc_dch_2_spec_"+str(i)+".txt",T_nc_dch_2[i])
        np.savetxt("outputs/T_nt_dch_2_spec_"+str(i)+".txt",T_nt_dch_2[i])
if load_or_calc == "load":
    T_nc_dch_2 = {} 
    T_nt_dch_2 = {} 
    for i in range(dch_2_n_init_states):
        T_nc_dch_2[i] = np.loadtxt("outputs/T_nc_dch_2_spec_"+str(i)+".txt")
        T_nt_dch_2[i] = np.loadtxt("outputs/T_nt_dch_2_spec_"+str(i)+".txt")

T_nc_dch_2_all = []
T_nt_dch_2_all = [] 

# normalize each dch_2 to the dch_1 intensities 
for i in range(dch_2_n_init_states):
    norm = np.sum(T_nc_dch_2[i]) / T_nc_dch_1[0][i]
    T_nc_dch_2[i] = T_nc_dch_2[i] /norm

    norm = np.sum(T_nt_dch_2[i]) / T_nt_dch_1[0][i]
    T_nt_dch_2[i] =  T_nt_dch_2[i] / norm

energies_nc_dch_2_final = np.loadtxt('inputs/energies_nc_dch_2_final.txt')
energies_nt_dch_2_final = np.loadtxt('inputs/energies_nt_dch_2_final.txt')
energies_nc_dch_2_init  = np.loadtxt('inputs/energies_nc_dch_2_init.txt')
energies_nt_dch_2_init  = np.loadtxt('inputs/energies_nt_dch_2_init.txt')

KE_nc_dch_2_all   = []
KE_nt_dch_2_all   = []
spec_nc_dch_2_all = []
spec_nt_dch_2_all = []
for i in range(dch_2_n_init_states):
    T_nc_dch_2_all  = np.append([T_nc_dch_2_all], [T_nc_dch_2[i]])
    T_nt_dch_2_all  = np.append([T_nt_dch_2_all], [T_nt_dch_2[i]])
    KE_nc_dch_2_all = np.append([KE_nc_dch_2_all], [(((-1*energies_nc_dch_2_final) + energies_nc_dch_2_init[i])*27.2114)])
    KE_nt_dch_2_all = np.append([KE_nt_dch_2_all], [(((-1*energies_nt_dch_2_final) + energies_nt_dch_2_init[i])*27.2114)])

#load_or_calc_spec = "calc"
load_or_calc_spec = "load"
if load_or_calc_spec == "calc": 
    spec_nc_dch_2 = linefit(np.array(KE_nc_dch_2_all), np.array(T_nc_dch_2_all), 1.0, False)
    spec_nt_dch_2 = linefit(np.array(KE_nt_dch_2_all), np.array(T_nt_dch_2_all), 1.0, False)
    np.savetxt("outputs/spec_nc_dch_2.txt" ,spec_nc_dch_2)
    np.savetxt("outputs/spec_nt_dch_2.txt" ,spec_nt_dch_2)
if load_or_calc_spec == "load": 
    spec_nc_dch_2 = np.loadtxt("outputs/spec_nc_dch_2.txt") 
    spec_nt_dch_2 = np.loadtxt("outputs/spec_nt_dch_2.txt")

# plots
SMALL_SIZE = 8
MEDIUM_SIZE = 13
BIGGER_SIZE = 14

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# 
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True, sharex=True)
fig.set_figwidth(17)
fig.set_figheight(7.5)
ax1.plot(spec_nc_sch[:,0], spec_nc_sch[:,1], color='k')
ax1.vlines(3.951879553365104130e+02, 0, 1.423971954561662256e+00, label="2$\pi$,$\pi^{*}$", color="midnightblue") 
ax1.vlines(3.896282546080284988e+02, 0, 2.413519081175463210e+00, label="1$\pi$,$\pi^{*}$", color="r") 
ax1.vlines(3.878210785939842253e+02, 0, 1.621192390908830605e+00, label="2$\pi$,2$\pi$", color="lime") 
ax1.vlines(3.727418056362919856e+02, 0, 5.179385495798404904e+00, label="1$\pi$,2$\pi$", color="g")
ax1.vlines(3.717746354720300701e+02, 0, 4.467321055960883136e+00, label="1$\pi$,1$\pi$", color="sienna") 
ax1.vlines(3.632687662616520470e+02, 0, 2.045834259347062201e+00, label="6$\sigma$,6$\sigma$", color="y") 
ax1.vlines(3.629052551555606669e+02, 0, 2.010064447516481945e+00, label="5$\sigma$,2$\pi$", color="darkorchid") 
ax1.vlines(3.596991252849041416e+02, 0, 1.673508788345455356e+00, label="4$\sigma$,2$\pi$", color="orange") 
ax1.vlines(3.539198753620301545e+02, 0, 2.532308666715839873e+00, label="5$\sigma$,1$\pi$", color="b") 
ax1.vlines(3.509052696649059158e+02, 0, 3.108976265082315393e+00, label="4$\sigma$,1$\pi$", color="m") 
ax1.vlines(3.362619761253304773e+02, 0, 1.241493990700291539e+00, label="4$\sigma$,5$\sigma$", color="burlywood") 
ax1.vlines(3.232071360877383768e+02, 0, 5.782351902219287076e-01, label="4$\sigma$,4$\sigma$", color="teal") 
ax1.vlines(3.689537779361602929e+02, 0, 3.860672779636676211e+00, color="r") 
ax1.vlines(3.835864495057464296e+02, 0, 3.284791568214054713e+00, color="g") 
ax1.vlines(3.748349761854763642e+02, 0, 3.183005087212559481e+00, color="g") 
ax1.vlines(3.707537309561880079e+02, 0, 3.157689229148726184e+00, color="r") 
ax1.vlines(3.506749910154943564e+02, 0, 3.101412905921678398e+00, color="m") 
ax1.vlines(3.689537782082745139e+02, 0, 3.082844419111839152e+00, color="g") 
ax1.vlines(3.455218514605886071e+02, 0, 3.004258512969302686e+00, color="m") 
ax1.vlines(3.451666738457463453e+02, 0, 2.986984733236683454e+00, color="m") 
ax1.vlines(3.425900206653520854e+02, 0, 2.935536327471275797e+00, color="m") 
ax1.vlines(3.835648504569960551e+02, 0, 2.922936835510550946e+00, color="g") 
ax1.vlines(3.705992719748239779e+02, 0, 2.866688555799760785e+00, color="g") 
ax1.vlines(3.762241957079660892e+02, 0, 2.763309836433406996e+00, color="g") 
ax1.vlines(3.705992719748239779e+02, 0, 2.740608706008512918e+00, color="r") 
ax1.vlines(3.835864495057464296e+02, 0, 2.700629888180773541e+00, color="g") 
ax1.vlines(3.748349764575905851e+02, 0, 2.667417780325463283e+00, color="g") 
ax1.vlines(3.850403249473206415e+02, 0, 2.630532117802182768e+00, color="g") 
ax1.vlines(3.491677009562899343e+02, 0, 2.573575215338246291e+00, color="m") 
ax1.vlines(3.770522655492524677e+02, 0, 2.533466533959868983e+00, color="g") 
ax1.vlines(3.541770424121244218e+02, 0, 2.493566359772766639e+00, color="b") 
ax1.vlines(3.587520676106100268e+02, 0, 1.636817210213176388e+00, color="orange") 
ax1.vlines(3.260303509471900725e+02, 0, 1.098295695819361928e+00, color="burlywood") 
ax1.set_ylabel("Intensity (arb. unit)")
ax1.set_title("N$_{c}$ K$^{-1}\pi^{*1}$")
ax1.legend(bbox_to_anchor=(1.00,0.5), loc="center left", borderaxespad=0)

ax2.plot(spec_nt_sch[:,0], spec_nt_sch[:,1], color='k')
ax2.vlines(3.913745725980859334e+02, 0, 4.554932164040013554e+00, label="2$\pi$,$\pi^{*}$", color="midnightblue") 
ax2.vlines(3.874965527164439436e+02, 0, 3.681525365014909568e+00, label="7$\sigma$,$\pi^{*}$", color="y") 
ax2.vlines(3.836386382498063199e+02, 0, 4.379412256588676833e+00, label="1$\pi$,$\pi^{*}$", color="r") 
ax2.vlines(3.795601762025104904e+02, 0, 3.544331878095579480e+00, label="6$\sigma$,2$\pi$", color="sienna") 
ax2.vlines(3.787528751901600685e+02, 0, 3.827290483245148422e+00, label="7$\sigma$,2$\pi$", color="b") 
ax2.vlines(3.753454046334220493e+02, 0, 4.213923875891203430e+00, label="1$\pi$,7$\sigma$", color="m") 
ax2.vlines(3.738306717120902363e+02, 0 ,3.381441980465460073e+00, label="2$\pi$,2$\pi$", color="lime") 
ax2.vlines(3.642038637093043576e+02, 0, 3.461364783133779710e+00, label="5$\sigma$,2$\pi$", color="darkorchid") 
ax2.vlines(3.601424591280978689e+02, 0, 4.481137314797337012e+00, label="1$\pi$,2$\pi$", color="g")
ax2.vlines(3.559914103991886236e+02, 0, 3.396800279312976123e+00, label="6$\sigma$,1$\pi$", color="cyan") 
ax2.vlines(3.509651037239105449e+02, 0, 3.051911079909336078e+00, label="4$\sigma$,7$\sigma$", color="m") 
ax2.vlines(3.373258675782080900e+02, 0, 2.180482930242096273e+00, label="4$\sigma$,2$\pi$", color="orange") 
ax2.vlines(3.231395231058166360e+02, 0, 8.020963217655018296e-01, label="4$\sigma$,5$\sigma$", color="burlywood") 
ax2.vlines(3.154666615173679816e+02, 0, 8.020963217655018296e-01, label="4$\sigma$,4$\sigma$", color="teal") 
ax2.set_title("N$_{t}$ K$^{-1}\pi^{*1}$")
ax2.legend(bbox_to_anchor=(1.00,0.5), loc="center left", borderaxespad=0)

ax3.plot(spec_nc_dch_1[0][:,0], spec_nc_dch_1[0][:,1], color='k')
ax3.vlines(3.999229449268080430e+02, 0, 4.503369945868054836e-00, label="$\pi^{*}$,$\pi^{*}$", color="darkgrey") 
ax3.vlines(3.926533478508260941e+02, 0, 5.013240960055350692e-01, label="2$\pi$,$\pi^{*}$", color="midnightblue") 
ax3.vlines(3.804107158573465881e+02, 0, 1.296062041274286036e+00, label="7$\sigma$,2$\pi$", color="teal") 
ax3.vlines(3.786155112266185370e+02, 0, 1.193338239547381052e+00, label="6$\sigma$,2$\pi$", color="burlywood") 
ax3.vlines(3.751454226125422338e+02, 0, 1.064452340730660529e+00, label="6$\sigma$,7$\sigma$", color="pink") 
ax3.vlines(3.698848358514823076e+02, 0, 2.112766407818916115e+00, label="1$\pi$,2$\pi$", color="lime") 
ax3.vlines(3.679294273686224983e+02, 0, 2.000249932646506412e+00, label="2$\pi$,2$\pi$", color="m") 
ax3.vlines(3.652247129860046471e+02, 0, 2.429961335286314306e+00, label="1$\pi$,2$\pi$", color="g")
ax3.vlines(3.650776312872946505e+02, 0, 2.045886593741642834e+00, label="1$\pi$,1$\pi$", color="b") 
ax3.vlines(3.645411043856084348e+02, 0, 2.305143211646726087e+00, label="1$\pi$,$\pi^{*}$", color="r") 
ax3.vlines(3.585078637993622124e+02, 0, 1.456133439443260569e+00, label="4$\sigma$,2$\pi$", color="sienna") 
ax3.vlines(3.463377352206126147e+02, 0, 2.226469127871899989e+00, label="4$\sigma$,1$\pi$", color="darkorchid") 
ax3.vlines(3.419716138447242315e+02, 0, 2.181255655150316386e+00, label="5$\sigma$,1$\pi$", color="y") 
ax3.vlines(3.215806886685043082e+02, 0, 2.327637303523349299e+00, label="4$\sigma$,5$\sigma$", color="orange") 
ax3.vlines(3.203618549597981087e+02, 0, 2.310286141891199296e+00, color="orange") 
ax3.vlines(3.334018737641519579e+02, 0, 2.267245983340700377e+00, color="orange") 
ax3.vlines(3.319542196649599646e+02, 0, 2.258780879664803454e+00, color="orange") 
ax3.vlines(3.479872843021045128e+02, 0, 2.196196876506590900e+00, color="darkorchid") 
ax3.vlines(3.386440773197139151e+02, 0, 2.194946411189642177e+00, color="darkorchid") 
ax3.vlines(3.518910940602106052e+02, 0, 2.128319374384212370e+00, color="y")
ax3.vlines(3.692751652508243865e+02, 0, 2.059625960962057967e+00, color="lime") 
ax3.vlines(3.596170298516744879e+02, 0, 1.534340499165032679e+00, color="y") 
ax3.vlines(3.708916364265906509e+02, 0, 1.657137776026848419e+00, color="g")
ax3.vlines(3.720685349188703412e+02, 0, 1.654923278911957674e+00, color="g")
ax3.vlines(3.716508040098221386e+02, 0, 1.637558281990267428e+00, color="g")
ax3.vlines(3.883731902807920733e+02, 0, 1.347334380540232246e+00, color="r")
ax3.vlines(3.878339142113640037e+02, 0, 1.223774378671995677e+00, color="r")
ax3.vlines(3.767309070786720895e+02, 0, 1.192389623730908843e+00, color="burlywood") 
ax3.vlines(3.833382050737122313e+02, 0, 1.087375414151956265e+00, color="m") 
ax3.plot(spec_nt_dch_2[:,0], spec_nt_dch_2[:,1], label='2nd Step')
ax3.set_title("N$_{c}$ K$^{-1}$K$^{-1}\pi^{*2}$")
ax3.set_ylabel("Intensity (arb. unit)")
ax3.set_xlabel("Kinetic Energy (eV)")
ax3.legend(bbox_to_anchor=(1.00,0.5), loc="center left", borderaxespad=0)

ax4.plot(spec_nt_dch_1[0][:,0], spec_nt_dch_1[0][:,1], color='k')
ax4.vlines(3.954530136926040882e+02, 0, 1.186528847494550121e+01, label="$\pi^{*}$,$\pi^{*}$", color="darkgrey") 
ax4.vlines(3.914028098678662673e+02, 0, 2.095708109944909747e+00, label="2$\pi$,$\pi^{*}$", color="midnightblue") 
ax4.vlines(3.849625251056942261e+02, 0, 3.816885944060097735e+00, label="7$\sigma$,2$\pi$", color="teal") 
ax4.vlines(3.780855882595981825e+02, 0, 3.616530222231264080e+00, label="2$\pi$,2$\pi$", color="m") 
ax4.vlines(3.738271614414906026e+02, 0, 2.451782757968772763e+00, label="6$\sigma$,2$\pi$", color="burlywood") 
ax4.vlines(3.645875264897803731e+02, 0, 1.865631741227667728e+00, label="5$\sigma$,$\pi^{*}$", color="peru") 
ax4.vlines(3.521183185020863675e+02, 0, 3.191411570781414131e+00, label="5$\sigma$,2$\pi$", color="y") 
ax4.vlines(3.443897875594045672e+02, 0, 2.610083996413527618e+00, label="4$\sigma$,2$\pi$", color="sienna") 
ax4.vlines(3.312238684101000104e+02, 0, 1.485496416225833949e+00, label="4$\sigma$,1$\pi$", color="darkorchid") 
ax4.vlines(3.286164320613419250e+02, 0, 2.143550411251418986e+00, label="4$\sigma$,5$\sigma$", color="orange") 
ax4.vlines(3.158928157152861331e+02, 0, 1.718753074003380732e+00, label="4$\sigma$,4$\sigma$", color="teal") 
ax4.vlines(3.191575924115646217e+02, 0, 1.973519880845070906e+00, color="orange") 
ax4.vlines(3.399641608382245295e+02, 0, 1.820460908261972044e+00, color="sienna") 
ax4.vlines(3.796148809126141259e+02, 0, 3.397869833484449487e+00, color="m") 
ax4.vlines(3.572361844634003205e+02, 0, 2.838844367398446256e+00, color="y") 
ax4.vlines(3.595013571835282278e+02, 0, 2.809990851100531817e+00, color="y") 
ax4.vlines(3.722153219181181498e+02, 0, 2.434841437554053467e+00, color="burlywood") 
ax4.vlines(3.717613176686426186e+02, 0, 2.548202749198912986e+00, color="m") 
ax4.vlines(3.702392466490721858e+02, 0, 2.604390504898592251e+00, color="m") 
ax4.vlines(3.428540204979859709e+02, 0, 2.705338619481242546e+00, color="sienna") 
ax4.plot(spec_nc_dch_2[:,0], spec_nc_dch_2[:,1], label='2nd Step')
ax4.set_title("N$_{t}$ K$^{-1}$K$^{-1}\pi^{*2}$")
ax4.set_xlabel("Kinetic Energy (eV)")
ax4.legend(bbox_to_anchor=(1.00,0.5), loc="center left", borderaxespad=0)
plt.xlim(312, 402)
plt.subplots_adjust(top = 0.95, bottom = 0.07, right = 0.91, left = 0.05,
            wspace=0.23, hspace=0.12)
plt.savefig("filename.pdf", bbox_inches = 'tight',
    pad_inches = 0)
plt.savefig("figures/test.png",dpi=300)
plt.show()

