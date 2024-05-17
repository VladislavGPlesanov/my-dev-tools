import numpy as np
import matplotlib.pyplot as plt

E_jetmin = 30 #default# Gev
s = 14000 # Tev
abs_eta_max = 3.9 # for eta_3,_4 |<4|
eta_max = 3.9
eta_min = eta_max*(-1)
eta_step = 0.01

def calcInvMass(x1,x2,s):
    return x1*x2*s

def calcJetMass(Ejet, eta3, eta4):
    M = 2*Ejet*np.cosh((eta3-eta4)/2)
    return M

def calcThetaStar(eta3,eta4):
    ts = np.tanh((eta3-eta4)/2)
    return ts

def calcPartonMomFrac(Ejet, Ecom, eta3, eta4, parton_index):
    parton_p = None
    if(parton_index==1):
       parton_p = (Ejet/Ecom)*(np.exp(eta3)+np.exp(eta4))
    if(parton_index==2):
       parton_p = (Ejet/Ecom)*(np.exp(-eta3)+np.exp(-eta4))
    if(parton_index>2 or parton_index<1):
       print("Use only 1,2 for parton index")
       exit(0)
    return parton_p

def getEtaM(eta3,eta4):
    return (eta3+eta4)/2

def getEtaStar(eta3,eta4):
    return (eta3-eta4)/2

etas3 = np.arange(eta_min,eta_max,eta_step)
#etas3 = np.linspace(-4,4,1000)
etas4 = np.arange(eta_min,eta_max,eta_step)
#etas4 = np.arange(-3.9,3.9,0.01)
#etas4 = np.linspace(-4,4,1000)

part_x1 = []
part_x2 = []
etaMs = []
eta_star = []
jet_Ms = []

###############################################
for e3 in etas3:
    for e4 in etas4:
        x1 = calcPartonMomFrac(E_jetmin, s, e3, e4 ,1)
        part_x1.append(x1) 
        x2 = calcPartonMomFrac(E_jetmin, s, e3, e4 ,2) 
        part_x2.append(x2) 
        nM = getEtaM(e3,e4)
        nS = getEtaStar(e3,e4)
        etaMs.append(nM)        
        eta_star.append(nS)        
        #jmass = calcJetMass(E_jetmin, e3,e4)
        #jet_Ms.append(jmass)


sorted_index = np.argsort(part_x1)
print(len(sorted_index))
print(len(part_x1))
x1_arr = np.array(part_x1)[sorted_index]
x2_arr = np.array(part_x2)[sorted_index]
nM_arr = np.array(etaMs)[sorted_index]
nS_arr = np.array(eta_star)[sorted_index]

#eta3_arr = np.array(etas3)[sorted_index]
#eta4_arr = np.array(etas4)[sorted_index]

plt.figure(10,figsize=(10,10))
plt.scatter(x1_arr,nM_arr,c='r',s=1,label='x1')
plt.scatter(x2_arr,nM_arr,c='g',s=1,label='x2')
plt.xlabel('p_x1 & p_x2 [GeV]')
plt.ylabel('etaM')
plt.xscale('log')
#plt.iyscale('log')
plt.grid(True)
plt.savefig("huya.png")

x1_arr_plot = x1_arr[np.abs(nM_arr)<0.01]
x2_arr_plot = x2_arr[np.abs(nM_arr)<0.01]

# y = ax + b
a,b = np.polyfit(x1_arr_plot, x2_arr_plot, 1)

x1line = np.linspace(0,1,1000000)
x2line = a*x1line + b 

cnt = 0
for point in x1_arr:
    if(cnt%1000==0):
        print(f"x1={point:.5f},x2={x2_arr[cnt]:.5f}")
    cnt+=1

min_x1 = np.min(x1_arr)
min_x2 = np.min(x2_arr)
max_x1 = np.max(x1_arr)
max_x2 = np.max(x2_arr)

#msg="x1(min)={}, x1(max)={}\nx2(min)={},x2(max)={}".format(min_x1,max_x1,min_x2,max_x2)
msg=f"x1(min)={min_x1:.7f}, x1(max)={max_x1:.5f}\nx2(min)={min_x2:.7f},x2(max)={max_x2:.5f}"
print(msg)

plt.figure(1,figsize=(10,10))
#plt.scatter(part_x1,part_x2)
plt.plot(x1_arr,x2_arr)
#plt.scatter(x1_arr,x2_arr)
plt.plot(x1line,x2line,c='orange')
#plt.scatter(min_x1,min_x2,c='r')
#plt.scatter(max_x1,max_x2,c='c')
plt.plot(min_x1,max_x2,'+g')
plt.plot(max_x1,min_x2,'+y')
plt.axvline(0.0045, color='green')
plt.axhline(0.0045, color='green')
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.00005,0.8)
plt.ylim(0.00005,0.8)
plt.xlabel("x1,[GeV]")
plt.ylabel("x2,[Gev]")
plt.grid(True)
#plt.show()
plt.savefig("partonmomentum.png")
###############################################

plt.figure(2,figsize=(10,10))
#EjetList = [30,80, 100,300]
#EjetList = [30,300]
colors = ['blue','yellow','green']
#colors = ['blue','yellow','green','red']
EjetList = np.linspace(30,300,3)
cntr=0
for Ej in EjetList:
    Elist = []
    for e3 in etas3:
        for e4 in etas4:
            jmass = calcJetMass(Ej, e3,e4)
            Elist.append(jmass)

    plt.hist(Elist, bins=100, range=(0,20000),alpha=0.25,label=f'{Ej} Gev')
    print("E(jet) = {}".format(Ej))
    print("E(jet)(min) = {}".format(min(Elist)))
    print("E(jet)(max) = {}\n".format(max(Elist)))
    Emin = min(Elist)
    plt.axvline(min(Elist), color=colors[cntr],label=f'E_min={Emin}[Gev]')
    cntr+=1

plt.xlabel('M [Gev]')
plt.ylabel('Events [N]')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig("jetInvMass.png")


###############################################
thetas_star = []
for e3 in etas3:
    for e4 in etas4:
        theta = calcThetaStar(e3,e4)
        thetas_star.append(theta)

##############################################
plt.figure(3,figsize=(10,10))
plt.hist(thetas_star, bins=100, range=(-1,1), alpha=0.5)
plt.xlabel('theta_star')
plt.ylabel('Events[N]')
plt.grid(True)
#plt.show()
plt.savefig("jetDirection.png")



