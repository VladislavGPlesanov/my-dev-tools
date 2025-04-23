import numpy as np
import matplotlib.pyplot as plt
import sys


#angles = sys.argv[1]

distance = sys.argv[1]

def angleToRadian(angle):
    return angle*3.14159/180

epsilon_x, epsilon_y = 16, 22 # in units of [m] instead of [mm]
sigmaX, sigmaY = 2.83, 1.94 # mm

# using 4*epsilon=sigmaX*sigma(prim)X

sigmaThetaX = (4*epsilon_x/sigmaX)/1000
sigmaThetaY = (4*epsilon_y/sigmaY)/1000

#x0 = np.sqrt(epsilon_x)
#angle0 = 1 # deg
#slope0 = angleToRadian(angle0)

myAngles = []
myPositions = []

#for i in range(200):
#    #myAngles.append(np.random.uniform(-1,1))
#    #myAngles.append(np.random.uniform(-float(sigmaThetaX),float(sigmaThetaX)))
#    #myAngles.append(np.random.uniform(-float(sigmaThetaX),float(sigmaThetaX)))
#    myAngles.append(np.random.normal(0,3*sigmaThetaX))
#
#for i in range(200):
#    #myPositions.append(0+(np.random.uniform(-float(sigmaX/1000),float(sigmaX/1000))))
#    myPositions.append(np.random.normal(0,3*sigmaX/1000))

#u0 = np.array([[x0],[slope0]]) # initial state u(s)

#L_distances = np.linspace(0,10,100) # drift space distances in L = s-s0 [m]
#L_distances = [0,0.4] # drift space distances in L = s-s0 [m]
#L_distances = [0,float(distance)] # drift space distances in L = s-s0 [m]
L_distances = np.linspace(0,float(distance),10) # drift space distances in L = s-s0 [m]

allX = []
allY = []

finalX_list, finalY_list= [], []
finalTX_list, finalTY_list= [], []

#for theta,xpos in zip(myAngles,myPositions): 
#
#    new_x = []
#    
#    for l in L_distances:
#        #u0_mod = np.array([[xpos],[angleToRadian(theta)]])
#        u0_mod = np.array([[xpos/1000],[theta]])
#        M_ds = np.array([[1,l],[0,1]]) # transformation matrix fro Drift Space
#        x1 = np.dot(M_ds,u0_mod) # calc. new x by transforming initial state
#        new_x.append(x1[0,0])# getting x from the new state vector 
#        finalX_list.append(x1[0,0])    
#
#    allX.append(new_x)

newTX, newTY = [],[]
new_x, new_y = [],[]

# calculating for y coordinate:
for l in L_distances:
    u0_mod = np.array([[sigmaX/1000],[sigmaThetaX]])
    M_ds = np.array([[1,l],[0,1]]) # transformation matrix fro Drift Space
    x1 = np.dot(M_ds,u0_mod) # calc. new x by transforming initial state
    new_x.append(x1[0,0])# getting x from the new state vector 
    newTX.append(x1[1,0])# getting x from the new state vector 
    finalX_list.append(x1[0,0])    

allX.append(new_x)

# calculating for y coordinate:
for l in L_distances:
    u0_mod = np.array([[sigmaY/1000],[sigmaThetaY]])
    M_ds = np.array([[1,l],[0,1]]) # transformation matrix fro Drift Space
    x1 = np.dot(M_ds,u0_mod) # calc. new x by transforming initial state
    new_y.append(x1[0,0])# getting x from the new state vector 
    newTY.append(x1[1,0])# getting x from the new state vector 
    finalY_list.append(x1[0,0])    

allY.append(new_y)

print(finalX_list)
print(len(finalX_list))

plt.figure()
plt.scatter(L_distances,finalX_list, marker='*',label=r'Beam $\sigma_{x}$')
plt.scatter(L_distances,finalY_list, marker='+', label=r'Beam $\sigma_{y}$')
plt.xlabel("Drift space distance, [m]")
plt.ylabel("Transformed x, [m]")
#plt.ylim([0.1, 0.15])
plt.grid(True)
plt.legend()
plt.savefig("BeamDivergenceEstimate-sigmaChange.png")

plt.figure()
plt.scatter(L_distances,newTX, marker='*',label=r'Beam $\sigma_{\theta\,x}$')
plt.scatter(L_distances,newTY, marker='+', label=r'Beam $\sigma_{\theta\,y}$')
plt.xlabel("Drift space distance, [m]")
plt.ylabel("Transformed x, [m]")
#plt.ylim([0.1, 0.15])
plt.grid(True)
plt.legend()
plt.savefig("BeamDivergenceEstimate-sigmaThetaChange.png")


plt.figure()
for xlist in allX:
    plt.plot(L_distances,xlist)
plt.xlabel("Drift space distance")
plt.ylabel("Transformed x")
plt.xlim([0.0, 0.01])
plt.ylim([-0.025, 0.025])
plt.savefig("BeamDivergenceEstimate-zoom.png")


plt.figure()
plt.hist(myPositions)
plt.xlabel("x position of the particle")
plt.ylabel("N particles")
plt.title(r"Particle position distribution($\sigma_{x}$)")
plt.savefig("beamDistributionSigmaX.png")

plt.figure()
plt.hist(myAngles)
plt.xlabel(r"$\theta$ angle of the particle")
plt.ylabel("N particles")
plt.title(r"Particle angular distribution($\sigma_{x^{'}}$)")
plt.savefig("beamDistributionSigmaTheta.png")


beam_stdev_x = np.std(finalX_list)

print("=================================================================================")
print(f"At a distance of {distance} [m]:\n")
print(f"Beam deviation in drift space : {beam_stdev_x*1000} [mm]")
print(f"maximum spread in x is : {np.max(finalX_list)*1000 - np.min(finalX_list)*1000} [mm]")
print(f"maximum spread in y is : {np.max(finalY_list)*1000 - np.min(finalY_list)*1000} [mm]")
print("=================================================================================")


