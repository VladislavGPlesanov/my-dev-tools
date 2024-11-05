import numpy as np
import sys
import matplotlib.pyplot as plt

infile = sys.argv[1]
columns = sys.argv[2].split(',') #this shluld be comma sep. list
separator = None
plottype = None

columns = list(columns)
print(columns)
#columns = [col for col in columns if col != ',']
#print(columns)
plotdesc =  None
plotname = None

if (len(sys.argv) > 3):
    plotname = sys.argv[3]
else:
    plotname = "greekTragedy"

if(len(sys.argv)>4):
    plotdesc = list(sys.argv[4].split(','))
else:
    plotdesc = ["x","y","title"]
xdata = []
ll_ydata = [[] for i in range(0,len(columns)-1)]

plottype = sys.argv[5]

if(plottype == None):
    print("sys.argv[5] has to be eihter 'plot' or 'hist'. Defaulting to 'plot' option")
    plottype = 'plot'

# TRYNA DETERMINE THE SEPARATOR TYPE
###################### 

######################
f = open(infile)
bad_list = [' ','\n','\t',',']
cnt = 0
bare_line = []
for line in f:
    words = line.split(separator)
    words = [word for word in words if word != '']
    print(words[:-1],len(words[:-1]))
    if("ESTAB" in words):
        print("line contains \"ESTAB\"")
        for i in range(0,len(columns)):
            if(i==0):
                x = words[int(columns[0])] 
                print("adding {} to xdata".format(x))
                xdata.append(int(x))
            else:
                y = words[int(columns[i])] 
                print("adding {} to ydata".format(y))
                ll_ydata[i-1].append(int(y))
    cnt+=1
print("xdata={}".format(xdata[0:10]))
print("ydata[0]={}".format(ll_ydata[0:10]))
print(type(ll_ydata))
print(type(ll_ydata[0]))

plt.figure(figsize=(10,10))
for ydata in ll_ydata:
    print("tryina' plot {} vs {}".format(len(xdata), len(ydata)))
    plt.scatter(xdata, ydata)
plt.xlabel(plotdesc[0])
plt.ylabel(plotdesc[1])
plt.title(plotdesc[2])
plt.savefig("test-"+plotname+".png")



