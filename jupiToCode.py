import json
import sys

f = open(sys.argv[1],'r') # input jupiter nb

jpnb = json.load(f)

ofile = open(sys.argv[2],'w') #clear code out
mfile = None

if(len(sys.argv)>3):
    mfile = open(sys.argv[3],'w')

if(jpnb["nbformat"] >=4):
    for i,cell in enumerate(jpnb["cells"]):
        if(cell["cell_type"]=="markdown"):
            if(mfile is not None):
                for line in cell["source"]:
                    mfile.write(line)
                mfile.write("\n\n")
            else:
               continue
        ofile.write("#cell "+str(i)+"\n")
        for line in cell["source"]:
            ofile.write(line)
        ofile.write("\n\n")
else:
    for i,cell in enumerate(jpnb["worksheets"][0]["cells"]):
        if(cell["cell_type"]=="markdown"):
            if(mfile is not None):
                for line in cell["source"]:
                    mfile.write(line)
                mfile.write("\n\n")
            else:
                continue
        ofile.write("#cell "+str(i)+"\n")
        for line in cell["input"]:
            ofile.write(line)
        ofile.write("\n\n")

ofile.close()


    
