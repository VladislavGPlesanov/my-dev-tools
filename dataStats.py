import sys
import numpy as np


#idef

infile = open(sys.argv[1])

cnt = 0

datawords = np.array([],dtype=np.uint32) 

skip = [" ", "\n", "\t",'']
for line in infile:
    words = line.split("\t")
    if(cnt<5):
        print(words)
    for word in words:
        if(word not in skip):
            datawords = np.append(datawords, int(word))
    print("< {} >".format(cnt), end='\r')
    if(cnt==3000):
        break
    cnt+=1

print("EBAL!")
print(len(datawords))
nwords = len(datawords) #!
stdev = datawords.std() #!
mean = datawords.mean() #!
list_nonZero = datawords.nonzero()
has_nZeros = len(list_nonZero) #!
var = datawords.var() #!

uniq = []
cnt2 = 0 
for w in datawords:
    if w not in uniq:
        uniq.append(w)
    print("< counting unique {}/100 >\r".format( round((cnt2/len(datawords)*100),2)),end='\r')
    cnt2+=1

nuniq = len(uniq) #!

basestring = "Stats:\n nwords={}\n mean={},\n stdev={},\n var={},\n nas_N_zeros={},\n has_n_unique={}\n"
print(basestring.format(nwords,
                        mean,
                        stdev,
                        var,
                        has_nZeros,
                        nuniq
                        ))


