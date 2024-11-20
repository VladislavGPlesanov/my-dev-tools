import sys
import os
import subprocess

#def do_grep(pattern,file):
#    try:
#        result = subprocess.run(['grep',pattern,file],capture_output=True, text=True, check=True)
#        return result.stdout
#    except subprocess.CalledProcessorError as e:
#        print("Failed to call grep. WHY? -> {}".format(e))
##########################################

searchfor = "INCREF\|DECREF\|readoutToDeque"
thisfile = "myReadoutBuffer.cpp"

f = open(thisfile)

cnt=0;
lock = False
wordlist = []
cnt=0;
for line in f:
    if(cnt<1733):
        cnt+=1
        continue

    words = line.split(';')
    #print(words)
    for w in words:
        if("Py_INCREF" in w or "Py_DECREF" in w and "//" not in w):
            wordlist.append(w)
    cnt+=1

print("removing spaces/slashes")
ptrlist_all = []
for word in wordlist:
    print(word)
    cleanword = ""
    for c in word:
        if(c==" " or c=="/"):
            continue
        else:
            cleanword+=c
    ptrlist_all.append(cleanword)

print("uncommented pointer refs")
uniqueptr_inc = []
uniqueptr_dec = []
for w in ptrlist_all:
    print(w)
    if(w not in uniqueptr_inc and "INCREF" in w):
        uniqueptr_inc.append(w)
    if(w not in uniqueptr_dec and "DECREF" in w):
        uniqueptr_dec.append(w)


occurances = {}
for ptr in uniqueptr_inc:
    cnt = 0
    for word in ptrlist_all:
        if ptr == word:
           cnt+=1
        else:
            continue 
    occurances[ptr] = cnt

for ptr in uniqueptr_dec:
    cnt = 0
    for word in ptrlist_all:
        if ptr == word:
           cnt+=1
        else:
            continue 
    occurances[ptr] = cnt

#singleref = []
#lendec = len(uniqueptr_dec)
#leninc = len(uniqueptr_inc)
#bareptr = []
#for ptr in uniqueptr_dec:
#    #print(ptr[9:]) #word without first 9 characters
#    bareptr.append(ptr[9:])


cnt1 = 0
print("pointer Py_INCREF counts:")
for ptr in uniqueptr_inc:
    print("{}={}".format(ptr,occurances[ptr]))
    cnt1+=1
print(cnt1)
cnt2=0
print("pointer Py_DECREF counts:")
for ptr in uniqueptr_dec:
    print("{}={}".format(ptr,occurances[ptr]))
    cnt2+=1
print(cnt2)


print("________COMPARISON__________\n")
notinIncref = []
#for ptr in uniqueptr_inc:
for ptr in uniqueptr_dec:
    name = ptr[9:]
    nameinc = None
    #print(name)
    for item in uniqueptr_inc:
        if(item[9:]==name):
            nameinc = item
    if(nameinc==None):
        #print("{}={} NOT in INCREF list!".format(ptr,occurances[ptr]))
        notinIncref.append(ptr)
        continue
    if(occurances[ptr]!=occurances[nameinc]):
        print("-----\n>> {}={},\t{}={}\n-----".format(ptr, occurances[ptr], nameinc, occurances[nameinc]))
    else:
        print("{}={},\t{}={}".format(ptr, occurances[ptr], nameinc, occurances[nameinc]))
          
print("\nCould not find following INCREFs:")  
for tag in notinIncref:
    print("{}={}".format(tag,occurances[tag]))


#sorted_inc = sorted(uniqueptr_inc)
#sorted_dec = sorted(uniqueptr_dec)
#
#print(sorted_inc)

#for (inc_i, dec_i) in zip(sorted_inc, sorted_dec):
#    print("{}={}, {}={}".format(inc_i, occurances[inc_i], dec_i, occurances[dec_i]))






