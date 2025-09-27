import json
import sys
from pprint import pprint

f = open(sys.argv[1],'r') # input backup

search = None
if(len(sys.argv)>2):
    search = sys.argv[2]

bak = json.load(f)

cnt = 0
n_skips = 0
for key, val in bak.items():

    if(search is not None):
        if(search in key):
            outstr = "{} = {}".format(key,val)
            pprint(outstr)
        else:
            n_skips+=1            
    else:
        outstr = "{} = {}".format(key,val)
        pprint(outstr)
    cnt+=1

if(n_skips == cnt and search is not None):
    print("Culd not find <{}>".format(search))

f.close()
