import sys
from time import sleep


def countMonitor(objlist, namelist):
    while True:
        try:
            cnt=0
            for obj in objlist:
                print("obj[{}] => {}".format(namelist[cnt],sys.getrefcount(obj)))
                cnt+=1
            sleep(1)
        except KeyboardInterrupt:
            try:
                input("pausing. ENTER->continue, CTRL+C->quit")
            except KeyboardInterrupt:
                print("\n")
                return

if __name__ == '__main__':
    
    #tpx3objects = ["tpx3.fifo_readout.FifoReadout", "tpx3.tpx3.TPX3", "basil.HL.sitcp_fifo.sitcp_fifo"]
    tpx3objects = ["self", "chip", "fifo"]
    objNames = ["self", "chip", "fifo"]
    countMonitor(tpx3objects, objNames)
    sys.exit(0)

    

