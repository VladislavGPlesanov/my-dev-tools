import os
import numpy as np
#import 
import ctypes
import pathlib
from time import time, sleep
from testClass.test_dataManagement import DataManagement
from testClass.test_extendedDM import ExtendedDataManagement
from array import array

if __name__ == "__main__":
  
    clib = ctypes.CDLL('/home/vlad/readoutSW/tools/ArraySorter.so')
    #clib = ctypes.CDLL('/home/vlad/ArraySorter.so')
    copy_data = clib.copyDataArray
    update_array = clib.updateArray
    getFifoData = clib.getFifoData
    merge_arrays = clib.mergeArrays
    ##########################
    #fiddle = clib.fiddleWDict
    #fiddle.argtypes = [ctypes.py_object]

    #######################
    #copy_data = clib.fastInput.copyDataArray
    copy_data.argtypes = [ctypes.POINTER(ctypes.c_int),
                          ctypes.POINTER(ctypes.c_int), 
                          ctypes.c_size_t]

    update_array.argtypes = [np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='C_CONTIGUOUS'),
                             np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='C_CONTIGUOUS'),
                             ctypes.c_size_t,
                             ctypes.POINTER(ctypes.c_size_t)]

    getFifoData.argtypes = [ctypes.POINTER(ctypes.c_uint32),
                            ctypes.POINTER(ctypes.c_uint32), 
                            ctypes.c_int]

    merge_arrays.argtypes = [np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='C_CONTIGUOUS'),
                             ctypes.c_size_t,
                             np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='C_CONTIGUOUS'),
                             ctypes.c_size_t]
     

    def copyData(array_in, size):
        i_array = (ctypes.c_int * size)(*array_in)
        o_array = (ctypes.c_int * size)()
        copy_data(i_array, o_array, size)
        o_array = list(i_array)
        return o_array

    def updateArray(data_in, data_out):
        outSize = data_out.size
        print("outsize="+str(outSize))
        inSize = data_in.size
        print("insize="+str(inSize))
        update_array(data_out, data_in, inSize, ctypes.byref(ctypes.c_size_t(outSize)))

    def addToData(ex_data, input_data, exsize, isize):
        d_in = (ctypes.c_uint32 * isize)(*input_data)
        d_ex = (ctypes.c_uint32 * exsize)(*ex_data)
        return getFifoData(d_ex, d_in, isize)

    def appArrays(curr_array, new_array):
        curr_size = curr_array.shape[0]
        new_size = new_array.shape[0]
        result = np.empty(curr_size+new_size, dtype=np.uint32)
        #result[:curr_size] = curr_array
        result = merge_arrays(curr_array,
                              ctypes.c_size_t(curr_size),
                              new_array,
                              ctypes.c_size_t(new_size))
        #result[:curr_size] = curr_array
        #result[curr_size:] = new_array

        return result;

    ##################################################
    ## Load the shared library
    #lib = ctypes.CDLL("your_shared_library.so")  # Change to your actual shared library name
    #
    ## Declare the return type of the function
    #lib.query_data_from_device.restype = ctypes.py_object
    #
    ## Call the C++ function
    #result = lib.query_data_from_device()
    #
    ## Convert the result to a NumPy array
    #result_np = np.array(result)
    #
    #print(result_np)
    #################################################

    #t0 = time()
    #data0 = np.array(range(0,10000000,1))
    #data2 = copyData(data0, len(data0))
    ##print(np.array(data2))
    #t1 = time()
    #print(t0-t1)


    #t2 = time()
    #data1 = np.array(range(0,10000000,1))
    #data3 = np.asarray(data1,dtype=np.uint32)
    #t3 = time()
    #print(t2-t3)

    #t4 = time()
    #sleep(1)
    #t5 = time()
    #print(t5-t4)

    
    hoba = np.array([],dtype=np.uint32)
    yoba0 = np.array([666],dtype=np.uint32)
    #hoba = np.array([9,8,7,6,5,4,3,2,1],dtype=np.uint32)
    yoba1 = np.array([1,4,5,4],dtype=np.uint32)
    yoba2 = np.array(range(6,30,5),dtype=np.uint32)
    yoba3 = np.array(range(30,120,10),dtype=np.uint32)

    #print(hoba)
    #print(yoba)
    #print(yoba1)
    #print(yoba2)

    #hoba = addToData(hoba, yoba, len(hoba), len(yoba))
    print(hoba)
    print(yoba1)

    #hoba = appArrays(hoba, yoba1)
    hoba = appArrays(yoba0, yoba1)

    print(hoba)
    print(yoba1)
    

    #mergeArrays(hoba,yoba1)
    #print(hoba)

    #mergeArrays(hoba,yoba2)
    #print(hoba)

#################################################################

    testlist = [1,2,3,4,5,6,7,8,9,10]
    yoba = testlist[:7] # ---> [1, 2, 3, 4, 5, 6, 7]
    hoba = testlist[7:] # ---> [8,9,10]

    print("testlist slicing: \nbefore={}\nafter1={}\nafter2={}".format(testlist, yoba, hoba))

    huita = array('B',[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    print("huita={}".format(huita))
    print("huita[:7]={}".format(huita[:7]))
    print("huita[7:]={}".format(huita[7:]))


    testhex = 0x1cfd
    length = 16
    print(f'{testhex:0>{length}b}')
    print(bin(testhex))
    print(format(testhex,'#016b'))

    myint = 89

    print("myint is {}".format(myint))
    print("myint div by 8 is {}".format(myint//8))
    print("myint times divider {}".format((myint//8)*8))
    print("myint remainder {}".format(myint - (myint//8)*8))

    print("huyase:")
    shitlist = ['1720100521', 'ESTAB', '', '', '', '', '', '56664', '', '', '0', '', '', '', '', '', '', '192.168.10.1:52464', '', '', '192.168.10.16:24', '', '', '', '', '', '', '', '', '','\n']
    for item in shitlist:
        try:
            print("item={}[{}] > int(item)={}".format(item, type(item), int(item)))
        except:
            print("failed to do int({})".format(item))
            pass

#################################################################
    # Creating a dictionary with 5 arbitrary key/value pairs using ExtendedDataManagement
   # EM = ExtendedDataManagement()  

   # EM.set_data({"key1": "value1", "key2": "value2", "key3": "value3", "key4": "value4", "key5": "value5"})
   # 
   # # Adding a couple of key/value pairs at the end of the dictionary
   # DM.update_data("new_key1", "new_value1")
   # DM.update_data("new_key2", "new_value2")
   # 
   # EM.update_data("new_key1", "new_value1")
   # EM.update_data("new_key2", "new_value2")


#################################################################
#    e = np.array([0],dtype=np.uint32)
#    #print(e.shape)
#    b = np.array([2,3,4],dtype=np.uint32)
#    #print(b.shape)
#    a = np.array([3,6,8,240,100000,0],dtype=np.uint32)
#    #print(a.shape)
#    t = np.array([0,2,3,4],dtype=np.uint32)
#    #print(t.shape)
#
#    p0 = time()
#    eb = np.concatenate((e,b))
##    print("eabt")
#    eba = np.concatenate((eb,a))
##    print("ebat")
#    ebat = np.concatenate((eba,t))
#    print("ebat")
#    print(time()-p0)
#
#    #salam = np.concatenate((ebat,eb))
#    #aleikum = np.concatenate((salam,eb))
#
#    #print(e)
#    #print("e - khuy")
#    #print(eb)
#    #print("eb - khuy")
#    #print(eba)
#    #print("eba - khuy")
#    print(ebat)
#    print(len(ebat))
#    #print("ebat - khuy") 
#
#    #print(aleikum)
#
#    dlist = []
#
#    p1=time()
#    dlist.extend(e)
#    dlist.extend(b)
#    dlist.extend(a)
#    dlist.extend(t)
#    print(dlist)
#    eblo = np.asarray(dlist, dtype=np.uint32)
#    print(eblo)
#    print(len(eblo))
#    print(time()-p1)
#
#
#   # #hexlist = [hex(x) for x in range(256,512,1)]
#   # hexlist = ["0x%02x" % x for x in range(256,512,1)]
#   # hexarray = np.asarray(hexlist)
# 
#   # print(hexarray)

