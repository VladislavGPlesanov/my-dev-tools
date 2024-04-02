import os
import numpy as np
#import 
import ctypes
import pathlib
from time import time, sleep
from testClass.test_dataManagement import DataManagement
from testClass.test_extendedDM import ExtendedDataManagement

if __name__ == "__main__":
  
    clib = ctypes.CDLL('/home/vlad/readoutSW/tools/ArraySorter.so')
    #clib = ctypes.CDLL('/home/vlad/ArraySorter.so')
    copy_data = clib.copyDataArray
    update_array = clib.updateArray
    getFifoData = clib.getFifoData
    merge_arrays = clib.mergeArrays
    ##########################
    fiddle = clib.fiddleWDict
    fiddle.argtypes = [ctypes.py_object]

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

    os.system("lspci | grep RAM")

    DM = DataManagement()  
 
    #DM.set_data({"key1": "value1", "key2": "value2", "key3": "value3", "key4": "value4", "key5": "value5"})
    DM.key1 = "huya"
    DM.key2 = 999   

    print(DM.get_data())

    EM = ExtendedDataManagement()  
    #EM.yoba = "hooita"
    #EM.hoba = 1000

    for i in range(0,10):
        i_key = "yoba"+str(i)
        EM[i_key] = i*2

    print(EM.get_data())

    for i in range(0,10):
        EM.update_data("yoba"+str(i),"pizdec"+str(i))


    print(EM.get_data())

    fiddle(EM)

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

