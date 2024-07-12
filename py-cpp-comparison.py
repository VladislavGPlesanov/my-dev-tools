import numpy
import array as arr
import random as rd
import ctypes

#clib = ctypes.CDLL('/home/vlad/readoutSW/tools/ArraySorter.so')
#
#getFifoData = clib.tb_getFifoData
#getFifoData.restype = ctypes.py_object
#getFifoData.argtypes = [ctypes.py_object, ctypes.c_float]

class TB():

    def __init__(self):
        self.socket_data_buffer = None
        self.python_buffer = None
        self.cpp_buffer = None
        self.infile = None
        self.outfile = None
        self.nwords = None
    
    def reset_main_buffer(self):
        self.socket_data_buffer = arr.array('B')
    
    
    def generate_uint32(self):
        return rd.getrandbits(32).to_bytes(4,'little')
    
    #def get_data_size(self):
    
    def get_tcp_data(self):
        ret_size = min((size, self.get_data_size()))
        ret_size = (ret_size - (ret_size % 4))
        ret = self.socket_data_buffer[:ret_size]
        return ret

if __name__ == '__main__':

     TB = TB()
     TB.reset_main_buffer()

     for i in range(0,10):
        TB.socket_data_buffer.extend(TB.generate_uint32())
        print(len(TB.socket_data_buffer))
        

