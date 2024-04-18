//
// C++ class for reading data from TPX3 chip located on FPGA FIFO (of basil SiTCP type)
//
// Uni-Bonn, Vlad
// 
// Description of python object structures and interplay are described in how-to-firmware.html
//
//
#define PY_SSIZE_T_CLEAN

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <Python.h>
#include <ctime>
#include <chrono>
#include <iomanip>
#include <thread>
#include <string>
#include <numeric>
#include <sstream>
#include <algorithm>
#include <bitset>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL readoutBuffer_ARRAY_API

#include <numpy/arrayobject.h>
#include <tpx3constants.h>

#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC PyInit_Mod(void){
    Py_Initialize();
    import_array();
    if(PyErr_Occurred()){
        std::cerr<<"Failed to import numpy Python module(s)"<<std::endl;
    }
    assert(PyArray_API);
    return NULL;
}

typedef std::chrono::_V2::system_clock::time_point  chronoTime;

//template<class I>
//    auto is_contiguous(I first, I last)
//    { 
//        auto test = true;
//        auto const n = std::distance(first, last);
//        for (auto i = 0; i < n && test; ++i) {
//            test &= *(std::next(first, i)) == *(std::next(std::addressof(*first), i));
//        }        
//        return test;        
//    };

/// global time variable for time stamping:
//possibly move to header file later

std::chrono::steady_clock::time_point l_global_start_time;

extern "C" {

       //////// some utils here /////////
    struct {
        std::vector<long int> tries;
        std::vector<float> times;
    } debugs;

    ///////////////////////////////////////////////////////////

    chronoTime tick(){
        return std::chrono::high_resolution_clock::now();
    }


    float getVectorFloatAvg(std::vector<float> vect){
        
        int vsize = vect.size();
        float sum = std::accumulate(vect.begin(),vect.end(),0.0);
        float result = sum/vsize;
        return result;

    };


    float getVectorLintAvg(std::vector<long int> vect){
        
        int vsize = vect.size();
        long int sum = std::accumulate(vect.begin(),vect.end(),0.0);
        float result = sum/vsize;
        return result;

    };

    int getOccurance(std::vector<uint32_t> vect, uint32_t word){

        return std::count(vect.begin(),vect.end(), word);

    }

    void scream(std::string msg){
        std::cout<<msg<<"\n"<<std::flush;
    };

    uint8_t reverseBitOrder(uint8_t byte){
        byte = (byte & 0xF0) >> 4 | (0x0F) << 4;
        byte = (byte & 0xCC) >> 2 | (0x33) << 2;
        byte = (byte & 0xAA) >> 1 | (0x55) << 1;
        return byte;
    };

    void dumpVector(std::vector<int> vect){
        int cnt = 0;
        for(auto &el: vect){
            std::cout<<"RX["<<cnt<<"]=["<<el<<"]"<<"\t"<<std::flush;
            cnt++;
        }
    };

    void tpxdecoder(std::vector<uint32_t> data){
        
        std::vector<uint32_t> headless_data;

        std::copy_if(data.begin(), data.end(), std::back_inserter(headless_data),
            [](uint32_t part) {
                return(part & 0xF0000000) != 0b0101;
        });
        
        if(headless_data.size() % 2 != 0 ){
            std::cout<<"["<<headless_data.size()<<"] - missing a 32-bit word !!!\n"<<std::flush;
        }else{
            std::cout<<"["<<headless_data.size()<<"] - got even Nr of 32-bit words\n"<<std::flush;
        

            for(int i=0; i<headless_data.size(); i += 2){
                //if(i == 2*i-1){//skipping odd nrs
                //    continue;
                //}
                uint32_t d_one = headless_data[i];
                uint32_t d_two = headless_data[i+1];
                
                std::vector<uint8_t> byte_list_one(reinterpret_cast<uint8_t*>(&d_one), 
                                      reinterpret_cast<uint8_t*>(&d_one) + sizeof(uint32_t));
                std::vector<uint8_t> byte_list_two(reinterpret_cast<uint8_t*>(&d_one), 
                                      reinterpret_cast<uint8_t*>(&d_one) + sizeof(uint32_t));
                if(i<16){
                    std::cout<<"Size of bytelists - [1]: "
                             <<byte_list_one.size()<<", [2];"
                             <<byte_list_two.size()<<"\n"<<std::flush;
                   for(const auto& byte: byte_list_one){
                        std::cout<<"{"<<std::hex<<static_cast<int>(byte)<<"}"<<std::flush;
                   }
                   for(const auto& byte: byte_list_one){
                        std::cout<<"{"<<std::bitset<8>(byte)<<"}"<<std::flush;
                        //std::cout<<"{"<<std::bitset<8>(reverseBitOrder(byte))<<"}"<<std::flush;
                   }
                }
                std::vector<uint8_t> dataByteList; 
                for(int i=1; i<4; i++){
                    dataByteList.push_back(byte_list_one[i]);
                }
                for(int i=1; i<4; i++){
                    dataByteList.push_back(byte_list_two[i]);
                }
                for(const auto &item: dataByteList){
                    std::cout<<"48bdata:<"<<std::bitset<8>(item)<<">\n"<<std::flush;
                }
                /// continue here....

            }

        }

    };

    //void checkNPYversion(){
    //    std::cout<<"numpy api version:"<<NPY_VERSION<<"\n"<<std::flush;
    //    std::cout<<"numpy C version:"<<PyArray_GetNDArrayCVersion<<"\n"<<std::flush;
    //    std::cout<<"numpy Feature version:"<<PyArray_GetNDArrayCFeatureVersion<<"\n"<<std::flush;
    //}

    void checkArray(uint32_t* array, std::size_t size){
        
        for(std::size_t i=0; i<size; i++){
            if(i % 8 ==0 || i==0){
                if(i==0){
                    std::cout<<"["<<&array[i]<<"]\n"<<std::flush;
                }else{
                    std::cout<<&array[i]<<"\n"<<std::flush;
                }
                if(i % 32 == 0 ){
                    std::cout<<"\n"<<std::flush;
                }
            }
        }

    };

    void printArray(uint32_t* array, std::size_t size){
        
        for(std::size_t i=0; i<size; i++){
            if(i % 8 ==0 || i==0){
                if(i==0){
                    std::cout<<std::hex<<"["<<array[i]<<"]\n"<<std::flush;
                }else{
                    std::cout<<std::hex<<array[i]<<"\n"<<std::flush;
                }
                if(i % 32 == 0 ){
                    std::cout<<"\n"<<std::flush;
                }
            }
            if(i>128){
                break;
            }
        }

    };


    void compare_addr(std::vector<std::string> initAddr, std::vector<std::string> endAddr){
        if(initAddr.size()==endAddr.size()){
            std::cout<<"Address sizes [SAME]\n"<<std::flush;
            long int mcount = 0;
            for(std::size_t i=0; i<initAddr.size(); i++){
                if(initAddr.at(i) != endAddr.at(i)){
                    mcount++;
                    std::cout<<"mismatch at ["<<i
                             <<"]: {"<<initAddr.at(i)<<" != "<<endAddr.at(i)<<"};\n"<<std::flush;
                }
            }
            std::cout<<"total nr.of mismatch = "<<mcount<<"\n"<<std::flush;
        }else{
            std::cout<<"Address sizes [DIFFERENT]\n"<<std::flush;
        }

    };

    PyObject* makeNumpyArray(void* data, std::size_t size){
    
        const npy_intp dsize[] = {(npy_intp)(size)};

        PyObject* nuarray = PyArray_SimpleNewFromData(1, dsize, NPY_UINT32, data);
        if(nuarray ==NULL){
            return NULL;
        }
        return nuarray;

    };

    int checkObjectRef(PyObject *obj){
        if(obj == NULL){
            scream("Object is NULL");
            return 0;
        }else{
            return 1;
        }
    };

    void printVectUint(std::vector<uint32_t> vect){    
        //std::cout<<"\nRECDATA:{"<<vect.data()<<"}\n"<<std::flush;
        for(int i = 0; i < vect.size();i++){
            std::cout<<vect.at(i)<<"|"<<std::flush;
            if(i !=0 && i % 8 == 0){
              scream("\n");
            }
            if(i==63){
                break;
            }
        }
        std::cout<<"and "<<vect.size()-64<<" more\n"<<std::flush;
    }

    // needs adjustments 
    //double duration_time(chronoTime time){
    //     float millisec =  std::chrono::duration<float, std::milli>(
    //                       time - 0.0).count();
    //                        |      |
    //                        |      float
    //                        chronoTime
    //     return millisec;
    //};

    float time_to_read(chronoTime start_time){
         float millisec =  std::chrono::duration<float, std::milli>(
                           std::chrono::high_resolution_clock::now() - start_time).count();
  
         return millisec;
    };


    float timeDiff(chronoTime t0, chronoTime t1){
         float millisec = std::chrono::duration<float, std::milli>(t1 - t0).count();
         return millisec;
    };


    void updateTimeStamp(float &t_global, chronoTime t_start, chronoTime t_end){
        float t_diff = timeDiff(t_start, t_end);
        //std::cout<<"\t OLO:"<<t_diff<<"\t"<<std::flush;
        t_global = t_global + t_diff;
    };

    void updateTimes(float &t_global, float t_end){
        t_global = t_global + t_end;
    };

    void checkObject(PyObject *obj, std::string opt){
        PyObject_Print(obj, stdout, 0);
        if(opt=="length"){
            Py_ssize_t qlen = PyObject_Length(obj);
            std::cout<<"\nobject-length: "<<qlen<<"\n"<<std::flush; 
        }else if(opt=="size"){
            Py_ssize_t qsize = PyObject_Size(obj);
            std::cout<<"\nobject-size: "<<qsize<<"\n"<<std::flush; 
        }else if(opt=="type"){
            std::cout<<"\nobject-type: "<<PyObject_Type(obj)<<"\n"<<std::flush; 
        }else{
            ///;
        }
    };

    void copyDataArray(uint32_t* input, uint32_t* output, std::size_t arrSize){
    
        std::memcpy(output, input, arrSize);
        //for(std::size_t i=0; i<arrSize; ++i){
        //    output[i] = input[i];
        //};
    
    };

    void updateArray(uint32_t* data_in, uint32_t* data_out, std::size_t inputArrSize, std::size_t &outArrSize){

       std::memcpy(data_out + inputArrSize, data_in, inputArrSize * sizeof(uint32_t));
       outArrSize += inputArrSize;

    }

    uint32_t* updateData(uint32_t* current_data, uint32_t* data_in, int len_data_in){

         std::size_t data_in_size = static_cast<std::size_t>(len_data_in);
         
        //std::memcpy(dest,orig,size)
        std::memcpy(current_data, data_in, data_in_size * sizeof(uint32_t));

        return current_data;
    };

    void mergeArrays(uint32_t* current_data, std::size_t cur_data_size, uint32_t* new_data, std::size_t new_data_size){
         
         uint32_t* resizedArray = new uint32_t[cur_data_size+new_data_size];
         std::memcpy(resizedArray, current_data, cur_data_size * sizeof(uint32_t));
         std::memcpy(resizedArray + cur_data_size, new_data, new_data_size * sizeof(uint32_t));
         delete[] current_data;
         current_data = resizedArray;
 
    };


/////////////// TODO: for these two make error handling if they work ////////////////
long int getNerror(PyObject *list){

    long int n_errors = 0;
    Py_ssize_t listlen = PyObject_Length(list);
    for(Py_ssize_t it = 0; it < listlen; ++it){
        PyObject *item = PyList_GetItem(list, it);
        if(item!=NULL){
            Py_INCREF(item);
            n_errors+=PyLong_AsLong(item); 
            Py_DECREF(item);
        }else{
            scream("can not access item in:");
            PyObject_Print(list,stdout,0);
        }
    };
    return n_errors;

};

int printobject(PyObject *obj){return PyObject_Print(obj,stdout,0);};

PyObject* getChip(PyObject *self){

    PyObject *chip; 

    chip = PyObject_GetAttrString(self, "chip");

    return chip;

}

PyObject* getFifo(PyObject *self){
    
    PyObject *chip, *fifo; 

    chip = PyObject_GetAttrString(self, "chip");
    
    fifo = PyObject_GetItem(chip, PyUnicode_FromString("FIFO"));

    return fifo;

};

PyObject* emptyList(){//probs working 
    PyObject *emList = PyList_New(8);
    for(int i=0; i<8; i++){
        PyObject *val = PyLong_FromLong(0);
        PyList_SetItem(emList, i, val);
    }
    return emList;
}

PyObject* fillPyArray(std::vector<uint32_t> &data){// works perfectly fine

    const std::size_t datasize = data.size();
    const npy_intp pyarrsize[] = {(npy_intp)(datasize)};

    PyObject* pyarray = PyArray_SimpleNew(1,pyarrsize,NPY_UINT32);
    if(pyarray==NULL){
        scream("ARRAY IS NULL!");
    }

    uint32_t* thisData = reinterpret_cast<uint32_t*>(
            PyArray_DATA(reinterpret_cast<PyArrayObject*>(pyarray)));

    thisData = &data[0];

    return pyarray;

};

std::vector<uint32_t> fillVectorFromPyData(PyArrayObject *pyarray){

    uint32_t* result_data = (uint32_t*)PyArray_DATA(pyarray);
    npy_intp* arr_dim = PyArray_DIMS(pyarray);
    npy_intp* arr_size = &arr_dim[0];
    std::vector<uint32_t> result_vector(result_data, result_data + *arr_size);

    return result_vector;

};

PyObject* fillPyList(std::vector<long int> &stats){// maybe works

    Py_ssize_t statlen = (Py_ssize_t)(stats.size());
    PyObject * pylist = PyList_New(0);
    for(Py_ssize_t ith = 0; ith < statlen; ++ith ){
        PyObject *pyval = PyLong_FromLong(stats[ith]);
        PyList_Append(pylist, pyval);
    }

    return pylist;

};

//////////////////////////////////////////////////////////////////////////
///---------- working numpy C-API functions --------------------------
//////////////////////////////////////////////////////////////////////////

//TODO: make a local version
PyObject* readFifoStatus(PyObject *self, const char* option){       

// test case when passing self in fifo_readout 
// object should be: <tpx3.fifo_readout.FifoReadout>
    //PyGILState_STATE gstate = PyGILState_Ensure(); 
    PyObject *res;
    /*returns status and counters for FIFO for following options:
        get_rx_sync_status,
        get_rx_en_status,
        get_rx_fifo_discard_count,
        get_rx_decode_error_count
    */
    if(self != NULL){
        Py_INCREF(self);
        //scream("readFifoStatus::self not null");
        res = PyObject_CallMethod(self,option,NULL);// not null
    }else{
        scream("readFifoStatus::self object is NULL!");
        return emptyList();
    }
   
    if(PyObject_Size(res)<=0){
        std::cout<<"result for status {"<<option<<"} is <=0!\n"<<std::flush;
        return emptyList();
    }
    Py_ssize_t llen = PyList_Size(res);
    PyObject *list = PyList_New(0);
    for(Py_ssize_t ith = 0; ith < llen; ++ith ){
        Py_INCREF(res);
        PyList_Append(list, PyList_GetItem(res,ith));
        Py_DECREF(res);
    }
    std::cout<<"readFifoStatus::obj=["<<option
              <<"] list before = "<<printobject(res)
             <<", list after = "<<printobject(list)<<"\n"<<std::flush;

    Py_DECREF(res);
    Py_DECREF(self);
    //delete res;
    ///PyGILState_Release(gstate);  //REENABLE THIS LATER!
    return list;
    // if one returns "res" object directly -> segfault!                     
};

std::vector<long int> local_readFifoStatus(PyObject *self, const char* option){       

// test case when passing self in fifo_readout 
// object should be: <tpx3.fifo_readout.FifoReadout>
    std::vector<long int> rxstatus;
    scream("[ebal]");
    PyGILState_STATE gstate = PyGILState_Ensure(); 
    PyObject *res;
    /*returns status and counters for FIFO for following options:
        get_rx_sync_status,
        get_rx_en_status,
        get_rx_fifo_discard_count,
        get_rx_decode_error_count
    */
    scream("[tvoy]");
    if(self != NULL){
        res = PyObject_CallMethod(self,option,NULL);// not null
    }else{
        scream("call object is NULL!");
    }

    scream("[suchiy]");

    long int llen = reinterpret_cast<long int>(PyObject_Length(res));

    std::cout<<"statlen="<<PyObject_Length(res)<<"\n"<<std::flush;
    std::cout<<"statsize="<<PyObject_Size(res)<<"\n"<<std::flush;
    if(PyObject_Size(res)<=0){
        std::cout<<"result for status {"<<option<<"} is <=0!"<<"\n"<<std::flush;
        return {0,0,0,0,0,0,0,0};
    }else{
        for(int i=0;i<llen;i++){
           Py_INCREF(res);
           rxstatus.push_back(reinterpret_cast<long int>(PyList_GetItem(res,i)));
           Py_DECREF(res);
        }
    }
    scream("[rot1]");
    Py_DECREF(res);
    Py_DECREF(self);
    PyGILState_Release(gstate); 
    return rxstatus;

};




PyObject* querryDataFromFifo(PyObject *chip){

    ///////////////////////////////////////////////////////////////
    // works for (self.chip['FIFO']) --- passed to ---> (*chip)
    //           (basil.HL.sitcp_fifo.sitcp_fifo)
    // 
    //       for self.chip one gets -> (tpx3.tpx3.TPX3)
    //      
    //////////////////////////////////////////////////////////////
    // put mutex lock on the bytecode of the object in interpretr
    // to have thread-safe execution
    PyGILState_STATE gstate = PyGILState_Ensure(); 

    PyObject *result; 

    if(chip != NULL){
        std::cout<<"ChipNotNULL! - OK!"<<std::endl;

        std::cout<<PyCallable_Check(chip)<<std::endl; // chip is not callable fsr
        PyObject *meth = PyObject_GetAttrString(chip, "get_data"); // but its attributes are...
        PyObject *meth2 = PyObject_GetAttrString(chip, "FIFO_SIZE"); //
        std::cout<<PyCallable_Check(meth)<<std::endl; // YES
        std::cout<<PyCallable_Check(meth2)<<std::endl; // YES
        std::cout<<"["<<PyCallable_Check(meth2)<<"]"<<std::endl; // YES
        std::cout<<"["<<PyObject_HasAttrString(chip,"FIFO_SIZE")<<"]"<<std::endl; // YES

        // increase and then decreaase reference to *chip
        Py_INCREF(chip); 
        result = PyObject_CallMethod(chip,"get_data",NULL);
        std::cout<<"FIFO_SIZE:("<<PyObject_CallMethod(chip, "FIFO_SIZE",NULL)<<")\n"<<std::flush;
        std::cout<<"FIFO_SIZE[1]:("<<PyObject_CallFunction(chip, "FIFO_SIZE",NULL)<<")\n"<<std::flush;
        std::cout<<"FIFO_SIZE[2]:("<<PyObject_CallFunction(chip, "FIFO_SIZE",NULL)<<")\n"<<std::flush;
        Py_DECREF(chip);
 
    }else{
        PyErr_SetString(PyExc_KeyError,"chip is null\nObject was not passed correctly");
        result = 0;
    }
    Py_DECREF(chip);
    //releases mutex lock
    PyGILState_Release(gstate); 

    return result;

}

//
// rewrite this one wit ha direct call to sitcp_fifo._intf!
//
//std::vector<uint32_t> intf_localQuerryFifo(PyObject *chipFifo){
//
//    PyGILState_STATE gstate = PyGILState_Ensure(); 
//        
//    //PyArrayObject *result;
//
//    if(chipFifo != NULL){
//        //PyObject *meth = PyObject_GetAttrString(chipFifo, "get_data");
//        // increase and then decrease reference to *chipFifo
//        Py_INCREF(chipFifo); 
//        result = reinterpret_cast<PyArrayObject*>(PyObject_CallMethod(chipFifo,"get_data",NULL));
//        Py_DECREF(chipFifo);
// 
//    }else{
//        scream("chipFifo is null\nObject was not passed correctly");
//        result = 0;
//    }
//
//    // test below
//    Py_INCREF(result);
//    std::vector<uint32_t> result_vector = fillVectorFromPyData(result);// technically works
//
//    //if(result_vector.size() > 0){ 
//    //    std::cout<<std::boolalpha<<is_contiguous(result_vector.begin(),result_vector.end());
//    //}
//
//    Py_DECREF(result);
//    //releases mutex lock
//    PyGILState_Release(gstate); 
//
//    return result_vector;
//
//};

std::vector<uint32_t> localQuerryFifo(PyObject *chipFifo){

    PyGILState_STATE gstate = PyGILState_Ensure(); 
        
    PyArrayObject *result; 

    if(chipFifo != NULL){
        // increase and then decrease reference to *chipFifo
        Py_INCREF(chipFifo); 
        result = reinterpret_cast<PyArrayObject*>(PyObject_CallMethod(chipFifo,"get_data",NULL));
        Py_DECREF(chipFifo);
 
    }else{
        scream("chipFifo is null\nObject was not passed correctly");
        result = 0;
    }

    // test below
    Py_INCREF(result);
    std::vector<uint32_t> result_vector = fillVectorFromPyData(result);// technically works

    //if(result_vector.size() > 0){ 
    //    std::cout<<std::boolalpha<<is_contiguous(result_vector.begin(),result_vector.end());
    //}

    Py_DECREF(result);
    //releases mutex lock
    PyGILState_Release(gstate); 

    return result_vector;

};


long int localFifoSize(PyObject *chipFifo){
    /*
     *   Requires PyObject => self.chip['FIFO']
     *   only for internal use inside this class
     */
 
    PyGILState_STATE gstate = PyGILState_Ensure(); 

    long int ndatawords = 0;

    PyObject *fsize = PyObject_GetItem(chipFifo, PyUnicode_FromString("FIFO_SIZE"));
    if(fsize == NULL){
        scream("ERROR ArraySorter::testFifosize: chipFifo->fsize is NULL!");
        return -1;      
    }else{
        ndatawords = PyLong_AsLong(fsize);
        std::cout<<"READING:{"<<ndatawords<<"}words in fifo\n"<<std::flush;
        if(ndatawords == -1 && PyErr_Occurred()){
            scream("ERROR ArraySorter::testFifosize: chipFifo->fsize->result is NULL!");
        }
    }
    Py_DECREF(fsize);
    Py_DECREF(chipFifo);

    PyGILState_Release(gstate); 

    return ndatawords;

};

// TODO: make a copy for internal use inside of this class
PyObject* getFifoData(PyObject *chipFifo, float t_interval){ 
    /*
     Accepts << self.chip['FIFO'] >> object from python side
    */
    // put mutex lock on the bytecode of the object in interpretr
    // to have thread-safe execution 

    PyGILState_STATE gstate = PyGILState_Ensure(); 
    Py_Initialize();
    import_array();

    std::vector<uint32_t> temp;
    std::vector<uint32_t> fifo_data;

    //Py_INCREF(chipFifo);
    //long int fifowords = localFifoSize(chipFifo);
    //Py_DECREF(chipFifo);
    //std::cout<<"FIFO size before reading = "<<fifowords<<"\n"<<std::flush;

    long int cnt = 0; 

    //auto start_time = std::chrono::high_resolution_clock::now();
    auto start_time = std::chrono::high_resolution_clock::now();
    while(time_to_read(start_time) < t_interval){
        temp = localQuerryFifo(chipFifo);
        if(temp.size()!=0){
            fifo_data.insert(fifo_data.end(),
                             temp.begin(),
                             temp.end());
        }
        temp.clear();
        cnt++;
    }
    ///////////////////////////////////////////////////////////////
    auto end_time = tick();

    float tdiff = timeDiff(start_time,end_time);
    
    std::cout<<"read="<<cnt<<"times\n"<<std::flush;
    std::cout<<"loop time="<<tdiff<<"[ms]\n"<<std::flush;

    //////test below//////////
    PyObject* array = fillPyArray(fifo_data);// works, substitutes lines above 
    

    if(fifo_data.data()!=NULL){
        PyGILState_Release(gstate); 
        Py_DECREF(chipFifo);
        return array;

    }else{/*else block will stay as is*/
        fifo_data.clear();
        Py_DECREF(chipFifo);
        PyGILState_Release(gstate); 
        return Py_None;
    } 

    Py_DECREF(chipFifo);
    //release mutex lock
    Py_Finalize();
    PyGILState_Release(gstate); 
    return Py_None;

};

/////////// needs testing ///////////
//
// grand theft code from basil sitcp_fifo->get_data()
//
//std::vector<uint32_t> getDatafromIntf(){
PyObject *getDataFromIntf(PyObject *intf){

    Py_ssize_t fifo_size = reinterpret_cast<Py_ssize_t>(
        PyObject_CallMethod(intf, "_get_tcp_fifo_size",NULL));
    Py_ssize_t fifo_int_size = (fifo_size - (fifo_size % 4)) / 4;
    PyObject *result = PyObject_CallMethod(intf, "_get_tcp_data", "i", fifo_int_size*4);
    return result;

};
///////////////////////////
std::vector<uint32_t> local_getFifoData(PyObject *chipFifo, float interval){

    PyGILState_STATE gstate = PyGILState_Ensure(); 
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<uint32_t> temp;
    std::vector<uint32_t> fifo_data;

    long int cnt = 0;   

    while(time_to_read(start_time) < interval){
        Py_INCREF(chipFifo);
        temp = localQuerryFifo(chipFifo);
        Py_DECREF(chipFifo);
        if(temp.size()!=0){
            fifo_data.insert(fifo_data.end(),
                             temp.begin(),
                             temp.end());
        }
        temp.clear();
        cnt++;
    }
    ///////////////////////////////////////////////////////////////
    auto end_time = tick();

    float tdiff = timeDiff(start_time,end_time);
    
    std::cout<<"tried="<<cnt<<"times, redorded "<<fifo_data.size()<<" events\n"<<std::flush;
    std::cout<<"loop time="<<tdiff<<"[ms]\n"<<std::flush;

    if(fifo_data.size()!=0){
        debugs.tries.push_back(cnt);
        debugs.times.push_back(tdiff);
    }    

    PyGILState_Release(gstate);  //REENABLE THIS LATER!
    return fifo_data;

};




////////////////////////////////////////////////////////
/// WIP, to be tested below///
////////////////////////////////////////////////////////

PyObject *testFifoSize(PyObject *self){

    /* 
     * trying to access fifo size from the top object "self"
     */
 
    PyGILState_STATE gstate = PyGILState_Ensure(); 
    //Py_Initialize();
    //import_array();

    PyObject *chip = PyObject_GetAttrString(self,"chip");

    std::size_t nent = 1;
    //const npy_intp size[] = {(npy_intp)(nent)};
    //scream("KAKOGO");
    
    long int ndatawords = 0;
    uint32_t temp;

    scream("HUYA");
    if(chip == NULL){
        scream("ERROR ArraySorter::testFifosize: chip is NULL!");
        //return NULL; // use NULL for pointers...
        //return -1;
    }else{
        PyObject *fifo = PyObject_GetItem(chip,PyUnicode_FromString("FIFO"));
        if(fifo == NULL){
            scream("ERROR ArraySorter::testFifosize: chip->fifo is NULL!");
            //return -1;      
            //return NULL;      
        }else{
            PyObject *fsize = PyObject_GetItem(fifo, PyUnicode_FromString("FIFO_SIZE"));
            if(fsize == NULL){
                scream("ERROR ArraySorter::testFifosize: chip->fifo->fsize is NULL!");
                //return -1;      
                //return NULL;      
            }else{
                scream("NIHERA");
                ndatawords = PyLong_AsLong(fsize);
                //temp = static_cast<uint32_t>(ndatawords);
                scream("NE");
                //std::cout<<"READING:{"<<ndatawords<<"}words in fifo\n"<<std::flush;
                //long ndatawords = PyLong_AsLong(fsize);
                ////smth = PyLong_FromLong(*ndatawords);
                //result = PyLong_FromLong(ndatawords);
                //PyObject_Print(result,stdout,0);
                if(ndatawords == -1 && PyErr_Occurred()){
                    scream("ERROR ArraySorter::testFifosize: chip->fifo->fsize->result is NULL!");
                }
             
            }
            Py_DECREF(fsize);
        }
        Py_DECREF(fifo);
    }
    Py_DECREF(chip);
    Py_DECREF(self);
    scream("PROISHODIT");

    // TODO: segfault after this point 
    //PyObject *result = PyArray_SimpleNew(1,size,NPY_UINT32);
    //scream("SUKa"); 

    //uint32_t* thisFifoSize = reinterpret_cast<uint32_t*>(
    //            PyArray_DATA(reinterpret_cast<PyArrayObject*>(result)));
     
    PyGILState_Release(gstate); 

    return PyLong_FromLong(ndatawords);

    //// might be working but not sure yet
    //return result;

};

PyObject* testOutput(PyObject *self){
    /*
    *
    * playground to test if i can read & return status of RX 
    * back to python 
    *
    */

    PyGILState_STATE gstate = PyGILState_Ensure(); 
    PyObject *chip; // top object reference
    PyObject *ENA, *DATA_DEL, *INV, *SAM_EDGE, *READY;

    std::vector<int> rx_status_ena;
    std::vector<int> rx_status_ddel;
    std::vector<int> rx_status_inv;
    std::vector<int> rx_status_sedge;
    std::vector<int> rx_status_rdy;
 
    chip = PyObject_GetAttrString(self, "chip");
    if(chip!=NULL){
        scream("chip [OK]");
        for(int i = 0; i<8; i++){
            std::string ithRX = "RX"+std::to_string(i);
            PyObject *irx = PyObject_GetItem(chip, PyUnicode_FromString(ithRX.c_str()));
            if(irx != NULL){
                scream(ithRX+" is [OK]");
                ENA = PyObject_GetItem(irx, PyUnicode_FromString("ENABLE"));            
                DATA_DEL = PyObject_GetItem(irx, PyUnicode_FromString("DATA_DELAY"));            
                INV = PyObject_GetItem(irx, PyUnicode_FromString("INVERT"));            
                SAM_EDGE = PyObject_GetItem(irx, PyUnicode_FromString("SAMPLING_EDGE"));            
                READY = PyObject_GetItem(irx, PyUnicode_FromString("READY"));            
                if(ENA != NULL){
                    int status_ena = PyLong_AsLong(ENA);
                    int status_ddel = PyLong_AsLong(DATA_DEL);
                    int status_inv = PyLong_AsLong(INV);
                    int status_sedge = PyLong_AsLong(SAM_EDGE);
                    int status_rdy = PyLong_AsLong(READY);
                    rx_status_ena.push_back(status_ena); 
                    rx_status_ddel.push_back(status_ddel); 
                    rx_status_inv.push_back(status_inv); 
                    rx_status_sedge.push_back(status_sedge); 
                    rx_status_rdy.push_back(status_rdy); 
                    //scream("===============");
                    Py_DECREF(ENA);
                    Py_DECREF(DATA_DEL);
                    Py_DECREF(INV);
                    Py_DECREF(SAM_EDGE);
                    Py_DECREF(READY);
                    Py_DECREF(irx);
                }else{
                   scream("ENA is NULL");
                   return Py_None;   
                }
            }else{
               scream(ithRX+" is NULL");
               return Py_None;    
            }
        }
    }else{
        scream("chip is NULL");
        return Py_None;
    }
    scream("ENABLE:");
    dumpVector(rx_status_ena);
    scream("\nREADY(SYNC):");
    dumpVector(rx_status_rdy);
    scream("\nINVERT:");
    dumpVector(rx_status_inv);
    scream("\nSAMPLING_EDGE:");
    dumpVector(rx_status_sedge);
    scream("\nDATA_DELAY:");
    dumpVector(rx_status_ddel);

    rx_status_ena.clear();
    rx_status_ddel.clear();
    rx_status_inv.clear();
    rx_status_sedge.clear();
    rx_status_rdy.clear();

    Py_DECREF(chip);

    PyGILState_Release(gstate); 
    return Py_None;

};

PyObject* getStatusAllRx(PyObject *self, const char* REG){

    // Can be used to directly obtain:
    // [ENABLE, DATA_DELAY, INVERT, SAMPLING_EDGE, READY] register values
    // of RX channels

    PyGILState_STATE gstate = PyGILState_Ensure(); 
    PyObject *chip; // top object reference
    PyObject *STAT; // statusobject reference

    std::vector<int> rx_status;
 
    chip = PyObject_GetAttrString(self, "chip");
    if(chip!=NULL){
        for(int i=0; i<tpx3::NUM_RX_CHAN ; i++){ //TODO: put nr of RX channels in a const file.
            std::string ithRX = "RX"+std::to_string(i);
            PyObject *irx = PyObject_GetItem(chip, PyUnicode_FromString(ithRX.c_str()));
            if(irx != NULL){
                STAT = PyObject_GetItem(irx, PyUnicode_FromString(REG));           
                if(STAT != NULL){
                    int status = PyLong_AsLong(STAT);
                    rx_status.push_back(status); 
                    Py_DECREF(STAT);
                    Py_DECREF(irx);
                }else{
                   std::stringstream ss;
                   ss<<"Pointer to (" << REG << ") is NULL!";
                   scream(ss.str());
                   return Py_None;
                }
            }else{
               scream(ithRX+" is NULL");
               return Py_None;    
            }
        }
    }else{
        scream("chip is NULL");
        return Py_None;
    }
    std::stringstream stream;
    stream<<"Dumping "<<REG<<"]:\n";
    scream(stream.str());
    dumpVector(rx_status);
    rx_status.clear();

    Py_DECREF(chip);

    PyGILState_Release(gstate); 
    return Py_None;
};

int checkSHUTTER(PyObject *self){

    //
    // 
    // OT ETU EBALU PROCHEKAI! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //
    //

    PyObject *chip = PyObject_GetAttrString(self, "chip"); 
    PyObject *control = PyObject_GetItem(chip, PyUnicode_FromString("CONTROL"));
    if(control == NULL){
        scream("\n [[ object \"CONTROL\" was not found!\n");
        return -1;
    }
    Py_INCREF(control);
    PyObject *shutter = PyObject_GetItem(control, PyUnicode_FromString("SHUTTER"));
    if(shutter==NULL){
        scream("\n [[ object \"CONTROL\" -> \"SHUTTER\" was not found!\n");
        return -1;   
    }
    Py_INCREF(shutter);

    int status = PyLong_AsLong(shutter);

    Py_DECREF(shutter);
    Py_DECREF(control);

    return status;

};

PyObject *setRegister(PyObject *self, const char* SETTING, int value){

    PyGILState_STATE gstate = PyGILState_Ensure(); 
    PyObject *chip; // top object reference
    PyObject *REG; // control reg object reference

    scream("<<< RESETING RX's >>>");

    chip = PyObject_GetAttrString(self, "chip");

    if(chip!=NULL){
        for(int i=0; i<tpx3::NUM_RX_CHAN ; i++){
            std::string ithRX = "RX"+std::to_string(i);
            PyObject *irx = PyObject_GetItem(chip, 
                                             PyUnicode_FromString(ithRX.c_str()));
            if(irx != NULL){
                //PyObject *REG = PyObject_GetItem(irx, 
                //                                 PyUnicode_FromString(SETTING.c_str()));
                Py_INCREF(chip);
                PyObject_SetItem(chip,REG,PyLong_FromLong(value));
                //PyObject_SetItem(chip,REG,value);
                std::stringstream ss;
                ss<<"Set ["<<SETTING<<"]="<<PyLong_FromLong(value);
                scream(ss.str());
                ss.clear();
                Py_DECREF(REG);
            }else{
                std::stringstream ss;
                ss<<"Register "<<SETTING<<"["<<REG<<"] is NULL";
                scream(ss.str());
                ss.clear();
                return Py_None;             
            }
        }
    }else{
        scream("chip is NULL");
        return Py_None;
    }

    Py_DECREF(chip);

    PyGILState_Release(gstate); 
    return Py_None;

};

PyObject* dequeDataOut(PyObject *self, float interval){/*works good*/

  PyGILState_STATE gstate = PyGILState_Ensure(); 
  // 2 lines below should be used in "main" function only, others 
  // should use only PyGILState f-ncs.
  Py_Initialize();
  import_array();

  // Allocating memory for output tuple
  PyObject *tuple = PyTuple_New(3);
  // add a check if *tuple is NULL!
  // release GIL if yes

  // initializin' objects for dat and rx error counters
  PyObject *fifo, *fifoData, *fifo_decerr, *fifo_discerr; 
  // vector for local readout of fifo data
  std::vector<uint32_t> data;

  PyObject *chip = PyObject_GetAttrString(self,"chip");
  fifo = PyObject_GetItem(chip,PyUnicode_FromString("FIFO"));

  //PyObject *intf = PyObject_GetAttrString(fifo, "_intf");
  //std::cout<<"looking for interface:"<<intf<<"!"<<std::flush;

  // recod fifo data locally
  Py_INCREF(fifo);
  data = local_getFifoData(fifo,interval); 
  Py_DECREF(fifo);

  if(debugs.tries.size()!=0){
     std::cout<<"N tries = "<<debugs.tries.size()<<"!\n"<<std::flush;
     std::cout<<"AVG(tries) = "<<getVectorLintAvg(debugs.tries)<<"!\n"<<std::flush;
     std::cout<<"N times = "<<debugs.times.size()<<"!\n"<<std::flush;
  }
  
  // debug check of contents of recorded data
  //printVectUint(data);
  
  // this on puts vector.data() into numpy array
  fifoData = fillPyArray(data);
  
  // reading fifo discard and decoding errors
  Py_INCREF(self); 
  fifo_discerr = readFifoStatus(self,tpx3::CNTR_DISCARD_ERR);
  Py_DECREF(self); 
  Py_INCREF(self); 
  fifo_decerr = readFifoStatus(self,tpx3::CNTR_DECODING_ERR);
  Py_DECREF(self); 

  // filling output tuple 
  // as (<fifo data>, <listof discard errors>, <list of decode errors>)
  // 
  // should become:
  // (<fifo data>, 
  // <timestamp_beg>, 
  // <timestamp_end>, 
  // <list of disc. err.>, 
  // <list of dec. err.>)
  // to fit _data_deque operation in fifo_readout.py and handleData in scan_base.py
  Py_INCREF(fifoData);
  PyTuple_SetItem(tuple, 0, fifoData);
  Py_DECREF(fifoData);

  Py_INCREF(fifo_discerr);
  PyTuple_SetItem(tuple, 1, fifo_discerr);
  Py_DECREF(fifo_discerr);

  Py_INCREF(fifo_decerr);
  PyTuple_SetItem(tuple, 2, fifo_decerr);
  Py_DECREF(fifo_decerr);

  ///// finita la comedia! //////
  Py_DECREF(chip); 
  PyGILState_Release(gstate);   
  return tuple;

};
//---------------------------------------------------------------------------------
//
//  Substitute for tpx3/fifo_readout/readout function
//
//---------------------------------------------------------------------------------
PyObject* readoutToDeque(PyObject *self, PyObject *deque, float interval){
//PyObject* readoutToDeque(PyObject *self, PyObject *deque, PyObject *workerCNT, float interval){

  PyGILState_STATE gstate = PyGILState_Ensure();
  // 2 lines below should be used in "main" function only, others 
  // should use only PyGILState f-ncs.
  Py_Initialize();
  import_array();
 
  scream("[DEBUG] starting readout"); 
  if(l_global_start_time == std::chrono::steady_clock::time_point()){
    l_global_start_time == std::chrono::steady_clock::now();
  }

  // GETTING ALL GLOBAL OBJECTS BEFOREHAND
 
  PyObject *stopreadout = PyObject_GetAttrString(self,"stop_readout");
  PyObject *forcestop = PyObject_GetAttrString(self,"force_stop");
  PyObject *stopflag = PyObject_GetAttrString(self,"STOPFLAG");
  PyObject *workercnt = PyObject_GetAttrString(self,"worker_cnt");

  // last chunk read time
  float last_read = 0.0, curr_time = 0.0;
  // for now, waiting only for a stop signal from user input
  // for some fuck-reasons PyObject_IsTrue evaluates "unset" state to logic 1...
  // thus logic in while loop is inverted wrt to python side
  //
  int glob_tries = 0;
  std::vector<float> iter_times; 
  //while(true){ 
  
  while (PyObject_IsTrue(forcestop) || PyObject_IsTrue(stopreadout)){

    auto iter_start = std::chrono::steady_clock::now();
    scream("[DEBUG] (while) iteration starts"); 
    // Allocating memory for output tuple
    // add a check if *tuple is NULL!
    // release GIL if yes
    PyObject *tuple = PyTuple_New(5);
    // (array(data), t_0, t_end, rx_disc, rx_dec)
    if(tuple==NULL){
        scream("Could not instantiate data tuple");
        PyGILState_Release(gstate); 
        return NULL;
    };

    scream("[DEBUG] tuple defined");
    // initializin' objects for dat and rx error counters
    PyObject *fifo, *fifoData, *fifo_decerr, *fifo_discerr; 
    // vector for local readout of fifo data
    std::vector<uint32_t> data;

    PyObject *chip = PyObject_GetAttrString(self,"chip");
    fifo = PyObject_GetItem(chip,PyUnicode_FromString("FIFO"));

    Py_INCREF(self);
    int shutStat = checkSHUTTER(self);
    Py_DECREF(self);

    scream("[DEBUG] got fifo object"); 
    long int data_available = localFifoSize(fifo);
    std::cout<<"[DEBUG] shutter=("
             <<shutStat<<"), data available = ("<<data_available<<")\n"<<std::flush;
    // recod fifo data locally
    Py_INCREF(fifo);
    data = local_getFifoData(fifo,interval); 
    Py_DECREF(fifo);
    
    scream("[DEBUG] recorded data");
    std::cout<<"Contains "<<getOccurance(data,0)<<" zeros\n"<<std::flush;
    long int n_words = data.size();
    if(n_words == 0 ){
        std::chrono::milliseconds sleeptime((int)interval);// this one accepts integers
        std::this_thread::sleep_for(sleeptime);
        std::cout<<"waiting for data for one readout interval ("<<interval<<" ms)\n"<<std::flush; 
        continue;
    }
    if(n_words != 0 ){
        float dloss = 100.0 - ((float)n_words/(float)data_available)*100.0;
        std::cout<<"[DEBUG] LOST "<<dloss<<"\% of data\n"<<std::flush;
    }
    // debug check of contents of recorded data
    //printVectUint(data);
    std::cout<<"\n[DEBUG] got *"<<n_words<<"* words\n"<<std::flush;
    // this on puts vector.data() into numpy array
    fifoData = fillPyArray(data);
    scream("[DEBUG] Put data into numpy array");    
    // reading fifo discard and decoding errors
    Py_INCREF(self); 
    fifo_discerr = readFifoStatus(self,tpx3::CNTR_DISCARD_ERR);

    scream("[DEBUG] got discard errors");
    Py_INCREF(fifo_discerr);
    long int n_errdisc = getNerror(fifo_discerr);
    Py_DECREF(fifo_discerr);
    Py_DECREF(self); 

    Py_INCREF(self); 
    fifo_decerr = readFifoStatus(self,tpx3::CNTR_DECODING_ERR);
    scream("[DEBUG] got decode errors");
    Py_INCREF(fifo_decerr);
    long int n_errdec = getNerror(fifo_decerr);
    Py_DECREF(fifo_decerr);
    Py_DECREF(self); 

    scream("[DEBUG] Obtained rx errors");
    std::cout<<"[DEBUG] N_DISCARD="<<n_errdisc
             <<", N_DECODE="<<n_errdec<<"\n"<<std::flush;
    // update timestamp   
    auto iter_end = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - iter_start);

    //float t_start = std::chrono::duration_cast<std::chrono::milliseconds>(
    //        l_global_start_time.time_since_epoch()).count();

    float t_end = std::chrono::duration_cast<std::chrono::milliseconds>(
            (l_global_start_time + iter_end).time_since_epoch()).count();

    curr_time = last_read + t_end;
    scream("[DEBUG] got timestamps"); 
    // filling output tuple 
    // (<fifo data>, 
    // <timestamp_beg>, 
    // <timestamp_end>, 
    // <list of disc. err.>, 
    // <list of dec. err.>)
    bool b_stopread = PyObject_IsTrue(stopreadout);
    bool b_forcestop = PyObject_IsTrue(forcestop);
    //bool b_stopflag = PyObject_IsTrue(stopflag);

    std::cout<<"[DEBUG] SIGNAL CONTROL ["<<b_stopread
             //<<"|"<<b_forcestop<<"|"<<b_stopflag<<"]\n"<<std::flush;
             <<"|"<<b_forcestop<<"]\n"<<std::flush;
    //if(!PyObject_IsTrue(stopreadout) || !PyObject_IsTrue(forcestop) || b_stopflag ){
    if(!PyObject_IsTrue(stopreadout) || !PyObject_IsTrue(forcestop)){

    //PyObject *stopsig = PyObject_GetAttrString(self, "STOPFLAG");

    //if(PyLong_AsLong(stopsig)==1){
       std::cout<<"\n\n---- received STOP/FORCE_STOP signal! ---- \n\n"<<std::flush;

       PyObject *endTuple = PyTuple_New(1);
       Py_INCREF(endTuple);
       PyTuple_SetItem(endTuple, 0, Py_None);
       Py_DECREF(endTuple);
       Py_INCREF(deque);
       PyObject *result = PyObject_CallMethod(deque, "append", "(O)", tuple);
       if(result==NULL){
            scream("[DEBUG] COULD NOT APPEND Py_None!");
       }
       Py_DECREF(deque);
       Py_DECREF(result);
       sleep(2);
       break;
    }

    Py_INCREF(fifoData);
    PyTuple_SetItem(tuple, 0, fifoData);
    Py_DECREF(fifoData);
    scream("[DEBUG] Put fifoData in tuple"); 
    PyTuple_SetItem(tuple, 1, PyFloat_FromDouble((double)last_read));
    scream("[DEBUG] Put start time in tuple"); 
    PyTuple_SetItem(tuple, 2, PyFloat_FromDouble((double)curr_time)); 
    scream("[DEBUG] Put end time in tuple"); 
    PyTuple_SetItem(tuple, 3, PyLong_FromLong(n_errdisc));
    scream("[DEBUG] Put discerr in tuple"); 
    PyTuple_SetItem(tuple, 4, PyLong_FromLong(n_errdec));

    //make the current time -> last_read time
    last_read = curr_time;

    scream("[DEBUG] Assembled the tuple"); 
    // Add tuple to deque:
    Py_INCREF(deque);
    PyObject *result = PyObject_CallMethod(deque, "append", "(O)", tuple);
    if(result==NULL){
        scream("[DEBUG] CAN NOT access deque!");
        Py_DECREF(result);
        PyGILState_Release(gstate);
        return NULL;
    }
    Py_DECREF(result);
    Py_DECREF(deque);
    scream("[DEBUG] Appended data to DEQUE"); 
    // adding recorded words to global word counter in fifo_readout class object
    PyObject *nw = PyObject_CallMethod(
            PyObject_GetAttrString(self,"_words_per_read"),"append","i", &n_words);
    Py_DECREF(nw);
    scream("[DEBUG] Added to _words_per_read"); 

    // add command to reset rx error counters //
    if(n_errdisc != 0 || n_errdec !=0){
        std::cout<<"Encountered "<<n_errdisc<<" discard errors and"
                 <<n_errdec<<" decode errors - Resetting error counters"<<std::endl;
        PyObject *reg = PyObject_CallMethod(self,"reset_rx",NULL);
        if(reg==NULL){
            scream("failed to reset RX!");
        }else{
            scream("---RX reset!---"); 
        }
        Py_DECREF(reg);
    }
    scream("[DEBUG] Passed RESET RX"); 
    // checking other fifo_readout attributes:
    //
//    if(PyObject_IsTrue(PyObject_GetAttrString(self, "_calculate")){
//        PyObject *calc = PyObject_GetAttrString(self,"_calculate");
//        PyObject *result = PyObject_GetAttrString(self,"_result");
//        Py_INCREF(calc);
//        PyObject *res = PyObject_CallMethod(calc, "clear");
//        if(res==NULL){
//            scream("could not clear self._calculate ...");
//        }
//
//        PyObject *put = PyObject_CallMethod(result,"put","i",)
//
//
//        Py_DECREF(res);
//        Py_DECREF(calc);
//    }

    Py_DECREF(tuple); 
    Py_DECREF(chip); 

    std::cout<<"timings:"
             <<"last_read:"<<last_read
             <<"\tinterval:"<<interval
             <<"\tcurr_time:"<<curr_time
             <<"\tt_end:"<<t_end
             //<<"\tt-diff:"<<t_diff
             //<<"\tstart_read:"<<duration_time(start_read)
             //<<"\tstop_read:"<<duration_time(stop_read)
             <<"\n"<<std::flush;
       
 
    //check if -> self._calculate_result.is_set()
    //if yes self._calculate_result.clear()

    auto glob_iter_end = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - iter_start);

    float t_iteration = std::chrono::duration_cast<std::chrono::milliseconds>(
            (iter_start + glob_iter_end).time_since_epoch()).count();

    iter_times.push_back(t_iteration);    

    data.clear();

    //PyObject_Print(workercnt,stdout,0);
    std::cout<<"workercnt="<<PyLong_AsLong(workercnt)<<std::flush;
    scream("");
    //PyObject_Print(workerCNT,stdout,0);
    //std::cout<<"workerCNT="<<PyLong_AsLong(workerCNT)<<std::flush;

    //scream("");

    scream("[DEBUG] iteration end"); 

  };// end the while loop

  Py_DECREF(stopreadout);
  Py_DECREF(forcestop);
  //Py_DECREF(stopflag);

  //std::cout<<"N tries = "<<debugs.tries.size()<<"!\n"<<std::flush;
  //std::cout<<"N times = "<<debugs.times.size()<<"!\n"<<std::flush;

  //check if self.callback 
  //append "None" to deque

  std::cout<<"N gloabl tries = "<<glob_tries<<"!\n"<<std::flush;
  std::cout<<"AVG(global loop times) = "<<getVectorFloatAvg(iter_times)<<"!\n"<<std::flush; 

  scream("[DEBUG] - while loop done"); 
  ///// finita la comedia! ////// 
  PyGILState_Release(gstate);   
  return Py_None;

};


///// the one below is just a reminder on checking PyObjects /////////////
PyObject* querryFifoSize(PyObject *fChipFifoSize){

   // wana read something like
   // self.chip['FIFO'].['FIFO_SIZE'] 
   //  or 
   // self.chip['FIFO'].get_data() 

        //========= testing shit here ===============
        PyGILState_STATE gstate = PyGILState_Ensure(); //puts mutex lock on the bytecode of the object
        PyObject *result;

        if(fChipFifoSize != NULL){
            PyObject_Print(fChipFifoSize,stdout,0);
            bool huya = PyCallable_Check(fChipFifoSize);
            std::cout<<"[["<<huya<<"]]"<<std::flush; //
            if(huya){
                Py_INCREF(fChipFifoSize);
                result = PyObject_CallObject(fChipFifoSize,NULL);
                Py_DECREF(fChipFifoSize);
            }else{
                result = 0;
                printf("\nResult is NULL\n");
            }
        }

        Py_DECREF(result);
        Py_DECREF(fChipFifoSize);
        ////// these last two should be uncommented at all times (temporarily...) ////////////
        PyGILState_Release(gstate); //releases mutex lock

        return result;
};


//struct PyCinterface{
//
//    PyCinterface();
//    ~PyCinterface(){};
//
//    private:   
//
//    public:
//
//}

// end of class

}


//////////////////////////////////////////////////

