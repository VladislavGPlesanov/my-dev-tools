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

extern "C" {

       //////// some utils here /////////

    chronoTime tick(){
        return std::chrono::high_resolution_clock::now();
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
        std::cout<<"\nRECDATA:{"<<vect.data()<<"}\n"<<std::flush;
        for(int i = 0; i < vect.size();i++){
            std::cout<<vect.at(i)<<"|"<<std::flush;
        }
    }

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

//////////////////////////////////////////////////////////////////////////
///---------- working numpy C-API functions --------------------------
//////////////////////////////////////////////////////////////////////////

PyObject* readFifoStatus(PyObject *self, const char* option){       

// test case when passing self in fifo_readout 
// object should be: <tpx3.fifo_readout.FifoReadout>

    PyGILState_STATE gstate = PyGILState_Ensure(); 
    PyObject *res;
    /*returns status and counters for FIFO for following options:
        get_rx_sync_status,
        get_rx_en_status,
        get_rx_fifo_discard_count,
        get_rx_decode_error_count
    */
    if(self != NULL){
        Py_INCREF(self);
        res = PyObject_CallMethod(self,option,NULL);// not null
        Py_DECREF(self);
    }   

    Py_ssize_t llen = PyList_Size(res);
    PyObject *list = PyList_New(0);
    for(Py_ssize_t ith = 0; ith < llen; ++ith ){
        PyList_Append(list, PyList_GetItem(res,ith));
    }
    Py_DECREF(res);
    PyGILState_Release(gstate); 
    return list;
    // if one returns "res" object directly -> segfault!                     
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

std::vector<uint32_t> localQuerryFifo(PyObject *chip){

    PyGILState_STATE gstate = PyGILState_Ensure(); 
        
    PyArrayObject *result; 
    //scream("I'm in \"localQuerry\"");
    if(chip != NULL){
        PyObject *meth = PyObject_GetAttrString(chip, "get_data");
        // increase and then decreaase reference to *chip
        Py_INCREF(chip); 
        result = reinterpret_cast<PyArrayObject*>(PyObject_CallMethod(chip,"get_data",NULL));
        Py_DECREF(chip);
 
    }else{
        scream("chip is null\nObject was not passed correctly");
        result = 0;
    }

    // TODO: optimize following 4 lines into util function/template
    uint32_t* result_data = (uint32_t*)PyArray_DATA(result);
    npy_intp* arr_dim = PyArray_DIMS(result);
    npy_intp* arr_size = &arr_dim[0];
    std::vector<uint32_t> result_vector(result_data, result_data + *arr_size);

    //if(result_vector.size() > 0){ 
    //    std::cout<<std::boolalpha<<is_contiguous(result_vector.begin(),result_vector.end());
    //}

    //scream("localQuerryFifo::recorded:"+std::to_string(result_vector.size()));
    Py_DECREF(result);
    //releases mutex lock
    PyGILState_Release(gstate); 

    return result_vector;

};

PyObject* getFifoData(PyObject *chip, float t_interval){

    // put mutex lock on the bytecode of the object in interpretr
    // to have thread-safe execution 
    auto start_time = std::chrono::high_resolution_clock::now();

    PyGILState_STATE gstate = PyGILState_Ensure(); 
    Py_Initialize();
    import_array();

    std::vector<uint32_t> temp;
    std::vector<uint32_t> fifo_data;
    long int cnt = 0; 

    float total_time = 0.0;

    //auto start_time = std::chrono::high_resolution_clock::now();

    while(time_to_read(start_time) < t_interval){
        temp = localQuerryFifo(chip);
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
    //auto end_time = std::chrono::high_resolution_clock::now();

    float tdiff = timeDiff(start_time,end_time);
    
    std::cout<<"t0="<<total_time<<"\n"<<std::flush;
    //std::cout<<"tdiff="<<tdiff<<"\n"<<std::flush;

    //total_time = total_time + tdiff;
    updateTimeStamp(total_time, start_time, end_time);
    std::cout<<"newt0="<<total_time<<"\n"<<std::flush;

    /////////////////////////////////////////////////////////////
    const std::size_t dsize = fifo_data.size();
    const npy_intp test_dsize[] = {(npy_intp)(dsize)};

    std::cout<<"got ["<<dsize<<"] words\n"<<std::flush;
    //std::cout<<"test:["<<test_dsize<<"]\n"<<std::flush;

    PyObject* array = PyArray_SimpleNew(1,test_dsize,NPY_UINT32);
    if(array==NULL){
        scream("ARRAY IS NULL!");
    }

    uint32_t* thisData = reinterpret_cast<uint32_t*>(
            PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)));

    thisData = &fifo_data[0];


    float tdiff_i = timeDiff(end_time,tick());
    
    std::cout<<"t0="<<total_time<<"\n"<<std::flush;
    //std::cout<<"tdiff="<<tdiff_i<<"\n"<<std::flush;
    
    //total_time = total_time + tdiff_i;
    updateTimeStamp(total_time, end_time, tick());
    std::cout<<"newt0="<<total_time<<"\n"<<std::flush;



    if(fifo_data.data()!=NULL){
        PyGILState_Release(gstate); 
        return array;

    }else{/*else block will stay as is*/
        fifo_data.clear();
        PyGILState_Release(gstate); 
        return Py_None;
    } 

    //release mutex lock
    Py_Finalize();
    PyGILState_Release(gstate); 

    return Py_None;

};


////////////////////////////////////////////////////////
/// WIP, to be tested below///
////////////////////////////////////////////////////////

//int testFifoSize(PyObject *self){
PyObject *testFifoSize(PyObject *self){
//void testFifoSize(PyObject *self){
 
    PyGILState_STATE gstate = PyGILState_Ensure(); 
    PyObject *chip = PyObject_GetAttrString(self,"chip");

    std::size_t nent = 1;
    const npy_intp size[] = {(npy_intp)(nent)};
    scream("KAKOGO");
    
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
                //std::cout<<"type(fsize)=long?:"<<PyLong_Check(fsize)<<"\n"<<std::flush; //YES!
                //std::cout<<"type(fsize)=long?:"<<PyLong_Check(fsize)<<"\n"<<std::flush; //YES!
                scream("NIHERA");
                ndatawords = PyLong_AsLong(fsize);
                temp = static_cast<uint32_t>(ndatawords);
                scream("NE");
                //Py_DECREF(fsize);
                std::cout<<"READING:{"<<ndatawords<<"}words in fifo\n"<<std::flush;
                //long ndatawords = PyLong_AsLong(fsize);
                ////smth = PyLong_FromLong(*ndatawords);
                //result = PyLong_FromLong(ndatawords);
                //PyObject_Print(result,stdout,0);
                if(ndatawords == -1 && PyErr_Occurred()){
                //if(result ==NULL){
                    scream("ERROR ArraySorter::testFifosize: chip->fifo->fsize->result is NULL!");
                }
                //Py_INCREF(thisSize);
                //thisSize = PyLong_FromLong(ndatawords);
                //Py_DECREF(self);
                //Py_DECREF(fifo);
                //Py_DECREF(chip);

                //PyGILState_Release(gstate); 
                //return ndatawords;
                //return result;
                //return PyLong_AsLong(fsize);// instant segfault

            }
            Py_DECREF(fsize);
        }
        Py_DECREF(fifo);
    }
    Py_DECREF(chip);
    Py_DECREF(self);
    scream("PROISHODIT");

    PyObject *result = PyArray_SimpleNew(1,size,NPY_UINT32);

    uint32_t* thisFifoSize = reinterpret_cast<uint32_t*>(
                PyArray_DATA(reinterpret_cast<PyArrayObject*>(result)));
   
    scream(",SUKA!");
    thisFifoSize = &temp; 

    PyGILState_Release(gstate); 
    //// might be working but not sure yet
    return result;

};

PyObject* testOutput(PyObject *self){

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

PyObject *ReaderLoop(PyObject *chip, float t_interval, int interupt){

    PyGILState_STATE gstate = PyGILState_Ensure(); 
    Py_Initialize();
    import_array();

    auto current_time = std::chrono::high_resolution_clock::now();

    std::vector<uint32_t> temp;
    std::vector<uint32_t> fifo_data;

    //get read_time()
    auto start_time = std::chrono::high_resolution_clock::now();

    float t_wait = 0.0;

    long int cnt = 0; 
    while(time_to_read(start_time) < t_interval){
        temp = localQuerryFifo(chip);
        if(temp.size()!=0){
            fifo_data.insert(fifo_data.end(),
                             temp.begin(),
                             temp.end());
        }
        temp.clear();
        cnt++;
    }

    // count words - fifo.size()

    // get timestamp

    // get error count
     
    auto end_time = std::chrono::high_resolution_clock::now();
    const std::size_t dsize = fifo_data.size();
    const npy_intp test_dsize[] = {(npy_intp)(dsize)};

    std::cout<<"got ["<<dsize<<"] words\n"<<std::flush;
    std::cout<<"test:["<<test_dsize<<"]\n"<<std::flush;

    PyObject* array = PyArray_SimpleNew(1,test_dsize,NPY_UINT32);
    if(array==NULL){
        scream("ARRAY IS NULL!");
    }

    uint32_t* thisData = reinterpret_cast<uint32_t*>(
            PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)));

    thisData = &fifo_data[0];
    ///////////////////////////////////
  
    //std::time_t tstamp = getTimeStamp(start_time, end_time);

    //////////////////////////////////

    if(fifo_data.data()!=NULL){
        PyGILState_Release(gstate); 
        return array;

    }else{/*else block will stay as is*/
        fifo_data.clear();
        PyGILState_Release(gstate); 
        return Py_None;
    } 

    //release mutex lock
    Py_Finalize();
    PyGILState_Release(gstate); 

    return Py_None;




};


//curr_time = self.get_float_time() [ ] questionable...
//self.get_rx_fifo_discard_count()  [V]
//self.get_rx_sync_status()         [V]
//self.get_rx_decode_error_count()  [V]
//
//self.chip.get_modules('tpx3_rx')  [X] -> DATA_DELAY,SAMPLING_EDGE,INVERT,RESET,READY
//
//  thisngs above can be accessed via self.chip[channel].<parname>
//  channel == "RX0"--"RX7"
//
//self.enable_rx();              []
//self.reset_rx();               []
//self.reset_sram_fifo();        []
//self.rx_error_reset();         []

//PyObject *getObject(PyObject *self, const char* name){
//    
//    PyGILState_STATE gstate = PyGILState_Ensure(); 
//    PyObject *chip, *fifo, *fifo_size;
//
//    if(name=="chip"){
//        chip = PyObject_GetAttrString(self,"chip");
//        if(chip != NULL){
//            return chip;
//        }
//
//
//
//
//
//                Py_DECREF(fsize);
//                Py_DECREF(fifo);
//                Py_DECREF(chip);
//                Py_DECREF(self);
//
//                PyGILState_Release(gstate); 
//                //return PyLong_FromLong(ndatawords);
//                //return result;
//                return ndatawords;
//                //return fsize; 
//
//    }else{
//        scream("chip is NULL")
//        return NULL;
//    }
//    PyGILState_Release(gstate); 
//
//};

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

