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

typedef std::chrono::milliseconds chrono_ms;

std::chrono::_V2::system_clock::time_point l_global_start_time;

std::string ansi_red = "\u001b[31m";
std::string ansi_green = "\u001b[32m";
std::string ansi_orange = "\u001b[33m";
std::string ansi_reset = "\u001b[0m";
///////////////////////////////////////////////////////

signed long int global_words_recorded = 0; 

//////////////////////////////////////////////////////

extern "C" {

    //////// some utils here /////////
    //struct {
    //    std::vector<long int> tries;
    //    std::vector<float> times;
    //} debugs;

    ///////////////////////////////////////////////////////////

    struct FIFO_DATA {
    
        uint32_t* data;
        std::size_t datasize;

        // constructor to init struct
        FIFO_DATA(uint32_t* data_ptr, std::size_t size) : 
            data(new uint32_t[size]), datasize(size) {
            std::copy(data_ptr, data_ptr+size, data);
        };
        
        //Destructor to release memory:
        ~FIFO_DATA() {
            delete[] data;
        };

        //copy constructor to do deep copy
        FIFO_DATA(const FIFO_DATA& other) :
            data(new uint32_t[other.datasize]), datasize(other.datasize){
            std::copy(other.data, other.data + datasize, data);
        }

        //Assignment operator to do deep copy
        FIFO_DATA& operator=(const FIFO_DATA& other) {
            if(this != &other){
               //Release old memory
               delete[] data;
               //Allocate new memory and copy
               data = new uint32_t[other.datasize];
               datasize = other.datasize;
               std::copy(other.data, other.data + datasize, data);
            }
            return *this;
        }

    };

    struct tpx3data { //testing structure output to py

        uint32_t* data;
        std::size_t dsize;
        //const double tstart;
        //const double tend;
        //long* discerr[8];
        //long* decerr[8];

       // const uint32_t* data;
       // std::size_t dsize;
       // //const double tstart;
       // //const double tend;
       // const long* discerr[8];
       // const long* decerr[8];

    };

    chronoTime tick(){
        return std::chrono::high_resolution_clock::now();
    }

    inline void printLine(std::string ltype, int nchar){
    
        std::string line="";
        for(int i=0; i<nchar; i++){
            line = line+ltype;
        }
        line = line+"\n";
        std::cout<<line<<std::flush;

    }

    float getVectorFloatAvg(std::vector<float> vect){
        
        int vsize = vect.size();
        float sum = std::accumulate(vect.begin(),vect.end(),0.0);
        float result = sum/(float)(vsize);
        return result;

    };


    float getVectorLongIntAvg(std::vector<long int> vect){
        
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


    void printPyArray(PyObject *pyarray){

        PyArrayObject *arr = reinterpret_cast<PyArrayObject*>(pyarray);

        uint32_t* array_data = reinterpret_cast<uint32_t*>(PyArray_DATA(arr));

        npy_intp array_size = PyArray_SIZE(arr);

        for(npy_intp i = 0; i < array_size; ++i) {
           if(i!=0 && i%8==0){
                std::cout<<"|"<<array_data[i]<<"\n"<<std::flush;
           }else{
                std::cout<<"|"<<array_data[i]<<std::flush;
           }
           if(i==64){
                break;
           }
        }

    };
    
    void printReversePyArray(PyObject *pyarray){

        PyArrayObject *arr = reinterpret_cast<PyArrayObject*>(pyarray);
        uint32_t* array_data = reinterpret_cast<uint32_t*>(PyArray_DATA(arr));
        npy_intp array_size = PyArray_SIZE(arr);

        int stopNum = 64;
        int nstart = (int)(array_size) - stopNum;
        for(npy_intp i = nstart; i < array_size; ++i) {
           if(i!=0 && i%8==0){
                std::cout<<"|"<<array_data[i]<<"\n"<<std::flush;
           }else{
                std::cout<<"|"<<array_data[i]<<std::flush;
           }
           //if((int)(array_size) - i == stopNum){
           //     break;
           //}
        }

    };

    void compareVectorAndPyArray(uint32_t* array_data,
                                 const npy_intp array_size,
                                 const std::vector<uint32_t> &vect){

        //uint32_t* array_data = reinterpret_cast<uint32_t*>(PyArray_DATA(pyarray));
        //npy_intp array_size = PyArray_SIZE(pyarray);

        // Check if the sizes of the vector and array are the same
        std::cout<<"SIZE CHECK-> "<<std::flush;
        if (vect.size() != array_size) {
            std::cout << "Size mismatch: Vector size = " 
                      << vect.size() << ", Array size = " << array_size << std::flush;
        }else{
            scream("SIZES ARE EQUAL!");
        }

        int offset, n_headers_vect, n_headers_array;

        scream("E TY,");
        npy_intp this_index = array_data[0]==array_data[2] && array_data[1]!=array_data[3] ? 1:2;

        scream("SUKA!");
        std::cout<<"EBAL-"<<vect[0]<<"\n"<<std::flush;
        std::cout<<"EBAL-"<<vect[1]<<"\n"<<std::flush;
        std::cout<<"EBAL-"<<vect[2]<<"\n"<<std::flush;
        std::cout<<"EBAL-"<<vect[3]<<"\n"<<std::flush;
        int headerAt = vect[0]==vect[2] && vect[1] != vect[3] ? 0:1;
        scream("got header positions");
        uint32_t HEADER = vect[headerAt];

        for(auto &elem: vect){
            if(elem==vect[headerAt]){
                n_headers_vect++;
            }
        }
        scream("got number of vector headers");
        for(npy_intp i=0; i<array_size; ++i){
            if(array_data[this_index]==vect[i]){
                if(i!=0){
                   offset=i-1;
                }else{
                   offset=i;
                }
                break;
            }
        }

        std::stringstream strstr;
        strstr<<"Found offset in data = --{"<<offset<<"}-- words\n";
        scream(strstr.str());

        for(npy_intp i=0; i<array_size; ++i){
            if(array_data[this_index-1]==vect[headerAt]){
                n_headers_array++;
            }
        }
        std::cout<<"Found header "<<vect[headerAt]
                 <<" at "<<headerAt<<"("<<n_headers_vect
                 <<")/("<<n_headers_array<<")"<<this_index<<"\n"<<std::flush;

        long int cnt = 0;
        scream("ELEMENT COMPARISON-> ");
        // Check if the elements of the vector and array are the same
        scream(" [index] | vector[index+offset] | array[index]\n----------------");
        for (npy_intp i = 0; i < array_size; ++i) {
            if (vect[i+offset] != array_data[i] && i+offset<=vect.size()-1) {
                cnt++;
            }
            if(i%2==0 && i<17){
                std::cout<<"["<<i+offset<<"] "<<vect[i+offset]<<"\t["
                         <<i<<"] "<<array_data[i]<<"\n"<<std::flush;
                scream("---------------------------------------------------");
                std::cout<<"["<<i+1+offset<<"] "<<vect[i+1+offset]<<"\t["
                         <<i+1<<"] "<<array_data[i+1]<<"\n"<<std::flush;
                scream("---------------------------------------------------");
            }

        }
        if(cnt!=0){
            std::cout<<"Found "<<cnt<<" ,mismatches for sets of ["
                     <<vect.size()<<" and "<<array_size
                     <<" ["<<((float)(cnt)/(float)(vect.size()))*100<<" \%]\n"<<std::flush;
        }

        scream("CONTENT CHECK-> ");
        std::vector<uint32_t> missing;
        for (const auto &elem: vect){
            bool found = false;
            for(npy_intp i=0; i< array_size; ++i){
                if(elem==array_data[i]){
                    found = true;
                    break;
                }
            }
            if(!found){
                missing.push_back(elem);
            }
        }
        if(missing.size()==0){
           scream("elements are same in both containers! Order might be different!");           
        }else{
           scream("INCONSISTENCY!");
           std::cout<<"MIssing:"<<missing.size()<<" words\n PyArray!!!n"<<std::flush;
        }
        missing.clear();

        offset = 0;
        n_headers_vect = 0;
        n_headers_array = 0;

    }

    void compare_vectors(std::vector<uint32_t> vect1, std::vector<uint32_t> vect2){

        if(vect1.size()!=vect2.size()){
            std::stringstream ss;
            ss<<"vector1 size="<<vect1.size()
              <<", vector2 size="<<vect2.size()<<"!";
            scream(ss.str());
        }else{
            scream("vectors are same size!");
        }

        bool sameContent = std::equal(vect1.begin(),vect1.end(),vect2.begin());
        if(sameContent){
            std::stringstream ss;
            ss<<"vector1 size="<<vect1.size()
              <<", vector1 size="<<vect2.size()<<"!";
            scream(ss.str());
        }else{
            scream("Vector content DIFFERS:");
            std::vector<int> positions;
            std::size_t minsize = std::min(vect1.size(),vect2.size());
            for(std::size_t i =0; i<minsize; i++){
                if(vect1[i]!=vect2[i]){
                    positions.push_back(i);
                }
            }
            std::cout<<"vector of indexes: SIZE="<<positions.size()
                    <<", element[0]="<<positions.at(0)
                    <<", element[last]="<<positions.at(positions.size()-1)<<"\n"<<std::flush;
        
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
              std::cout<<"\n"<<std::flush;
              //scream("\n");
            }
            if(i==63){
                break;
            }
        }
        std::cout<<"and "<<vect.size()-64<<" more\n"<<std::flush;
    }

    void printReverseVectUint(std::vector<uint32_t> vect){    

        //for(int i = vect.size()-1; i > 0; i--){
        for(int i = vect.size()-64; i<vect.size(); i++){
            std::cout<<vect.at(i)<<"|"<<std::flush;
            if(i !=vect.size()-64 && i % 8 == 0){
              std::cout<<"\n"<<std::flush;
            }
            if(i==vect.size()-1){
                break;
            }
        }
        std::cout<<"and "<<vect.size()-64<<" more\n"<<std::flush;
    }

    void checkIfObjNull(PyObject *obj, std::string objname){
        scream("");
        PyObject_Print(obj,stdout,0);
        if(obj==NULL){
            std::cout<<ansi_red<<"\nobject <"<<objname
                     <<"> is NULL!\n\n"<<ansi_reset<<std::flush;
        }
    }

    void printObjType(PyObject *obj){

        PyTypeObject *type = obj->ob_type;
        const char* type_name = type->tp_name;
        std::cout<<"[TYPE CHECK] ("<<type_name<<")\n"<<std::flush;

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

    //void copyDataArray(uint32_t* input, uint32_t* output, std::size_t arrSize){
    uint32_t *copyDataArray(uint32_t* input, std::size_t arrSize){
   
        uint32_t *output; 
        std::memcpy(output, input, arrSize);
        //for(std::size_t i=0; i<arrSize; ++i){
        //    output[i] = input[i];
        //};
        return output;
    
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

    //scream("[DEBUG] getting Errors:");
    long int n_errors = 0;

    Py_ssize_t listlen = PyObject_Length(list);
    for(Py_ssize_t it = 0; it < listlen; ++it){
        PyObject *item = PyList_GetItem(list, it);
        if(item!=NULL){
            Py_INCREF(item);
            n_errors+=PyLong_AsLong(item);
            // std::cout<<"RX"<<it<<"-"<<PyObject_Print(item,stdout,0)<<std::flush;
            Py_DECREF(item);
        }else{
            scream("can not access item in:");
            PyObject_Print(list,stdout,0);
            Py_DECREF(item);
        }
        Py_XDECREF(item);
        item=NULL;
    };
    return n_errors;
};
///////// assemblying an amalgamation of getNError & readFifoStatus

long int getNumError(PyObject *self, const char* option){


    PyGILState_STATE gstate = PyGILState_Ensure(); 
    PyObject *statuslist;

    scream("[DEBUG] getting Errors:");
    long int n_errors = 0;
    if(self!=NULL){
        statuslist = PyObject_CallMethod(self,option,NULL);
        if(statuslist==NULL){
            std::stringstream ss;
            ss<<"[ERROR] Could not obtain RX status for option"<<option;
            scream(ss.str());   
            Py_DECREF(self);
            PyGILState_Release(gstate);
            return 0;
        }
    }else{
        scream("[ERROR] object *self is NULL");
        Py_DECREF(self);
        PyGILState_Release(gstate);
        return 0;
    }
    /////////////////////////////////////////////////////////////// 

    Py_ssize_t listlen = PyObject_Length(statuslist);
    for(Py_ssize_t it = 0; it < listlen; ++it){
        PyObject *item = PyList_GetItem(statuslist, it);
        if(item!=NULL){
            Py_INCREF(item);
            n_errors+=PyLong_AsLong(item);
            // std::cout<<"RX"<<it<<"-"<<PyObject_Print(item,stdout,0)<<std::flush;
            Py_DECREF(item);
        }else{
            scream("can not access item in:");
            PyObject_Print(statuslist,stdout,0);
            Py_DECREF(item);
        }
        Py_XDECREF(item);
        item=NULL;
    };
    
    scream("[DEBUG] finished");
    Py_DECREF(statuslist);
    Py_DECREF(self);
    //statuslist=NULL;

    PyGILState_Release(gstate);

    return n_errors;
};




int printobject(PyObject *obj){return PyObject_Print(obj,stdout,0);};

///////////////////////////////////////////////

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

    //std::copy(data.begin(),data.end(),thisData); 
    // to try:
    memcpy(thisData, &data[0], datasize*sizeof(uint32_t)); // this is actually faster..
                                                           // but not sure why reads less...
    return pyarray;

};

std::vector<uint32_t> fillVectorFromPyData(PyArrayObject *pyarray){

    uint32_t* result_data = (uint32_t*)PyArray_DATA(pyarray);
    npy_intp* arr_dim = PyArray_DIMS(pyarray);
    npy_intp* arr_size = &arr_dim[0];
    std::vector<uint32_t> result_vector(result_data, result_data + *arr_size);
    Py_DECREF(pyarray); // Have to release pyarray reference  here
                        // - otherwise 2Gb leak in 5 seconds! 

    return result_vector;

};

PyObject* fillPyList(std::vector<long int> &stats){// maybe works

    Py_ssize_t statlen = (Py_ssize_t)(stats.size());
    PyObject * pylist = PyList_New(0);
    for(Py_ssize_t ith = 0; ith < statlen; ++ith ){
        PyObject *pyval = PyLong_FromLong(stats[ith]);
        Py_INCREF(pylist);
        PyList_Append(pylist, pyval);
        Py_DECREF(pylist);
        Py_DECREF(pyval);
    }

    return pylist;

};

//////////////////////////////////////////////////////////////////////////
///---------- working numpy C-API functions --------------------------
//////////////////////////////////////////////////////////////////////////

//TODO: make a local version
std::vector<long> readFifoStatusVector(PyObject *self, const char* option){       

    PyGILState_STATE gstate = PyGILState_Ensure(); 
    PyObject *res;
    /*returns status and counters for FIFO for following options:
        get_rx_sync_status,
        get_rx_en_status,
        get_rx_fifo_discard_count,
        get_rx_decode_error_count
    */
    std::vector<long> rxstats;
    if(self != NULL){
        Py_INCREF(self);
        //scream("readFifoStatus::self not null");
        res = PyObject_CallMethod(self,option,NULL);// not null
    }else{
        scream("readFifoStatus::self object is NULL!");
        //return rxstats.fill(0);
        return {0,0,0,0,0,0,0,0};
    }
   
    if(PyObject_Size(res)<=0){
        std::cout<<"result for status {"<<option<<"} is <=0!\n"<<std::flush;
        return {0,0,0,0,0,0,0,0};
        //return rxstats.fill(0);
    }
    
    Py_ssize_t llen = PyList_Size(res);
    for(Py_ssize_t ith = 0; ith < llen; ++ith ){
        //Py_INCREF(res);
        //rxstats.push_back(PyLong_AsLong(PyList_GetItem(res,ith)));
        PyObject *num = PyList_GetItem(res,ith);
        rxstats.push_back(PyLong_AsLong(num));
        std::cout<<"type check:"<<PyLong_Check(num)<<"\t"<<std::flush;
        PyObject_Print(num,stdout,0);
        scream("");
        Py_DECREF(num);
        Py_DECREF(res);
    }
    
    std::cout<<"local_readFifoStatus vector [size="<<rxstats.size()<<"]\n"<<std::flush;
    int nth =0;
    for(auto &stat: rxstats){
        std::cout<<"rxstats["<<nth<<"]="<<stat<<"\n"<<std::flush;
        nth++;
    }

    Py_DECREF(res);
    Py_DECREF(self);
    //res=NULL; //instead of delete statement
    PyGILState_Release(gstate);  //REENABLE THIS LATER!
    return rxstats;
    // if one returns "res" object directly -> segfault!                     
};
/////////////////////////////////////////////////////

PyObject* readFifoStatus(PyObject *self, const char* option){       

// test case when passing self in fifo_readout 
// object should be: <tpx3.fifo_readout.FifoReadout>

    PyGILState_STATE gstate = PyGILState_Ensure(); 
    std::cout<<"[DEBUG] Checking:"<<option<<"\n"<<std::flush;
    PyObject *status;

    /*returns status and counters for FIFO for following options:
        get_rx_sync_status,
        get_rx_en_status,
        get_rx_fifo_discard_count,
        get_rx_decode_error_count
    */
    if(self != NULL){
        Py_INCREF(self);
        status = PyObject_CallMethod(self,option,NULL);// not null
        if(status==NULL){
            std::stringstream ss;
            ss<<"[ERROR] Could not obtain RX status for option"<<option;
            scream(ss.str());
            return emptyList();
        }
    }else{
        scream("readFifoStatus::self object is NULL!");
        return emptyList();
    }

    Py_DECREF(self);
    PyGILState_Release(gstate);

    return status;
};

///////////// works, but leaks memory... ////////////////
long int readErrorCounters(PyObject *rxchannels, const char* option){       

    // PyObject *rxchannels has to be <tpx3.tpx3_rx.tpx3_rx> object from python side
    //

    PyGILState_STATE gstate = PyGILState_Ensure();
    long int n_err = 0;
    std::string this_option;
    
    scream("[DEBUG] readErrorCounters-> IN");
    if(option==tpx3::CNTR_DISCARD_ERR){
        this_option = "DECODER_ERROR_COUNTER";
    }
    if(option==tpx3::CNTR_DECODING_ERR){
        this_option = "LOST_DATA_COUNTER";
    }
    if(option !=tpx3::CNTR_DECODING_ERR && option != tpx3::CNTR_DISCARD_ERR){
        scream("Invalid option, use DECODING or DISCARD string attributes");
        return -1;
    }
    
    if(rxchannels!=NULL){
       
        Py_ssize_t nrx = PyObject_Size(rxchannels);
        for(Py_ssize_t i = 0; i < nrx; ++i){
            PyObject *ithRX = PyList_GetItem(rxchannels, i);
            Py_INCREF(ithRX);
            PyObject *errcnt = PyObject_GetItem(
                    ithRX, PyUnicode_FromString(this_option.c_str()));
            n_err += PyLong_AsLong(errcnt);
            //n_err+=cntr;
            //std::cout<<"N_Err="<<n_err<<"\n"<<std::flush;
            Py_DECREF(errcnt);
            Py_DECREF(ithRX);
            errcnt=NULL;
            ithRX=NULL;
        }   

    }else{
        scream("[ERROR] can not access TPX3_RX channels!\n");
        n_err = -1;
    }
    scream("[DEBUG] readErrorCounters-> past loop");
    //Py_XDECREF(rxchannels);
    PyGILState_Release(gstate);

    return n_err;

}

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
        return tpx3::EMPTY_RX_STATUS_VECTOR;
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

void checkIsRunning(PyObject *self){

    PyObject *isRunning = PyObject_GetAttrString(self, "_is_running");
    std::cout<<"\n\t[self._is_runnning]="<<PyObject_IsTrue(isRunning)<<"\n"<<std::flush;
    Py_DECREF(isRunning);
    isRunning=NULL;

}

bool checkSelfCalculate(PyObject *self){

    PyObject *calc = PyObject_GetAttrString(self, "_calculate");
    bool status = false;

    if(calc!=NULL){
        PyObject *is_set = PyObject_GetAttrString(calc,"is_set");
        if(is_set!=NULL){
            status = PyObject_IsTrue(is_set);
        }else{
            scream("Could not get attr=\"_calculate.is_set()\"");
        }
        Py_DECREF(is_set); 
    }else{
        scream("Could not get attr=\"_calculate\"");
    }    

    Py_DECREF(calc);
    return status;

}


void clearSelfCalculate(PyObject *self){
    
    PyObject *calc = PyObject_GetAttrString(self, "_calculate");
    bool status = false;

    if(calc!=NULL){
        PyObject *clear = PyObject_CallMethod(calc,"clear",NULL); 
        if(clear==NULL){
            scream("Could not call [\"self._calculate.clear()\"]");
        }
        Py_DECREF(clear); 
    }else{
        scream("Could not get attr=\"_calculate\"");
    }    

    Py_DECREF(calc);

}

void setSelfRecordCount(PyObject *self){

    PyGILState_STATE gstate = PyGILState_Ensure();
      
    PyObject *total_words = PyLong_FromLong(global_words_recorded);
    
    PyObject_SetAttrString(self,"_record_count", total_words);
    Py_DECREF(total_words);

    PyObject *record_count = PyObject_GetAttrString(self,"_record_count"); // this one's [int] 
    //std::cout<<PyObject_Print(record_count,stdout,0)<<"\n"<<std::flush;
    Py_DECREF(record_count);

    PyGILState_Release(gstate);

}


void checkCallback(PyObject *self){
    // based on the if check from python side "if(self.callback)"
    // which essentially evaluates to "True" as long as the callback object exists
    //

    PyObject *isCallback = PyObject_GetAttrString(self, "callback");
    if(isCallback!=NULL){
      scream("[DEBUG] self.callback (or scan_base->handle_data func object) [EXISTS]");
    }else{
      scream("[DEBUG] self.callback (or scan_base->handle_data func object) is NULL!");
    }
    //PyObject_Print(isCallback, stdout, 0);
    Py_DECREF(isCallback);
    isCallback=NULL;

}

int readoutIsRunning(PyObject *self){

    PyObject *isRunning = PyObject_GetAttrString(self, "_is_running");
    if(PyObject_IsTrue(isRunning)){
        Py_DECREF(isRunning);
        return 1;
    }else{
        Py_DECREF(isRunning);
        return 0;
    }
    Py_XDECREF(isRunning);
    return 0;
    //return PyObject_IsTrue(isRunning);

}

float getNoDataTimeout(PyObject *self){
  
    //Py_INCREF(self);
    PyObject *nd_timeout = PyObject_GetAttrString(self, "no_data_timeout");
    if(nd_timeout==NULL){
       scream("could not find no_data_timeout!");
       Py_XDECREF(nd_timeout);
       return 0.0; 
    }  
    if(PyFloat_Check(nd_timeout)){
        //scream("getNoDataTimeout::Object is PyFloat, returning corresponding number");
        float timeout = PyFloat_AsDouble(nd_timeout);
        Py_DECREF(nd_timeout);
        return timeout;
        //return (float)(PyFloat_AsDouble(nd_timeout));
    }
    if(!(nd_timeout==nd_timeout)){
        //scream("getNoDataTimeout::Object is Py_None, returning 0.0");
        Py_DECREF(nd_timeout);
        return 0.0;
    }
    Py_XDECREF(nd_timeout);
    nd_timeout=NULL;
    //Py_DECREF(self);

    return 0.0;
}


bool flagStopReadout(PyObject *self){

    scream("INSIDE [flagStopReadout]");
    PyObject *stop_readout = PyObject_GetAttrString(self, "stop_readout");
    PyObject_Print(stop_readout,stdout,0);
    std::cout<<"INSIDE, check with PyObjecIsTrue="<<PyObject_IsTrue(stop_readout)<<"\n"<<std::flush;
    scream("");
    if(stop_readout!=NULL){
        scream("[DEBUG] self.stop_readout EXISTS");
        PyObject *is_set = PyObject_GetAttrString(stop_readout, "is_set");
        PyObject_Print(is_set,stdout,0);
        scream("");
        if(is_set!=NULL){
            std::cout<<ansi_red<<"\n\t[DEBUG]->[self.stop_readout.is_set()]="
                     <<PyObject_IsTrue(is_set)<<ansi_reset<<"\n"<<std::flush;
            if(PyObject_IsTrue(is_set)){
                scream("is_set=[TRUE]");  
                return true;
            }else{
                scream("is_set=[FALSE]");  
                return false;
            }
        }else{
            scream("[DEBUG] self.stop_readout.is_set IS NULL");
            return false;
        } 
        Py_DECREF(is_set);
    }else{
        scream("[DEBUG] self.stop_readout IS NULL");
        return false;
    }
    Py_DECREF(stop_readout);
    return false;

}

void checkStopReadout(PyObject *self){

    PyObject *stop_readout = PyObject_GetAttrString(self, "stop_readout");
    PyObject *is_set = PyObject_GetAttrString(stop_readout, "is_set");
    std::cout<<"\n\t[self.stop_readout.is_set()]="<<PyObject_IsTrue(is_set)<<"\n"<<std::flush;
    Py_DECREF(is_set);
    Py_DECREF(stop_readout);

}

void checkIfForceStop(PyObject *self, float time_wait){

    std::cout<<"[DEBUG] \"self.force_stop\" -> Should wait for:"<<time_wait<<"\n"<<std::flush;

    float t_wait = time_wait >= 0.0 ? time_wait : 0.0;

    PyObject *fstop = PyObject_GetAttrString(self,"force_stop");
    // testing force stop Event()
    scream("[TEST] self.force_stop() in action!");
    PyObject *check = PyObject_CallMethod(fstop,"wait","f",t_wait);
    
    Py_DECREF(check);
    Py_DECREF(fstop);

}

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

std::vector<uint32_t> localQuerryFifo(PyObject *chipFifo){

    PyGILState_STATE gstate = PyGILState_Ensure(); 
        
    PyArrayObject *result; 

    if(chipFifo != NULL){
        // increase and then decrease reference to *chipFifo
        // working version below
        Py_INCREF(chipFifo); 
        result = reinterpret_cast<PyArrayObject*>(PyObject_CallMethod(chipFifo,"get_data",NULL));
        Py_DECREF(chipFifo); 
    }else{
        scream("localQuerryFifo:[Error] chipFifo is null\nOR Object was not passed correctly");
        result = 0;
        // TODO: maybe put in empty PyArray_DATA..?
    }

    // test below
    Py_INCREF(result);
    std::vector<uint32_t> result_vector = fillVectorFromPyData(result);// technically works
    Py_DECREF(result);
    result = NULL;
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
        scream("[ERROR] ArraySorter::testFifosize: chipFifo->fsize is NULL!");
        return -1;      
    }else{
        Py_INCREF(fsize);
        ndatawords = PyLong_AsLong(fsize);
        Py_DECREF(fsize);
        std::cout<<"READING:{"<<ndatawords<<"}words in fifo\n"<<std::flush;
        if(ndatawords == -1 && PyErr_Occurred()){
            scream("ERROR ArraySorter::testFifosize: chipFifo->fsize->result is NULL!");
        }
    }
    //Py_DECREF(fsize);
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

    long int cnt = 0; 

    //auto start_time = std::chrono::high_resolution_clock::now();
    auto start_time = tick();
    Py_INCREF(chipFifo);
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
    //
    PyObject* array = fillPyArray(fifo_data); 

    if(fifo_data.data()!=NULL){
        fifo_data.clear();
        Py_DECREF(chipFifo);
        PyGILState_Release(gstate); 
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

///////////////////////////
std::vector<uint32_t> local_getFifoData(PyObject *chipFifo, float interval){

    PyGILState_STATE gstate = PyGILState_Ensure(); 

    std::vector<uint32_t> temp;
    std::vector<uint32_t> fifo_data;

    long int cnt = 0;  
 
    auto start_time = std::chrono::high_resolution_clock::now();
    while(time_to_read(start_time) < interval){
        Py_INCREF(chipFifo);
        std::vector<uint32_t> temp = localQuerryFifo(chipFifo);
        Py_DECREF(chipFifo);
        if(temp.size()>0){
            fifo_data.insert(fifo_data.end(),
                             temp.begin(),
                             temp.end());
        }
        temp.clear();
        cnt++;
    }
    temp.clear();
    ///////////////////////////////////////////////////////////////
    auto end_time = tick();

    float tdiff = timeDiff(start_time,end_time);
    
    std::cout<<"tried="<<cnt<<"times, redorded "<<fifo_data.size()<<" events\n"<<std::flush;
    std::cout<<"loop time="<<tdiff<<"[ms]\n"<<std::flush;
   
    if(fifo_data.size()>0){ 
      scream("first 64 words of data in \"local_getFifoData\":");
      printVectUint(fifo_data);
      scream("last 64 words of data in \"local_getFifoData\":");
      printReverseVectUint(fifo_data);
    }

    PyGILState_Release(gstate);  //REENABLE THIS LATER!
    return fifo_data;

};

////////////////////////////////////////////////////////
/// WIP, to be tested below///
////////////////////////////////////////////////////////

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
                return NULL;
            }
        }

        Py_DECREF(result);
        Py_DECREF(fChipFifoSize);
        ////// these last two should be uncommented at all times (temporarily...) ////////////
        PyGILState_Release(gstate); //releases mutex lock

        return result;
};

void checkFifoSize(PyObject *fifo){

    PyObject *fsize = PyObject_GetItem(fifo,PyUnicode_FromString("FIFO_SIZE"));
    long int fifoSize = PyLong_AsLong(fsize);
    std::cout<<"[DEBUG] FIFO_SIZE >> "<<fifoSize<<" words\n"<<std::flush;
    Py_DECREF(fsize);
    fsize=NULL;

};

void resetRXErrorCounters(PyObject *self){

    //std::cout<<"Resetting RX error counters!\n"<<std::flush;
    PyObject *reg = PyObject_CallMethod(self,"rx_error_reset",NULL);
    if(reg==NULL){
        scream("failed to reset RX!");
    }else{
        //scream("---RX reset!---"); 
    }
    Py_DECREF(reg);//-reg
    reg=NULL;

};

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

std::string checkReadStatus(PyObject *self){
    
    PyGILState_STATE gstate = PyGILState_Ensure();
    const char* status;
    PyObject *stopreadout = PyObject_GetAttrString(self,"stop_readout");
    if(stopreadout!=NULL){
        PyObject_Print(stopreadout,stdout,0);
        /////////////////////
        PyObject *isSet = PyObject_CallMethod(stopreadout,"is_set",NULL);
        //const char* temp = 
        std::cout<<"[DEBUG] CHEKCING self.stop_readout.isSet() = "
                 <<std::atoi(PyUnicode_AsUTF8(PyObject_Str(isSet)))<<"\n"<<std::flush;
        ////////////////////////
        status = PyUnicode_AsUTF8(PyObject_Str(stopreadout));
        //std::cout<<"[DEBUG] CHCEKING readout FLAG <pepelaugh> = "<<status<<"\n"<<std::flush;
    }else{
        scream("[ERROR]: stopreadout object is NULL!");
        return "HUYASE!";
    };
    Py_DECREF(stopreadout);
    stopreadout = NULL;
    PyGILState_Release(gstate);
    return status;

};

int checkSHUTTER(PyObject *self){

    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject *chip = PyObject_GetAttrString(self, "chip"); 
    int status;
    if(chip != NULL){
        PyObject *control = PyObject_GetItem(chip, PyUnicode_FromString("CONTROL"));
        if(control != NULL){
            PyObject *shutter = PyObject_GetItem(control, PyUnicode_FromString("SHUTTER"));
            status = std::atoi(PyUnicode_AsUTF8(PyObject_Str(shutter)));
            Py_DECREF(shutter);
        }else{
            scream("[ERROR]: SHUTTER IS NULL!");
            Py_XDECREF(control);
            Py_XDECREF(chip); //might be redundant
            return -1;
        }
        Py_DECREF(control);
        control = NULL;
    }else{
        scream("[ERROR]: CONTROL IS NULL!");
        Py_XDECREF(chip);
        return -1;
    }

    Py_DECREF(chip);
    PyGILState_Release(gstate); 

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
                Py_INCREF(chip);
                PyObject_SetItem(chip,REG,PyLong_FromLong(value));
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

////////////////////////////////////////
// could work with the struct but need fine tuning
//
FIFO_DATA dequeDataOutNoErr(PyObject *self, float interval){/*works ok.. but segfaults*/

  PyGILState_STATE gstate = PyGILState_Ensure(); 
  // 2 lines below should be used in "main" function only, others 
  // should use only PyGILState f-ncs.
  //Py_Initialize();
  //import_array();

  // initializin' objects for FIFO data
  PyObject *fifo;//, *fifoData; 
  // vector for local readout of fifo data
  std::vector<uint32_t> data;

  PyObject *chip = PyObject_GetAttrString(self,"chip");
  Py_INCREF(chip);
  fifo = PyObject_GetItem(chip,PyUnicode_FromString("FIFO"));
  Py_DECREF(chip);

  // recod fifo data locally
  Py_INCREF(fifo);
  data = local_getFifoData(fifo,interval); 
  Py_DECREF(fifo);
  
  if(data.size()==0){
    data.push_back(-999);
  }

   if(data.size()!=0){
    std::cout<<"[DEBUG] vector data @ ["<<data.data()<<"]\n"<<std::flush;
    std::cout<<"[DEBUG] vector[0] @ ["<<&data[0]<<"]\n"<<std::flush;
  }

  ///// finita la comedia! //////

  //Py_DECREF(chip);
  Py_DECREF(self); 
  PyGILState_Release(gstate);   
  //return {data.data(), data.size()};
  //return FIFO_DATA(const_cast<uint32_t*>(data.data()), data.size());
  if(data.empty()){
      return FIFO_DATA(nullptr, 0);
  }

  return FIFO_DATA(data.data(), data.size());
  //return {&data[0], data.size()};
  //return FIFO_DATA;

};

PyObject* dequeDataOut(PyObject *self, float interval){/*works good*/
//tpx3data dequeDataOut(PyObject *self, float interval){/*works good*/

  PyGILState_STATE gstate = PyGILState_Ensure(); 
  // 2 lines below should be used in "main" function only, others 
  // should use only PyGILState f-ncs.
  //Py_Initialize();
  import_array();

  // Allocating memory for output tuple
  //PyObject *tuple = PyTuple_New(3);
  PyObject *tuple = PyTuple_New(1);
  // release GIL if yes

  // initializin' objects for dat and rx error counters
  PyObject *chip, *fifo, *fifoData;
  // vector for local readout of fifo data
  std::vector<uint32_t> data;

  Py_INCREF(self); 
  chip = PyObject_GetAttrString(self,"chip");
  Py_INCREF(chip); //test
  fifo = PyObject_GetItem(chip,PyUnicode_FromString("FIFO"));
  Py_DECREF(chip); //test

  // recod fifo data locally
  Py_INCREF(fifo);
  data = local_getFifoData(fifo,interval); 
  Py_DECREF(fifo);
 
  //if(data.size()==0){
  //   data.push_back(-999);
  //} 
  if(data.size()!=0){
    std::cout<<"[DEBUG] vector data @ ["<<data.data()<<"]\n"<<std::flush;
  }

  // debug check of contents of recorded data
  //printVectUint(data);
  
  // this on puts vector.data() into numpy array  
  fifoData = fillPyArray(data);

  //scream("[dequeDataOut->fillPyArray] pyobject contents:\n"); 
  //uint32_t* array_data = reinterpret_cast<uint32_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(fifoData)));
  //npy_intp array_size = PyArray_SIZE(reinterpret_cast<PyArrayObject*>(fifoData));

  //if(data.size()>0){
  //  compareVectorAndPyArray(array_data, array_size, data);
  //}
  //PyObject_Print(fifoData,stdout,0);
  //scream("\n////////////////////////////////////////////////////////////\n");
 
  // reading fifo discard and decoding errors
  //Py_INCREF(self); 
  ////DISC_ERR = readFifoStatusVector(self,tpx3::CNTR_DISCARD_ERR);
  //fifo_discerr = readFifoStatus(self,tpx3::CNTR_DISCARD_ERR);
  //Py_DECREF(self); 
  //scream("SUKA");
  //Py_INCREF(self); 
  ////DEC_ERR = readFifoStatusVector(self,tpx3::CNTR_DECODING_ERR);
  //fifo_decerr = readFifoStatus(self,tpx3::CNTR_DECODING_ERR);
  //Py_DECREF(self); 

  scream("checking pyarray [first 64]:");
  Py_INCREF(fifoData);
  printPyArray(fifoData);
  Py_DECREF(fifoData);
  
  scream("checking pyarray [last 64]:");
  Py_INCREF(fifoData);
  printReversePyArray(fifoData);
  Py_DECREF(fifoData);

  scream("PIZDEC");
  // filling output tuple 
  // as (<fifo data>, <listof discard errors>, <list of decode errors>)

  Py_INCREF(fifoData);
  PyTuple_SetItem(tuple, 0, fifoData);
  Py_DECREF(fifoData);

  //data.clear();
  //Py_INCREF(fifo_discerr);
  //PyTuple_SetItem(tuple, 1, fifo_discerr);
  //Py_DECREF(fifo_discerr);

  //Py_INCREF(fifo_decerr);
  //PyTuple_SetItem(tuple, 2, fifo_decerr);
  //Py_DECREF(fifo_decerr);

  ///// finita la comedia! //////
  
  //Py_DECREF(self); 
  //Py_Finalize();
  PyGILState_Release(gstate);   
  return tuple;
  //return {data.data(), data.size(), DISC_ERR.data(), DEC_ERR.data()};
  //return {data.data(), data.size()};

};
//---------------------------------------------------------------------------------
//
//  Substitute for tpx3/fifo_readout/readout function
//
//---------------------------------------------------------------------------------
//PyObject* readoutToDeque(PyObject *self, PyObject *deque, float interval){
PyObject* readoutToDeque(PyObject *self, PyObject *deque, PyObject *RXLIST, float interval){

  PyGILState_STATE gstate = PyGILState_Ensure();
  //PyEval_InitThreads(); // releases lock instantly maybe try later
  // together with PyEval_ReleaseLock() at the end
  // 2 lines below should be used in "main" function only, others 
  // should use only PyGILState func-ns.

  //Py_Initialize(); // should be enabled, but works without it

  import_array();
 
  scream("[DEBUG] starting readout"); 

  if(l_global_start_time == std::chrono::_V2::system_clock::time_point()){
    l_global_start_time == tick();
  }
  // last chunk read time, current time, time to wait (not implemented yet)
  float last_read = 0.0, curr_time = 0.0, time_wait = 0.0;

  // for now, waiting only for a stop signal from user input
  // for some fuck-reasons PyObject_IsTrue evaluates "unset" state to logic 1...
  // thus logic in while loop is inverted wrt to python side
  //
  int glob_tries = 0;
  int prev_shutter = 0;
  float noDataTimeout = 0.0;
  std::vector<float> iter_times;

  // getting common global objects
  //
  PyObject *chip = PyObject_GetAttrString(self,"chip");
  PyObject *fifo = PyObject_GetItem(chip,PyUnicode_FromString("FIFO"));

  // checking user-defined timeout when no data is available
  //
  Py_INCREF(self);
  noDataTimeout = getNoDataTimeout(self);
  Py_DECREF(self);

  std::cout<<"IS STOP_READOUT SET = ["<<flagStopReadout<<"]\n"<<std::flush;
  
  Py_INCREF(self);
  bool eba = checkSelfCalculate(self);
  Py_DECREF(self);

  std::cout<<"[debug] is \"calculate\" set? = "
           <<ansi_orange<<"["<<eba<<"]"<<ansi_reset<<"\n"<<std::flush;  

  scream("\n_______________________STARTIN'_WHILE_LOOP___________________________\n");
  while (true){/*just use inf loop that breaks by first signal of shutter being closed*/

    auto iter_start = tick();//----------------------------> start here

    // SHUTTER state check
    Py_INCREF(self);
    int shutter_now = checkSHUTTER(self); // ~6 us
    Py_DECREF(self);

    //fprintf(stdout, ">>> [t1]=%.4f\n",timeDiff(iter_start,tick()));

    std::cout<<ansi_green<<"[DEBUG] (while) iteration starts"<<ansi_reset<<"\n"<<std::flush;
    // Allocating memory for output tuple
    // add a check if *tuple is NULL! release GIL if yes
    PyObject *tuple = PyTuple_New(tpx3::NUM_DEQUE_TUPLE_ITEMS);
    if(tuple==NULL){
        scream("Could not instantiate data tuple");
        PyGILState_Release(gstate); 
        return NULL;
    };

    // initializin' objects for data and rx error counters
    //
    PyObject *fifoData, *fifo_decerr, *fifo_discerr;
  
    // recod fifo data locally
    //
    Py_INCREF(fifo);
    std::vector<uint32_t> data = local_getFifoData(fifo,interval);// usually conforms to
                                                                  // interval set by user
                                                                  // with acc-cy of O(us)

    Py_DECREF(fifo);
    //fprintf(stdout, ">>> [t2]=%.4f\n",timeDiff(iter_start,tick()));

    long int n_words = data.size();

    // recording number of recorded words
    //
    global_words_recorded += n_words;
    
    // convertning number of words read in interation to pyobject tto fill deque
    // on the python side for (self.__words_per_read) calculation
    //
    PyObject *py_n_words = PyLong_FromLong(n_words);
    std::cout<<"\n"<<ansi_red<<"[DEBUG] got ["
             <<n_words<<"] words"<<ansi_reset<<"\n"<<std::flush;
    
    // this one puts vector.data() into numpy array
    //
    fifoData = fillPyArray(data); // ~ 40 us @ 100ms interval

    //fprintf(stdout, ">>> [t3]=%.4f\n",timeDiff(iter_start,tick()));
    /////////////////////
    // reading fifo discard and decoding error lists
    //
    Py_INCREF(self); 
    fifo_discerr = readFifoStatus(self,tpx3::CNTR_DISCARD_ERR);
    Py_DECREF(self); 
    //scream("[DEBUG] got discard errors");

    Py_INCREF(self); 
    fifo_decerr = readFifoStatus(self,tpx3::CNTR_DECODING_ERR);
    Py_DECREF(self); 
    //scream("[DEBUG] got decode errors");

    ///////////////////////////////
    // counting numbers in error lists
    Py_INCREF(fifo_discerr);
    long int n_errdisc = getNerror(fifo_discerr);
    Py_DECREF(fifo_discerr);
   
    Py_INCREF(fifo_decerr);
    long int n_errdec = getNerror(fifo_decerr);
    Py_DECREF(fifo_decerr);
    ///////////////////////////////
    //fprintf(stdout, ">>> [t4]=%.4f\n",timeDiff(iter_start,tick()));

    /////////////////////////////////////////////
    // code below takes ~3ms to execute
    // and its "leaky" 
    //Py_INCREF(RXLIST);
    //long int n_errdisc = readErrorCounters(RXLIST, tpx3::CNTR_DISCARD_ERR);
    //Py_DECREF(RXLIST);

    //Py_INCREF(RXLIST);
    //long int n_errdec = readErrorCounters(RXLIST, tpx3::CNTR_DECODING_ERR);
    //Py_DECREF(RXLIST);

    // update timestamp   
    float t_end = timeDiff(iter_start,tick());//---------------> ~53ms from start

    curr_time = last_read + t_end;
    scream("[DEBUG] got timestamps"); 

    //fprintf(stdout, ">>> [t5]=%.4f\n",timeDiff(iter_start,tick()));
    // assemble python tuple that will sent to fifo_readout::worker function
    //
    PyTuple_SetItem(tuple, 0, fifoData);
    PyTuple_SetItem(tuple, 1, PyFloat_FromDouble((double)last_read));
    PyTuple_SetItem(tuple, 2, PyFloat_FromDouble((double)curr_time)); 
    PyTuple_SetItem(tuple, 3, PyLong_FromLong(n_errdisc));
    PyTuple_SetItem(tuple, 4, PyLong_FromLong(n_errdec));

    //make the current time -> last_read time
    last_read = curr_time;

    scream("[DEBUG] Assembled the tuple"); 
    // Add python tuple to deque:
    Py_INCREF(deque);
    PyObject *result = PyObject_CallMethod(deque, "append", "(O)", tuple);
    if(result==NULL){
        scream("[DEBUG] CAN NOT access deque!");
        Py_DECREF(result);
        Py_DECREF(self);
        PyGILState_Release(gstate);
        return NULL;
    }
    // < 1.0 us to set tuple items and append to deque

    //fprintf(stdout, ">>> [t6]=%.4f\n",timeDiff(iter_start,tick()));
    Py_DECREF(result);
    Py_DECREF(tuple);
    Py_DECREF(deque);
    // setting obj-ts to NULL EXCEPT deque!
    result = NULL;
    tuple = NULL;
    fifoData = NULL;
    fifo_discerr = NULL;
    fifo_decerr = NULL;

    //scream("[DEBUG] Appended data to DEQUE"); 

    //////////////////////////////////////////////////////////////
    // THIS ONE IS --NONE-- if you print it... (cuz i dont append anything to it!)
    // Add n words to calculate average reading rate
    //
    Py_INCREF(self); //+self 
    PyObject *wpr = PyObject_GetAttrString(self,"_words_per_read");
    PyObject *wpread = PyObject_CallMethod(wpr, "append", "O", py_n_words);

    if(wpread==NULL){
        scream("[PyObject ISSUE] Could not append to _words_per_read!\n");
    }
    Py_DECREF(self); //-self
    //scream("");
    //PyObject_Print(wpread,stdout,0);
    Py_DECREF(wpread);// -wpread
    Py_DECREF(wpr);// -wpread
    wpread=NULL;
    wpr=NULL;
    scream("\n[DEBUG] Added to _words_per_read"); 
    //////////////////////////////////////////////////////////////
    
    // reset rx error counters if detected any discard
    // or decoding errors // Now actually works!
    //
    //fprintf(stdout, ">>> [t7]=%.4f\n",timeDiff(iter_start,tick()));
    if(n_errdisc != 0 || n_errdec !=0){
        std::cout<<ansi_orange<<"[DEBUG] Detected: (N_DISCARD="<<n_errdisc
                 <<"), (N_DECODE="<<n_errdec<<") errors.\n"<<ansi_reset<<std::flush;

        Py_INCREF(self);
        resetRXErrorCounters(self);
        Py_DECREF(self);
        n_errdisc = 0;
        n_errdec = 0;
        
    }

    scream("[DEBUG] Passed RESET RX"); 

    //fprintf(stdout, ">>> [t9]=%.4f\n",timeDiff(iter_start,tick()));

    float t_iteration = timeDiff(iter_start,tick());
    time_wait = interval - t_iteration;

    // temp timing debug
    //fprintf(stdout, "t_wait=%.4f, interval=%.4f, t_iter=%.4f\n"
    //        "curr_time=%.4f, last_read=%.4f, t_end=%.4f\n", 
    //        time_wait, interval, t_iteration, curr_time, last_read, t_end);

    /////// break loop if SHUTTER==1 -> 0 (closed)!
    //
    bool shutterDown = shutter_now==0 && prev_shutter==1 ? 1 : 0;
  
    if(shutterDown){
       std::string str = "\n\n_________________SHUTTER_DOWN->BREAKING_LOOP_!________________\n\n";
       std::cout<<str<<std::flush;
       data.clear();
       break;

    }

    prev_shutter = shutter_now;

    //////////////////////////////////////////

    data.clear();

    glob_tries++;
   
    scream("[DEBUG] iteration end"); 

  };// end the while(true) loop

    
  Py_INCREF(self);
  setSelfRecordCount(self);
  Py_DECREF(self);

  // Alternatively i can add NONE in the python func level
  // for now it sucks though....
  scream("[DEBUG] OUT OF WHILE LOOP!"); 

  //checkCallback(self);

  // adding Py_None tuple here
  //
  PyObject *lastTuple = PyTuple_Pack(1,Py_None);
  if(lastTuple==NULL){
    scream("Can not create tuple with PY_NONE!");
  }
  PyObject *append = PyObject_CallMethod(deque, "append", "O", lastTuple);
  if(append == NULL){
    scream("[DEBUG] big sad, can not append to deque...");
  }else{
    scream("[DEBUG] <<POGCHAMP!>> added tuple with __PY_NONE__!");
  }
  
  Py_DECREF(append);
  Py_DECREF(lastTuple);
  append = NULL;
  lastTuple=NULL;

  std::cout<<"N global tries = "<<glob_tries<<"!\n"<<std::flush;
  //scream("[DEBUG] FIFO_SIZE @ END is:");
  Py_INCREF(fifo);
  checkFifoSize(fifo);
  Py_DECREF(fifo);

  scream("[DEBUG] - readoutToDeque DONE!"); 
  ///// finita la comedia! ////// 

  PyGILState_Release(gstate);   

  return Py_None;

};

// end of class

}


//////////////////////////////////////////////////
//
    //TODO: realize this:
    // checking other fifo_readout attributes:
    // [v] self.rx_error_reset
    // [] self._calculate.clear()
    // [] self._result.put(sum(self._words_per_read)) // this mf does not have implementation in python side...
    // [] check for n_words==0 && self.stop-readout.is_set() to break sampling loop
    //    check for 
    //      [] self.callback-> put data to self._data_deque
    //      [] self.fill_buffer-> put data in self._data_buffer
    // [] add entries to self.logger.debug(...)
    //
    /////////////////////////////////////////////////////////////////
    //if(PyObject_IsTrue(PyObject_GetAttrString(self, "_calculate"))){
    //    scream("[DEBUG] found self._calculate");
    //    PyObject *calc = PyObject_GetAttrString(self,"_calculate");
    //    PyObject *result = PyObject_GetAttrString(self,"_result");
    //    Py_INCREF(calc);
    //    PyObject *res = PyObject_CallMethod(calc, "clear");
    //    if(res==NULL){
    //        scream("could not clear self._calculate ...");
    //    }
    //    
    //    PyObject *put = PyObject_CallMethod(result,"put","i",)
    //    Py_DECREF(res);
    //    Py_DECREF(calc);
    //}

    //TODO: realize this:
    //check if -> self._calculate_result.is_set()
    //if yes self._calculate_result.clear()
///////////////////////////////////////////////////////////////////////////
//
//  list of funtions to try without PyGILState_Ensure/Release()!
//  [] local_getFifoData
//  [] localQuerryFifo
//  [] fillVectorFromPyData
//  [] fillPyArray
//  [] checkSHUTTER
//  [] getNoDataTimeout
//  [] resetRXErrorCounters
//  [] checkFifoSize
//
///////////////////////////////////////////////////////////////////////////////////
// for some fuck reason the execution time between 
// Py_DECREF(fifo) after reading data from fifo
// and 
// filling the pytuple takes several ms
// AND explodes to O(100ms) after the SHUTTER goes 1->0
// ???
///////////////////////////////////////////////////////////////////////////////////
//
//  try caching atributes as:
//  
//   PyObject* getShutterAttribute(PyObject* self) {
//    // Cache the 'chip' attribute reference if not already cached
//    static PyObject* chipAttr = NULL;
//    if (chipAttr == NULL) {
//        chipAttr = PyObject_GetAttrString(self, "chip");
//        if (chipAttr == NULL) {
//            // Handle error
//        }
//    }
//
//    // Cache the 'CONTROL' attribute reference if not already cached
//    static PyObject* controlAttr = NULL;
//    if (controlAttr == NULL) {
//        controlAttr = PyObject_GetItem(chipAttr, PyUnicode_FromString("CONTROL"));
//        if (controlAttr == NULL) {
//            // Handle error
//        }
//    }
//
//    // Cache the 'SHUTTER' attribute reference if not already cached
//    static PyObject* shutterAttr = NULL;
//    if (shutterAttr == NULL) {
//        shutterAttr = PyObject_GetItem(controlAttr, PyUnicode_FromString("SHUTTER"));
//        if (shutterAttr == NULL) {
//            // Handle error
//        }
//    }
//
//    return shutterAttr;
//}
//
//int checkSHUTTER(PyObject* self) {
//    PyGILState_STATE gstate = PyGILState_Ensure();
//
//    // Get the cached 'SHUTTER' attribute reference
//    PyObject* shutterAttr = getShutterAttribute(self);
//    if (shutterAttr == NULL) {
//        // Handle error
//    }
//
//    // Convert the attribute to an integer
//    int status = std::atoi(PyUnicode_AsUTF8(PyObject_Str(shutterAttr)));
//
//    PyGILState_Release(gstate);
//    return status;
//}
//
//
    //std::stringstream initssA;
    //initssA<<"[DEBUG] (while) REFERENCE COUNTERS:"
    //  <<"<self>="<<Py_REFCNT(self)
    //  <<", <chip>="<<Py_REFCNT(chip)
    //  <<", <fifo>="<<Py_REFCNT(fifo)
    //  <<"\n"<<std::flush;
    //scream(initssA.str());

//

