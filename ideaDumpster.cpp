//
//
//  THIS IS JUST AN IDEA STORAGE
// 
// 
//  DO NOT COMPILE OR USE DIRECTLY THIS FILE 
// 
// 
//
PyObject* devRecordFifoData(PyObject *chip, float t_interval){

    // put mutex lock on the bytecode of the object in interpretr
    // to have thread-safe execution 
    PyGILState_STATE gstate = PyGILState_Ensure(); 
    Py_Initialize();
    import_array();

    auto start_time = std::chrono::high_resolution_clock::now();
    
    ///// forward object defs
    PyObject *result;
    PyArrayObject *copy;

    scream("I'm in \"devRecordFifoData\"");
    //////////
    std::vector<uint32_t> temp;
    std::vector<uint32_t> fifo_data;
    std::vector<std::string> addr_init, addr_end;
    //uint32_t * fifo_array; 

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
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout<<"while loop took ["<<timeDiff(start_time, end_time)<<"] ms\n"<<std::flush;
    //if(fifo_data.size()>0){
        //scream("First vector data check:\n");
        //check_data(fifo_data, addr_init);
        //scream("\n------------------\n");
    
    //tpxdecoder(fifo_data);

    //fifo_array = &fifo_data[0];
    std::size_t dsize = fifo_data.size();
    //std::memcpy(fifo_array,fifo_data.data(), dsize); // *** stack smashing detected ***(thats new....)
    //checkArray(fifo_array,fifo_data.size() );

    //}
    //uint32_t * fifo_array(fifo_data.data(),fifo_data.size());// SF? 
    //uint32_t *fifo_array(&fifo_data[0],dsize);// compiler complains 
    //uint32_t *fifo_array[dsize] = &fifo_data[0];// compiler complains

    //scream("n_tries=["+std::to_string(cnt)+"]\n");
    ////
    //printVectUint(fifo_data);
    ///
    //const npy_intp nwords_received[] = {(npy_intp)fifo_data.size()};
    const npy_intp test_dsize[] = {(npy_intp)(dsize)};
    //void* reint_data = reinterpret_cast<void*>(fifo_data.data());// pointer to memory 

    std::cout<<"got ["<<dsize<<"] words\n"<<std::flush;
    //std::cout<<"test_dsize ["<<test_dsize<<"]\n"<<std::flush;
    //std::cout<<"nwords_received ["<<nwords_received<<"] words\n"<<std::flush;
    std::cout<<"first data element at:  ["<<&fifo_data[0]<<"]\n"<<std::flush;
    //std::cout<<"first array element at:  ["<<&fifo_array[0]<<"]\n"<<std::flush;
    //std::cout<<"first array data at:  ["<<&fifo_array<<"]\n"<<std::flush;
    if(dsize>2){
        std::cout<<"last data element at:  ["<<&fifo_data[dsize-1]<<"]\n"<<std::flush;
    //    std::cout<<"last array element at:  ["<<&fifo_array[dsize-1]<<"]\n"<<std::flush;
    }

    //scream("A");  
    //if(fifo_data.data() == &fifo_data[0]){
    //    scream("[G] data is contiguous!");
    //}else{
    //    scream("[G] huy v rot a ne contiguous!");
    //}
    //printArray(fifo_array,dsize);
    //#################################################
    PyObject* array = PyArray_SimpleNew(1,test_dsize,NPY_UINT32);
    if(array==NULL){
        scream("ARRAY IS NULL!");
    }

    uint32_t* thisData = reinterpret_cast<uint32_t*>(
            PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)));

    thisData = &fifo_data[0];

    scream("EEEEMacaena!");
    if(fifo_data.data()!=NULL){
        //scream("Second vector data check:\n");
        //check_data(fifo_data, addr_end);
        //scream("\n------------------\n");
        //compare_addr(addr_init, addr_end);
        //scream("fifo_data valid");
        //scream("AAI!");

        // --------------------------------------
        //      try this tomorrow 
        // --------------------------------------
        //uint32_t* udata = (uint32_t*)malloc(test_dsize * sizeof(uint32_t));

        //uint32_t* udata = (uint32_t*)malloc(dsize * sizeof(uint32_t));

        //void* udata = malloc(dsize * sizeof(uint32_t));
        //void* udata = malloc(temp_size * sizeof(uint32_t));

        ////udata = &fifo_data[0];
        //udata = &temp_data[0];

        ////PyObject *tmp = makeNumpyArray(udata, dsize);
        //PyObject *tmp = makeNumpyArray(udata, temp_size);

        ////fifo_data.clear();
        //free(udata);
        //udata = NULL;
        //Py_DECREF(tmp)
        // --------------------------------------
        //PyObject *tmp = PyArray_SimpleNewFromData(1,
        //                                   test_dsize,
        //                                   NPY_UINT32,
        //                                   //&fifo_array[0]);
        //                                   fifo_data.data());

        //PyArray_ENABLEFLAGS((PyArrayObject*)tmp, NPY_ARRAY_OWNDATA);
        //PyArray_ENABLEFLAGS((PyArrayObject*)tmp, NPY_ARRAY_ENSURECOPY);

        //PyObject_SetBaseObject(<<PyArrayObject*>>,<<PyObject*>>); 

        scream("exited \"devRecordFifoData\"");
        PyGILState_Release(gstate); 
        //return tmp;
        return array;
        //return Py_BuildValue("O", tmp);
        //return PyArray_SimpleNewFromData(1,
        //                                   test_dsize,
        //                                   NPY_UINT32,
        //                                   //&fifo_array[0]);
        //                                   fifo_data.data());

//        return Py_BuildValue("[i]",fifo_data.data());


    }else{/*else block will stay as is*/
        //scream("fifo_data NULL");
        scream("exited \"devRecordFifoData\"");
        fifo_data.clear();
        PyGILState_Release(gstate); 
        return Py_None;
        //return PyArray_ZEROS(1,test_dsize,NPY_UINT32,0);//instant segfault...
    }

    ///============== dev part ENDS =====================
    //release mutex lock
    Py_Finalize();
    PyGILState_Release(gstate); 

    return Py_None;

};


    // PIZDIT!
    void check_data(std::vector<uint32_t> vect, std::vector<std::string> &compVector){
        for(std::size_t i = 0; i < vect.size(); i++ ){
            //std::cout<<&vect[i]<<" "<<std::flush;
            if(i % 8 == 0 || i ==0){
                if(i==0){
                    std::cout<<"["<<&vect[i]<<"]\n"<<std::flush;
                }else{
                    std::cout<<&vect[i]<<" "<<std::flush;
                }
                if(i % 32 ==0){
                    std::cout<<"\n"<<std::flush;
                }
            }
            const unsigned int *x = new unsigned int(&vect[i]);
            std::ostringstream thisAdd;
            thisAdd << x;
            compVector.push_back(thisAdd.str());
            delete x;
        }
    };


//template <typename T>
//    PyObject* vector_to_pylist(const std::vector<T> &cpp_vec){
//                               //PyObject *(*ConvertToPy)(const T&)){
//        assert(cpython_asserts());
//        //PyObject *r = PyList_New(cpp_vec.size());
//        PyObject *r = PyArray_simpleNewFromData(1, cpp_vec.size(), NPY_UINT32, &cpp_vec.data());
//        if(!r){
//            goto except;
//        }
//        for(Py_ssize_t i = 0; i<cpp_vec.size(); ++i){
//            PyObject * item = (*ConvertToPy)(cpp_vec[i]);
//            if(!item || PyErr_Occurred() || PyArray_SetItem(r, i, item)){
//                goto except;
//            }
//        }
//        assert(!PyErr_Occurred());
//        assert(r);
//        goto finally;
//    except:
//        assert(PyErr_Occurred());
//        if(r){
//            for(Py_ssize_t i = 0; i < PyList_GET_SIZE(r); ++i){
//                Py_XDECREF(PyList_GET_ITEM(r,i));
//            }
//            Py_DECREF(r);
//            r = NULL;
//        }
//    finally:
//        return r;
//    };
//



///////////////////////////////////////////////////////////////////////////////////
//                      stuff from above to-be-used, maybe....
//Py_INCREF(querry);
//PyObject* result_array = PyArray_FROM_OTF(querry,
//                                          NPY_UINT32,
//                                          NPY_ARRAY_C_CONTIGUOUS); 

//PyArray_Descr* dscr = PyArray_DESCR(
//                      reinterpret_cast<PyArrayObject*>(querry));
//npy_intp* dimns = PyArray_DIMS(
//                      reinterpret_cast<PyArrayObject*>(querry));

//PyObject* result_array = 
//          PyArray_SimpleNewFromData(1, 
//                                    , 
//                                    NPY_UINT32, 
//                                    reinterpret_cast<void*>(querry)); 

//return reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1,0,NPY_UINT32));// segfault
//return reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1,NULL, NPY_UINT32, 0));// segfault
//return PyArray_SimpleNew(1,0,NPY_UINT32);// segfault


// since i always pass empty array initially, the pointer to shape is always zero
//npy_intp* curSize = reinterpret_cast<npy_intp*>(data_dims[0]); 

//npy_uint32* devRecordFifoData(PyObject *chip, float t_interval){
//NPY_UINT32* devRecordFifoData(PyObject *chip, float t_interval){ // gives  error: ‘NPY_UINT’ does not name a type; did you mean ‘NPY_UINT8’?


    //}else{

    //    std::cout<<"EBAT FIFO NE NABRALOS!\n"<<std::flush;
    //    std::cout<<fifo_data.data()<<"\n"<<std::flush; // this is zero if vector is empty
    //    //result = reinterpret_cast<PyObject*>(PyArray_SimpleNew(1,0,NPY_UINT32)); //segfault
    //    //result = reinterpret_cast<PyObject*>(PyArray_EMPTY(1,NULL,NPY_UINT32,0)); //segfault
    //    Py_INCREF(result);
    //    result = 0;

    //    scream("EBAL");

    //}


    //result_vector.data() -> address in memory
    // -----//-----.size() -> atual number of words
    //std::cout<<result_vector.data()<<","<<std::flush;
    //if(result_vector.data()!=NULL){
    //    for(int i=0; i<result_vector.size();i++){
    //        std::cout<<result_vector.at(i)<<"|"<<std::flush;
    //    }
    //}
    // check below says that result_vector data is contiguous
    //if(result_vector.data() == &result_vector[0]){
    //    scream("[L] result is contiguous!");
    //}else{
    //    scream("[L] huy v rot a ne contiguous!");i
    //}


       // FOKIN SEGFAULT HERE! (if i assign it to local shit)

       //          |
       //          V
//        PyObject *local_result = PyArray_ZEROS(1,test_dims,NPY_UINT32,0);
//        scream("YA");
//        Py_XINCREF(local_result);
//        PyObject_Print(local_result,stdout,0);
//        scream("EBAL");
//        PyObject *caps = PyCapsule_New(copy,"yoba", (PyCapsule_Destructor)&capsule_cleanup);
//        scream("ETO");
//        int baseIsSet = PyArray_SetBaseObject(copy,caps);
//        scream("..ETO...");
//        if(baseIsSet == -1){
//            scream("SRANOE");
//            Py_DECREF(result);
//            return NULL;
//        }
//        scream("GAVNNO");
//        int r = PyArray_CopyObject(copy,local_result);
      //  //PyObject *tempres = PyArray_ZEROS(1,(npy_intp*)(1),NPY_UINT32,0);
      //  PyObject *tempres = PyArray_SimpleNewFromData(1,
      //                                     nwords_received, 
      //                                     NPY_UINT32, 
      //                                     fifo_data.data())



