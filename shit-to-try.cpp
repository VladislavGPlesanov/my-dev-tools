#include <vector>
#include <Python.h>

extern "C" {
    PyObject* query_data_from_device() {
        PyObject *pName, *pModule, *pFunc, *pValue;
        PyObject *pArgs, *pResult;
        
        // Initialize Python interpreter
        Py_Initialize();
        
        // Import the Python module
        pName = PyUnicode_DecodeFSDefault("your_python_module");
        pModule = PyImport_Import(pName);
        Py_DECREF(pName);

        // Check if module is imported successfully
        if (pModule != NULL) {
            // Get the function from the module
            pFunc = PyObject_GetAttrString(pModule, "query_data_from_device");

            // Check if the function is callable
            if (pFunc && PyCallable_Check(pFunc)) {
                // Call the function in a while loop until some condition is reached
                std::vector<int> data;
                while (true) {
                    // Call the Python function
                    pValue = PyObject_CallObject(pFunc, NULL);
                    
                    // Check if the function call was successful
                    if (pValue != NULL) {
                        // Convert the result to C++ data type
                        int result = PyLong_AsLong(pValue);
                        
                        // Append data to the array
                        data.push_back(result);
                        
                        // Release the object
                        Py_DECREF(pValue);
                    }
                    
                    // Some condition to break the loop
                    // For example, break when the size of the data array reaches a certain limit
                    if (data.size() >= 1000) {
                        break;
                    }
                }

                // Convert the C++ vector to a NumPy array
                PyObject* numpy_array = PyArray_SimpleNewFromData(1, data.size(), NPY_INT, data.data());

                // Return the NumPy array
                return numpy_array;
            } else {
                PyErr_Print();
                fprintf(stderr, "Error: Cannot find function\n");
            }
            Py_XDECREF(pFunc);
            Py_DECREF(pModule);
        } else {
            PyErr_Print();
            fprintf(stderr, "Error: Cannot import module\n");
        }
        
        // Cleanup
        Py_Finalize();
        
        return NULL;
    }
}




extern "C" {
    PyObject* querryDataFromFifo(PyObject *chip, PyObject *timeDiff, PyObject *interval){
        if (chip != NULL) {
            // Convert timeDiff and interval from Python float to C++ double
            double pTime = PyFloat_AsDouble(timeDiff);
            double pInterval = PyFloat_AsDouble(interval);

            // Check if chip is callable
            if (chip && PyCallable_Check(chip)) {
                // Call the chip function
                PyObject* pValue = PyObject_CallObject(chip, NULL);

                // Check if the function call was successful
                if (pValue != NULL) {
                    std::vector<uint32_t> fifo_data;

                    // Convert interval from seconds to C++ time_t
                    time_t interval_sec = static_cast<time_t>(pInterval);

                    // Get current time
                    time_t start_time = time(NULL);

                    // While loop to query data until the time difference exceeds the interval
                    while (difftime(time(NULL), start_time) < interval_sec) {
                        uint32_t data = PyLong_AsLong(pValue);
                        fifo_data.push_back(data);
                    }

                    // Release the object
                    Py_DECREF(pValue);

                    // Convert the C++ vector to a NumPy array (not implemented here)
                    // Return the NumPy array (not implemented here)
                }
            }
        }
        return NULL;
    }
}


if (chip != NULL) {
            // Convert timeStart and interval from Python float to C++ double
            double pTimeStart = PyFloat_AsDouble(timeStart);
            double pInterval = PyFloat_AsDouble(interval);
            
            // Check if chip is a dictionary
            if (PyDict_Check(chip)) {
                // Get the 'string' key from the chip dictionary
                PyObject* string_chip = PyDict_GetItemString(chip, "string");
                
                // Check if string_chip is not NULL and callable
                if (string_chip != NULL && PyCallable_Check(string_chip)) {
                    // Call the read_data method of string_chip
                    PyObject* pValue = PyObject_CallMethod(string_chip, "read_data", NULL);
                    
                    // Check if the function call was successful
                    if (pValue != NULL) {
                        std::vector<uint32_t> fifo_data;

                        // Convert timeStart and interval from seconds to C++ time_t
                        std::time_t time_start_sec = static_cast<std::time_t>(pTimeStart);
                        std::time_t interval_sec = static_cast<std::time_t>(pInterval);

                        // Get the start time
                        std::time_t start_time = std::time(nullptr);

                        // While loop to query data until the time difference exceeds the interval
                        while (std::difftime(std::time(nullptr), start_time) < interval_sec) {
                            // Retrieve data from pValue (you may need to adjust this based on the actual data type)
                            uint32_t data = PyLong_AsLong(pValue);
                            fifo_data.push_back(data);
                        }

                        // Release the object
                        Py_DECREF(pValue);

                        // Convert the C++ vector to a NumPy array (not implemented here)
                        // Return the NumPy array (not implemented here)
                    }
                }
            }
        }

        return nullptr;
    }
