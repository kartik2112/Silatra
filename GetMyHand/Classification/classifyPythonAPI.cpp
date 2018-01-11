/*

Author: Kartik Shenoy
All used references cited wherever necessary

*/



// #include <Python.h>  //This doesn't work for python3. This will work for python2

/**
Reference: https://docs.python.org/2/extending/embedding.html
*/

/**

To compile this file from commandline, use:
    `g++ classifyPythonAPI.cpp -I/usr/local/include/python3.5 -lpython3.5m`

Here, 
-I is used to include a directory
    Directory to be included here is `/usr/local/include/python3.5`
-l is used to include a library
    Library to be included here is `python3.5m`

To run a.out file from commandline, use:
    `./a.out`

For parameter passing main function, use:
    `./a.out pythonscript multiply 5 3`

*/


/*
If you are getting error: 
    fatal error: pyconfig.h: No such file or directory
    compilation terminated.
You should first execute: export CPLUS_INCLUDE_PATH=/usr/include/python3.5m

In builder.sh file, this is already included

Reference: https://github.com/BVLC/caffe/issues/410
*/


/**
Reference: http://answers.opencv.org/question/25642/how-to-compile-basic-opencv-program-in-c-in-ubuntu/
For defining the necesary Mat to PyObject conversion functions, wrapper functions for Opencv3 were used
Ref: https://github.com/spillai/numpy-opencv-converter/issues/9

To compile this file from commandline use:
    `g++ c++script.cpp -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib/ -I/usr/local/include/python3.5
     -g 
     -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect 
     -lopencv_contrib -lopencv_legacy -lopencv_stitching -lpython3.5m`
Preferably use ./builder.sh The compiling will be simpler

Using ./builder technique is easier.

Run `./builder.sh`
Then run
    `./build/PythonInterfacingTrial pythonscript displayImageParams ./4.png`

*/


#include <python3.5/Python.h>

#include <iostream>
#include <boost/python.hpp>
// For this to work you need to install boost
// sudo apt-get install libboost-all-dev
// Reference: https://stackoverflow.com/a/12578564/5370202

using namespace std;
using namespace boost::python;


/**
Second Attempt: Python function invocation with argument passing to function
*/
PyObject *pName, *pModule, *pDict, *pFunc;
PyObject *pArgs, *pValue;

bool PythonInterpreterInvoked = false;

char *moduleName = "testThisSampleInPython", *funcName = "predictSignByKNN";

bool initializePythonInterpreter(){
    Py_Initialize();

    PyRun_SimpleString("import sys\n"
    "sys.path.insert(0, './')\n"
    "if not hasattr(sys, 'argv'): sys.argv  = ['']\n");  // This statement added from reference: https://stackoverflow.com/a/24492775/5370202

    // pName = PyUnicode_DecodeFSDefault(moduleName);
    // /* Error checking of pName left out */

    // pModule = PyImport_Import(pName);
    // Py_DECREF(pName);

    // if (pModule != NULL) {
    //     pFunc = PyObject_GetAttrString(pModule, funcName);
    //     /* pFunc is a new reference */

    //     if (pFunc && PyCallable_Check(pFunc)) {
    //         pArgs = PyTuple_New(1);
    //         PythonInterpreterInvoked = true;
    //         return true;
    //     }
    //     else {
    //         if (PyErr_Occurred())
    //             PyErr_Print();
    //         fprintf(stderr, "Cannot find function \"%s\"\n", funcName);
    //     }
    //     Py_XDECREF(pFunc);
    //     Py_DECREF(pModule);
    // }
    // else {
    //     PyErr_Print();
    //     fprintf(stderr, "Failed to load \"%s\"\n", moduleName);        
    // }
    // PythonInterpreterInvoked = false;
    // return false;
}

long predictSignByKNN_Py_Interface(char* CCDC_Data){
    // if(!PythonInterpreterInvoked){
    //     initializePythonInterpreter();
    // }




    pName = PyUnicode_DecodeFSDefault(moduleName);
    /* Error checking of pName left out */

    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, funcName);
        /* pFunc is a new reference */

        if (pFunc && PyCallable_Check(pFunc)) {
            pArgs = PyTuple_New(1);
            PythonInterpreterInvoked = true;
            // return true;

            // cout<<CCDC_Data<<endl;

            pValue = PyUnicode_FromString(CCDC_Data); //Reference: https://docs.python.org/3.5/c-api/unicode.html#c.PyUnicode_FromString
            if (!pValue) {
                Py_DECREF(pArgs);       // Py_DECREF would simply free the memory allocated
                Py_DECREF(pModule);
                fprintf(stderr, "Cannot convert argument\n");
                
                // Py_Finalize();
                // initializePythonInterpreter();
            }
            /* pValue reference stolen here: */
            PyTuple_SetItem(pArgs, 0, pValue);
            
            pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);
            if (pValue != NULL) {
                printf("Predicted Sign: %ld\n", PyLong_AsLong(pValue));
                Py_DECREF(pValue);
            }
            else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                fprintf(stderr,"Call failed\n");

                // Py_Finalize();
                // initializePythonInterpreter();
            }


        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \"%s\"\n", funcName);
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", moduleName);        
    }
    PythonInterpreterInvoked = false;
    // return false;







    
}

// int main(int argc, char *argv[])
// {
//     if(initializePythonInterpreter()){
//         predictSignByKNN_Py_Interface(argv[1]);
//     }
    
//     // if (Py_FinalizeEx() < 0) {
//     //     return 120;
//     // }

//     Py_Finalize();
//     return 0;
// }