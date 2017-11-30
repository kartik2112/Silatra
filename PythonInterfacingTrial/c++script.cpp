// #include <Python.h>  //This doesn't work for python3. This will work for python2

/**
Reference: https://docs.python.org/2/extending/embedding.html
*/

/**

To compile this file from commandline, use:
    `g++ c++script.cpp -I/usr/local/include/python3.5 -lpython3.5m`

Here, 
-I is used to include a directory
    Directory to be included here is `/usr/local/include/python3.5`
-l is used to include a library
    Library to be included here is `python3.5m`

To run a.out file from commandline, use:
    `./a.out`

For parameter passing main function, use:
    `./a.out pythonscript multiply 5,3`

*/


/**
Reference: http://answers.opencv.org/question/25642/how-to-compile-basic-opencv-program-in-c-in-ubuntu/
To compile this file from commandline use:

*/


#include <python3.5/Python.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "opencv2/core/types_c.h"
#include "opencv2/opencv_modules.hpp"
// #include "pycompat.hpp"

#include <iostream>
#include <boost/python.hpp>

#include "cv_cpp_py_interface.h"

using namespace std;
using namespace cv;
using namespace boost::python;

/**
First Attempt: Normal python script invocation
*/

// int main(int argc, wchar_t *argv[])
// {
//     // Py_SetProgramName(argv[0]);  /* optional but recommended */
//     Py_Initialize();
//     //   PyRun_SimpleString("from time import time,ctime\n"
//     //                      "print 'Today is',ctime(time())\n");

//     FILE* file = fopen("pythonscript.py","r");
//     PyRun_SimpleFile(file,"pythonscript.py");
//     // PyRun_SimpleFile(file,"LoadSavedModel.py");
//     // Py_Finalize();   //Moved to main() of Silatra.cpp
//     fclose(file);
//     Py_Finalize();
//     return 0;
// }


/**
Second Attempt: Python function invocation with argument passing to function
*/

// int main(int argc, char *argv[])
// {
//     PyObject *pName, *pModule, *pDict, *pFunc;
//     PyObject *pArgs, *pValue;
//     int i;

//     if (argc < 3) {
//         fprintf(stderr,"Usage: call pythonfile funcname [args]\n");
//         return 1;
//     }

//     Py_Initialize();

//     PyRun_SimpleString("import sys\n"
//     "sys.path.insert(0, './')\n");  // This statement added from reference: https://stackoverflow.com/a/24492775/5370202

//     // PyRun_SimpleString("import sys\n"
//     // "sys.path.insert(0, '/home/kartik/Documents/GDrive/Ubuntu Synced/OPENCV/Silatra/PythonInterfacingTrial/')\n");  // This statement added from reference: https://stackoverflow.com/a/24492775/5370202

//     pName = PyUnicode_DecodeFSDefault(argv[1]);
//     /* Error checking of pName left out */

//     pModule = PyImport_Import(pName);
//     Py_DECREF(pName);

//     if (pModule != NULL) {
//         pFunc = PyObject_GetAttrString(pModule, argv[2]);
//         /* pFunc is a new reference */

//         if (pFunc && PyCallable_Check(pFunc)) {
//             pArgs = PyTuple_New(argc - 3);
//             for (i = 0; i < argc - 3; ++i) {
//                 pValue = PyLong_FromLong(atoi(argv[i + 3]));
//                 if (!pValue) {
//                     Py_DECREF(pArgs);       // Py_DECREF would simply free the memory allocated
//                     Py_DECREF(pModule);
//                     fprintf(stderr, "Cannot convert argument\n");
//                     return 1;
//                 }
//                 /* pValue reference stolen here: */
//                 PyTuple_SetItem(pArgs, i, pValue);
//             }
//             pValue = PyObject_CallObject(pFunc, pArgs);
//             Py_DECREF(pArgs);
//             if (pValue != NULL) {
//                 printf("Result of call: %ld\n", PyLong_AsLong(pValue));
//                 Py_DECREF(pValue);
//             }
//             else {
//                 Py_DECREF(pFunc);
//                 Py_DECREF(pModule);
//                 PyErr_Print();
//                 fprintf(stderr,"Call failed\n");
//                 return 1;
//             }
//         }
//         else {
//             if (PyErr_Occurred())
//                 PyErr_Print();
//             fprintf(stderr, "Cannot find function \"%s\"\n", argv[2]);
//         }
//         Py_XDECREF(pFunc);
//         Py_DECREF(pModule);
//     }
//     else {
//         PyErr_Print();
//         fprintf(stderr, "Failed to load \"%s\"\n", argv[1]);
//         return 1;
//     }
//     // if (Py_FinalizeEx() < 0) {
//     //     return 120;
//     // }

//     Py_Finalize();
//     return 0;
// }



/**

After a lot of surfing found:
https://github.com/opencv/opencv/blob/2.4.2/modules/python/src2/cv2.cpp
Line 291: static PyObject* pyopencv_from(const Mat& m)

This is opencv c++ wrapper for python. So this is a file in python's library. 
Can't use this here according to my knowledge.

If you get "fatal error: pyconfig.h: No such file or directory"
        `sudo apt-get install python-dev libxml2-dev libxslt-dev`
        Tried this. Didn't work
    Referring: https://github.com/BVLC/caffe/issues/410
        `make clean
        export CPLUS_INCLUDE_PATH=/usr/include/python2.7
        make all -j8`
        This worked.



*/


int main(int argc, char *argv[])
{
    PyObject *pName, *pModule, *pDict, *pFunc;
    PyObject *pArgs, *pMat, *pValue;
    int i;

    if (argc < 3) {
        fprintf(stderr,"Usage: call pythonfile funcname [imageFilePath]\n");
        return 1;
    }

    double startTime=(double)getTickCount();

    Py_Initialize();


    PyRun_SimpleString("import sys\n"
    "sys.path.insert(0, './')\n");  // This statement added from reference: https://stackoverflow.com/a/24492775/5370202


    pName = PyUnicode_DecodeFSDefault(argv[1]);
    /* Error checking of pName left out */

    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, argv[2]);
        /* pFunc is a new reference */

        if (pFunc && PyCallable_Check(pFunc)) {

            Mat image = imread(argv[3],1);
            cout<<"Image ("<<image.size().width<<","<<image.size().height<<") loaded"<<endl;

            pArgs = PyTuple_New(1);
            pMat = pyopencv_from(image);
            PyTuple_SetItem(pArgs, 0, pMat);

            // pArgs = PyTuple_New(argc - 3);
            // for (i = 0; i < argc - 3; ++i) {
            //     pValue = PyLong_FromLong(atoi(argv[i + 3]));
            //     if (!pValue) {
            //         Py_DECREF(pArgs);       // Py_DECREF would simply free the memory allocated
            //         Py_DECREF(pModule);
            //         fprintf(stderr, "Cannot convert argument\n");
            //         return 1;
            //     }
            //     /* pValue reference stolen here: */
            //     PyTuple_SetItem(pArgs, i, pValue);
            // }
            pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pMat);
            if (pValue != NULL) {
                printf("Result of call: %ld\n", PyLong_AsLong(pValue));
                Py_DECREF(pValue);
            }
            else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                fprintf(stderr,"Call failed\n");
                return 1;
            }
        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \"%s\"\n", argv[2]);
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", argv[1]);
        return 1;
    }
    // if (Py_FinalizeEx() < 0) {
    //     return 120;
    // }

    Py_Finalize();

    cout<<"Time taken by entire endeavour: "<<(getTickCount()-(double)startTime)/getTickFrequency()<<endl;

    return 0;
}


