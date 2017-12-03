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
        export CPLUS_INCLUDE_PATH=/usr/include/python3.5m
        make all -j8`
        This worked.


*/


int main(int argc, char *argv[])
{
    PyObject *pName, *pModule, *pDict, *pFunc;
    PyObject *pArgs, *pMat, *pResponse;
    int i;

    if (argc < 3) {
        fprintf(stderr,"Usage: call pythonfile funcname [imageFilePath]\n");
        return 1;
    }

    double startTime=(double)getTickCount();

    Py_Initialize();

    cout<<"Check 1"<<endl;

    PyRun_SimpleString("import sys\n"
    "sys.path.insert(0, './')\n"
    "print(sys.path)");  // This statement added from reference: https://stackoverflow.com/a/24492775/5370202

    cout<<"Check 2"<<endl;

    pName = PyUnicode_DecodeFSDefault(argv[1]);
    /* Error checking of pName left out */

    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    cout<<"Check 3"<<endl;

    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, argv[2]);
        /* pFunc is a new reference */

        if (pFunc && PyCallable_Check(pFunc)) {

            Mat image = imread(argv[3],1);
            cout<<"Image ("<<image.size().width<<","<<image.size().height<<") loaded"<<endl;

            cout<<"Check 4"<<endl;

            pArgs = PyTuple_New(1);
            imshow("Original image before python processing",image);
            pMat = pyopencv_from(image);
            PyTuple_SetItem(pArgs, 0, pMat);

            cout<<"Check 5"<<endl;

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
            pResponse = PyObject_CallObject(pFunc, pArgs);
            // Py_DECREF(pMat);
            Py_DECREF(pArgs);   //DECReffing pArgs would DECRef pMat as well because pArgs tuple has pMat
            //Reference for reason of commenting DECRef of pMat:
            //https://stackoverflow.com/a/14244382/5370202

            cout<<"Check 6"<<endl;

            cout<<PyTuple_Check(pResponse)<<endl;
            if (pResponse != NULL ) {
                // printf("Result of call: %ld\n", PyLong_AsLong(pResponse));
                cout<<"Received frame from python script"<<endl;
                // pMat = PyTuple_GetItem(pResponse,0);
                pyopencv_to(pResponse,image,"Grayscale");
                imshow("Image after python processing",image);
                Py_DECREF(pResponse);
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

    waitKey(0);

    return 0;
}


