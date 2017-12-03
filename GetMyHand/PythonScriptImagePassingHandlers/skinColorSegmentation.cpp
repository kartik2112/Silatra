//Explanation and references information in /PythonInterfacingTrial/*


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


bool initializeEnvironmentFor_CV_Python(){
    PyRun_SimpleString("import sys\n"
    "sys.path.insert(0, './')");  
    // This statement added from reference: https://stackoverflow.com/a/24492775/5370202

    // \n"
    // "from keras.models import model_from_json, Model\n"
    // "from keras import backend as k\n"
    // "from numpy import array,uint8\n"
    // "from keras.activations import relu\n"
    // "from math import floor\n"
    // "from PIL import Image\n"
    // "from os import system\n"
    // "import cv2, time, numpy as np
}


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


    initializeEnvironmentFor_CV_Python();


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
            imshow("Original image before python processing",image);
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
            pResponse = PyObject_CallObject(pFunc, pArgs);
            // Py_DECREF(pMat);
            Py_DECREF(pArgs);   //DECReffing pArgs would DECRef pMat as well because pArgs tuple has pMat
            //Reference for reason of commenting DECRef of pMat:
            //https://stackoverflow.com/a/14244382/5370202
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


