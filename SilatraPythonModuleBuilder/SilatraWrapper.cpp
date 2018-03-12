// Main Reference: https://docs.python.org/3.5/extending/extending.html
// Python 2.7 and Python 3.5 have different ways of creating modules,
// To handle this, Macros must be added. Reference: https://stackoverflow.com/a/42578018/5370202
// #include <Python.h>


#include <python3.5/Python.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "GetMyHand/handDetection.hpp"
// #include "GetMyHand/Classification/classifyPythonAPI.hpp"

#include <iostream>
#include <ctime>
#include <experimental/filesystem>



// These are for passing images across Python C++ interface
#include "opencv2/core/types_c.h"
#include "opencv2/opencv_modules.hpp"

#include <boost/python.hpp>
// For this to work you need to install boost
// sudo apt-get install libboost-all-dev
// Reference: https://stackoverflow.com/a/12578564/5370202

#include "PythonInterfacingEssentials/cv_cpp_py_interface.h"

#if PY_MAJOR_VERSION >= 3
#define PY3K
#endif


#define OVERALL 0
#define SKIN_COLOR_EXTRACTION 1
#define MORPHOLOGY_OPERATIONS 2
#define MODIFIED_IMAGE_GENERATION 3
#define HAND_CONTOURS_GENERATION 4
#define CONTOURS_PRE_PROCESSING 5
#define CONTOURS_IMPROVEMENT 6
#define CONTOURS_POST_PROCESSING 7
#define CONTOUR_CLASSIFICATION_IN_PY 8

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem;




using namespace boost::python;

long long predictedSign;


Rect faceBBox;
bool faceFound = false;





// int maxNoOfSamples = 300;

// void processFrame(Mat& image);
// void maintainTrackOfTimings();

string subDirName;

string tempTimesLabels[] = 
					{"Overall",
					"  Skin Color Extraction",
					"  Morphology Operations",
					"  Modified Image Generation",
					"  Hand Contours Generation",
					"    Contours Pre-processing (Canny, ApproxPolyDP)",
					"    Contours Improvement (Connect, reduce cluster points)",
					"    Contours Post-processing (Convex Hull, Normalized CCDC computation n storage)",
					"    Contour Classification (Python invocation)"};

vector<string> timesLabels(tempTimesLabels, tempTimesLabels + sizeof(tempTimesLabels)/sizeof(string));
vector<double> maxTimes(timesLabels.size(),0);
vector<double> minTimes(timesLabels.size(),10000);
vector<double> avgTimes(timesLabels.size(),0);
vector<double> frameStepsTimes(timesLabels.size());
double noOfFramesCollected = 0;

char** args_v;
int args_c;

bool wrapperModeOn = true;


// static PyObject *SpamError;


// module functions
static PyObject * 
processFrame(PyObject *self, PyObject *args)
{
    // const char *command;
    // int sts;

    // if (!PyArg_ParseTuple(args, "s", &command))
    //     return NULL;
    // sts = system(command);
    // if (sts < 0) {
    //     PyErr_SetString(SpamError, "System command failed");
    //     return NULL;
    // }

    PyObject *returnList = PyList_New(3);  //PyList references: https://docs.python.org/3.5/c-api/list.html

    PyObject *PyImg, *PyFaceFound, *faceRect;

    // cout<<"H1"<<endl;

    // cout<<PyTuple_Check(args)<<endl;

    if (!PyArg_ParseTuple(args, "O", &PyImg)){  //Reference for format string: https://docs.python.org/2.0/ext/parseTuple.html
        //Don't mistake it as small o or 0. It is capital 'O' Reference: https://stackoverflow.com/a/13276530/5370202
        return NULL;
    }

    Mat image;

    pyopencv_to(PyImg,image,"Incoming image from Python");

    // cout<<"H2"<<endl;

    Mat dst = getMyHand(image);

    // cout<<"HFin"<<endl;

    // Py_INCREF(Py_None);
    // return Py_None;  //Reference to none return type: https://docs.python.org/3.5/extending/extending.html#back-to-the-example
    // return PyLong_FromLongLong(predictedSign);
    // return pyopencv_from(dst);

    PyImg = pyopencv_from(dst);

    if(faceFound){
        PyFaceFound = Py_True;
        faceRect = Py_BuildValue("(iiii)", faceBBox.x, faceBBox.y, faceBBox.width, faceBBox.height); //This function is derived from cv_cpp_py_interface.cpp. 
                                                                    //Instead of this the function could have been uncommented and used. 
                                                                    //But for having a better understanding, it wasnt uncommented.
    }
    else{
        PyFaceFound = Py_False;
        faceRect = Py_None;
    }

    /* Reference: https://docs.python.org/3.5/c-api/list.html */
    PyList_SetItem(returnList,0,PyImg);
    PyList_SetItem(returnList,1,PyFaceFound);
    PyList_SetItem(returnList,2,faceRect);

    return returnList;

}

// registration table
static PyMethodDef SilatraMethods[] = {
    {"segment",  processFrame, METH_VARARGS,
     "Pass an image containing some sign to this function"},
    {NULL, NULL, 0, NULL}
};



#ifdef PY3K

// module definition structure for python3
static struct PyModuleDef silatramodule = {
   PyModuleDef_HEAD_INIT,
   "silatra",   /* name of module */
   "Sign Language Translation",/*spam_doc,  module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
   SilatraMethods
};




PyMODINIT_FUNC PyInit_silatra(void)
{
    return PyModule_Create(&silatramodule);
}



#else

// module initializer for python2
PyMODINIT_FUNC initsilatra() {
    Py_InitModule3("silatra", SilatraMethods, "Sign Language Translation");
}

#endif
