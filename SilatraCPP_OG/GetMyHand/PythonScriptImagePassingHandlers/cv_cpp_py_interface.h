#include <Python.h>

#define MODULESTR "cv2"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

// #include "opencv2/core/types_c.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/core/types_c.h"
#include "opencv2/opencv_modules.hpp"

#include "pycompat.hpp"


static int failmsg(const char *fmt, ...);
struct ArgInfo;
static PyObject* failmsgp(const char *fmt, ...);
bool pyopencv_to(PyObject* o, cv::Mat& m, const char* name);
PyObject* pyopencv_from(const cv::Mat& m);
