#ifndef __classifyPythonAPI_hpp_INCLUDE__
#define __classifyPythonAPI_hpp_INCLUDE__

#include <python3.5/Python.h>

#include <iostream>
#include <boost/python.hpp>

using namespace std;
using namespace boost::python;

bool initializePythonInterpreter();
long long predictSignByKNN_Py_Interface(char* data);

#endif