// Main Reference: https://docs.python.org/3.5/extending/extending.html
// Python 2.7 and Python 3.5 have different ways of creating modules,
// To handle this, Macros must be added. Reference: https://stackoverflow.com/a/42578018/5370202
#include <Python.h>

#if PY_MAJOR_VERSION >= 3
#define PY3K
#endif

static PyObject *SpamError;


// module functions
static PyObject * 
spam_system(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    if (sts < 0) {
        PyErr_SetString(SpamError, "System command failed");
        return NULL;
    }
    return PyLong_FromLong(sts);
}

// registration table
static PyMethodDef SpamMethods[] = {
    {"system",  spam_system, METH_VARARGS,
     "Execute a shell command."},
    {NULL, NULL, 0, NULL}
};

#ifdef PY3K
// module definition structure for python3
static struct PyModuleDef spammodule = {
   PyModuleDef_HEAD_INIT,
   "spam",   /* name of module */
   "mod doc",/*spam_doc,  module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
   SpamMethods
};




PyMODINIT_FUNC PyInit_spam(void)
{
    return PyModule_Create(&spammodule);
}



#else

// module initializer for python2
PyMODINIT_FUNC initspam() {
    Py_InitModule3("spam", SpamMethods, "mod doc");
}

#endif
