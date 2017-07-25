/*
 * Copyright 2017 Nathan West.
 *
 * This file is part of pysdruhd.
 *
 * pysdruhd is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * pysdruhd is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>   // sudo apt-get install python-dev python3-dev
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

#include "usrp_wrapper.h"


// Python3 stuff, straight from https://docs.python.org/3/howto/cporting.html
struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyObject *
error_out(PyObject *m) {
    struct module_state *st = GETSTATE(m);
    PyErr_SetString(st->error, "something bad happened");
    return NULL;
}
/*
static PyMethodDef mymethods[] = {
    {"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},
    {NULL, NULL}
};
*/

static PyMethodDef mymethods[] = {
        {NULL, NULL, 0, NULL} // Sentinel
};


// Python 3's module definition
#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, // according to docs Always initialize this member to PyModuleDef_HEAD_INIT
        "pysdruhd", // Name for the new module
        NULL, // Docstring for the module; usually a docstring variable created with PyDoc_STRVAR() is used
        -1, // Setting m_size to -1 means the module does not support sub-interpreters because it has global state
        mymethods, // A pointer to a table of module-level functions, same ones used in python2
        NULL, // Prior to version 3.5, this member was always set to NULL
        NULL, // A traversal function to call during GC traversal of the module object, or NULL if not needed.
        NULL, // A clear function to call during GC clearing of the module object, or NULL if not needed.
        NULL  // A function to call during deallocation of the module object, or NULL if not needed.
};
#endif


/* Initialization function for the module */
#if PY_MAJOR_VERSION >= 3
#define RETVAL m
PyMODINIT_FUNC PyInit_pysdruhd(void)
#else
#define RETVAL
PyMODINIT_FUNC
initpysdruhd(void)
#endif
{
    PyObject *m;
    #if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
    #else
    m = Py_InitModule("pysdruhd", mymethods);
    #endif
    import_array();
    if (PyType_Ready(&UsrpType) < 0)
        return;
    Py_INCREF(&UsrpType);
    PyModule_AddObject(m, "Usrp", (PyObject *)&UsrpType);
#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}
