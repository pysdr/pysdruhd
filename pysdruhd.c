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
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

#include "usrp_wrapper.h"

static PyMethodDef mymethods[] = {
        {NULL, NULL, 0, NULL} /* Sentinel */
};


PyMODINIT_FUNC
initpysdruhd(void)
{
    PyObject *m = Py_InitModule("pysdruhd", mymethods);
    import_array();

    if (PyType_Ready(&UsrpType) < 0)
        return;
    Py_INCREF(&UsrpType);
    PyModule_AddObject(m, "Usrp", (PyObject *)&UsrpType);
}