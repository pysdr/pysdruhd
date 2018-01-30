//
// Created by Nathan West on 1/26/18.
//

#ifndef PYSDRUHD_CLOCK_AND_TIME_SOURCES_H
#define PYSDRUHD_CLOCK_AND_TIME_SOURCES_H

#include "wrapper_helper.h"
#include <uhd.h>

typedef enum {CLOCK, TIME} SOURCE_TYPE;

PyObject * get_source_list(Usrp *self, SOURCE_TYPE type)
{
    uhd_string_vector_handle sources;
    uhd_string_vector_make(&sources);

    switch (type) {
        case CLOCK:
            uhd_usrp_get_clock_sources(*self->usrp_object, 0, &sources);
            break;
        case TIME:
            uhd_usrp_get_time_sources(*self->usrp_object, 0, &sources);
            break;
        default:
            printf("got an unknown source type\n");
    }

    size_t number_sources;
    uhd_string_vector_size(sources, &number_sources);
    PyObject *sources_list = PyList_New((Py_ssize_t) number_sources);
    for (unsigned int tsource_index = 0; tsource_index < number_sources; ++tsource_index) {
        char this_time_source[128];
        const size_t this_time_source_length = 128;
        uhd_string_vector_at(sources, tsource_index, this_time_source, this_time_source_length);
        PyList_SetItem(sources_list, tsource_index, PyString_FromString(this_time_source));
    }
    uhd_string_vector_free(&sources);
    return sources_list;
}

static const char get_time_sources_docstring[] =
        "get a list of time sources";
static PyObject *
Usrp_get_time_sources(Usrp *self)
{
    return get_source_list(self, TIME);

}

static const char get_time_source_docstring[] =
        "get the current time source";
static PyObject *
Usrp_get_time_source(Usrp *self)
{
    const size_t bufflen = 128;
    char time_source[128];
    uhd_usrp_get_time_source(*self->usrp_object, 0, time_source, bufflen);
    return PyString_FromString(time_source);
}

static const char set_time_source_docstring[] =
        "set the current time source";
static PyObject *
Usrp_set_time_source(Usrp *self, PyObject *args)
{
    PyObject *source;

    if (!PyArg_ParseTuple(args, "O", &source)) {
        return NULL;
    }
    PyString_Check(source);

    if (!uhd_ok(uhd_usrp_set_time_source(*self->usrp_object, PyString_AsString(source), 0))) {
        PyErr_SetString(PyExc_AttributeError, "setting time on the usrp failed");
        return NULL;
    }
    return Py_None;
}

static const char get_clock_sources_docstring[] =
        "get a list of clock sources";
static PyObject *
Usrp_get_clock_sources(Usrp *self)
{
    return get_source_list(self, CLOCK);
}

static const char get_clock_source_docstring[] =
        "get the current clock source";
static PyObject *
Usrp_get_clock_source(Usrp *self)
{
    const size_t bufflen = 128;
    char clock_source[128];
    uhd_usrp_get_clock_source(*self->usrp_object, 0, clock_source, bufflen);
    return PyString_FromString(clock_source);
}

static const char set_clock_source_docstring[] =
        "set the current clock source";
static PyObject *
Usrp_set_clock_source(Usrp *self, PyObject *args)
{
    PyObject *source;

    if (!PyArg_ParseTuple(args, "O", &source)) {
        return NULL;
    }
    PyString_Check(source);

    if (!uhd_ok(uhd_usrp_set_clock_source(*self->usrp_object, PyString_AsString(source), 0))) {
        PyErr_SetString(PyExc_AttributeError, "setting clock on the usrp failed");
        return NULL;
    }
    return Py_None;
}

#endif //PYSDRUHD_CLOCK_AND_TIME_SOURCES_H
