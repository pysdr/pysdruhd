//
// Created by Nathan West on 2/1/18.
//

#ifndef PYSDRUHD_USRP_TIME_H
#define PYSDRUHD_USRP_TIME_H

#include "usrp_object.h"
#include <uhd.h>

static const char set_master_clock_rate_docstring[] =
        "set_master_clock_rate(rate)"
                "set the master clock rate to rate, a floating point/double type. Usually has some impact on ADC/DAC rate.";

static PyObject *
Usrp_set_master_clock_rate(Usrp *self, PyObject *args) {
    double clock_rate;
    if (!PyArg_ParseTuple(args, "d",
                          &clock_rate)) {
        return NULL;
    }

    uhd_usrp_set_master_clock_rate(*self->usrp_object, clock_rate, 0);
    uhd_usrp_get_master_clock_rate(*self->usrp_object, 0, &clock_rate);
    return PyFloat_FromDouble(clock_rate);
}


static const char set_time_docstring[] =
        "set_time(when='pps', time=0).\n\n"
                "   `time` is either a tuple of (full seconds, fractional seconds), 'gps', or 'pc'. full seconds should be an integer"
                " type and fractional seconds should be a float/double type. Default time is 0. 'gps' will set the device time to match"
                " the GPS time on the on-board GPSDO if it exists. 'pc' will set the device time to whatever localtime returns.\n"
                "   `when` should be a string matching either 'now' or 'pps'. Default is 'pps' which sets the time at the next PPS. If"
                " pps is used with 'gps' or 'pc' time, then the \n";

static PyObject *
Usrp_set_time(Usrp *self, PyObject *args, PyObject *kwds) {
    PyObject *when = NULL, *time = NULL, *offset = NULL;
    static char *kwlist[] = {"time", "when", "offset", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", kwlist,
                                     &time, &when, &offset
    )) {
        return NULL;
    }

    // We'll use the raw string value in 2 places, so just fetch it now
    char *when_data = NULL;
    if (when != NULL && PyString_Check(when)) {
        when_data = PyString_AsString(when);
    }

    int full_secs = 0;
    double fractional_secs = 0.0;

    if (time != NULL) {
        if (PyString_Check(time)) { /* if we got a string it should say gps */
            if (strncmp(PyString_AsString(time), "gps", MIN((size_t) PyString_Size(time), 3)) == 0) {
                // we want to set the time next pps to whatever the gpsdo is
                uhd_sensor_value_handle sensor_value;
                uhd_sensor_value_make_from_string(&sensor_value, "w", "t", "f");
                uhd_error uhd_errno = uhd_usrp_get_mboard_sensor(*self->usrp_object, "gps_time", 0, &sensor_value);
                if (uhd_errno == UHD_ERROR_NONE) {
                    uhd_sensor_value_data_type_t sensor_dtype;
                    uhd_sensor_value_data_type(sensor_value, &sensor_dtype);
                    full_secs = uhd_sensor_value_to_int(sensor_value, &full_secs);
                }
            } else if (strncmp(PyString_AsString(time), "pc", MIN((size_t) PyString_Size(time), 2)) == 0) {
                struct timeval tv;
                gettimeofday(&tv, NULL);
                full_secs = (int) tv.tv_sec;
                fractional_secs = (double) tv.tv_usec * 1e-6;
            }
            if (when_data != NULL && strncmp(when_data, "pps", 3) == 0) {
                if (fractional_secs > 0.) {
                    fractional_secs = 0.0;
                    full_secs += 1;
                }
            }
        } else if (PyTuple_Check(time) && PyTuple_Size(time) ==
                                          2) { /* if we got a tuple, then it's whole secs, fractional secs. a very sexy time */
            full_secs = (int) PyInt_AsLong(PyTuple_GetItem(time, 0));
            fractional_secs = PyInt_AsLong(PyTuple_GetItem(time, 1));
        }
    }

    if (when_data != NULL && strncmp(when_data, "now", 3) == 0) {
        uhd_usrp_set_time_now(*self->usrp_object, full_secs, fractional_secs, 0);
    } else if (when_data == NULL || strncmp(when_data, "pps", 3) == 0) { // default
        uhd_usrp_set_time_next_pps(*self->usrp_object, full_secs, fractional_secs, 0);
    } else {
        // when_data was provided, but we don't recognize it
        PyErr_Format(PyExc_TypeError,
                     "Usrp.set_time(...) when argument was %s. Argument is optional with acceptable values "
                             "of 'now' or 'pps'", when_data);
    }

    return Py_None;
}

#endif //PYSDRUHD_USRP_TIME_H
