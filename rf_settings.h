//
// Created by Nathan West on 2/1/18.
//

#ifndef PYSDRUHD_RF_SETTINGS_H
#define PYSDRUHD_RF_SETTINGS_H

#include "usrp_object.h"
#include <uhd.h>

static const char set_frequency_docstring[] =
        "set_frequency(subdev, center_frequency=900e6, offset=0)\n\n"
                "    set_frequency tunes the provided subdev to with a frequency spec. The spec can either be floating"
                " point type parameters with kwargs center_frequency and offset, or a tuple with"
                " (center_frequency, offset). The return value is the actual value that was tuned to.";

static PyObject *
Usrp_set_frequency(Usrp *self, PyObject *args, PyObject *kwds) {
    double frequency = 900e6;
    double offset = 0.0;
    PyObject *frequency_tuple = NULL;
    char *subdev_spec = NULL;

    static char *kwlist[] = {"subdev", "center_frequency", "offset", "frequency_spec", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|ddO", kwlist,
                                     &subdev_spec, &frequency, &offset, &frequency_tuple
    )) {
        return NULL;
    }

    pysdr_subdev_t subdev;
    subdev = subdev_from_spec(self, subdev_spec);

    if (frequency_tuple != NULL && PyTuple_Check(frequency_tuple)) {
        if (PyTuple_Size(frequency_tuple) == 2) {
            frequency = PyFloat_AsDouble(PyTuple_GetItem(frequency_tuple, 0));
            offset = PyFloat_AsDouble(PyTuple_GetItem(frequency_tuple, 1));
        } else {
            PyErr_SetString(PyExc_TypeError, "Usrp.set_frequency(frequency_tuple) was provided a tuple with the wrong "
                    "number of items. Two items (center frequency, offset) are required");
            return NULL;
        }
    }

    // From uhd/host/lib/types/tune.cpp, for no offset tuning you can set UHD_TUNE_REQUEST_POLICY_AUTO
    // but I don't know of any reason to just provide a default 0.0 offset and always do manual (which avoids forks!)
    uhd_tune_request_t tune_request = {
            .target_freq = frequency,
            .rf_freq_policy = UHD_TUNE_REQUEST_POLICY_MANUAL,
            .rf_freq = frequency + offset,
            .dsp_freq_policy = UHD_TUNE_REQUEST_POLICY_AUTO,
            .dsp_freq = 0.0,
    };
    double checkval;
    uhd_tune_result_t tune_result;

    if (subdev.mode == RX_STREAM) {
        if (!uhd_ok(uhd_usrp_set_rx_freq(*self->usrp_object, &tune_request, (size_t) subdev.index, &tune_result))) {
            return NULL;
        }
        if (!uhd_ok(uhd_usrp_get_rx_freq(*self->usrp_object, (size_t) subdev.index, &checkval))) {
            return NULL;
        }
        self->rx_streams[subdev.index].frequency = frequency;
        self->rx_streams[subdev.index].lo_offset = offset;
        frequency = tune_result.actual_rf_freq;
    } else if (subdev.mode == TX_STREAM) {
        if (!uhd_ok(uhd_usrp_set_tx_freq(*self->usrp_object, &tune_request, (size_t) subdev.index, &tune_result))) {
            return NULL;
        }
        if (!uhd_ok(uhd_usrp_get_tx_freq(*self->usrp_object, (size_t) subdev.index, &checkval))) {
            return NULL;
        }
        self->tx_streams[subdev.index].frequency = frequency;
        self->tx_streams[subdev.index].lo_offset = offset;
        frequency = tune_result.actual_rf_freq;
    }

    return PyFloat_FromDouble(frequency);
}


typedef enum GAIN_MODE {
    NORMALIZED, DB
};
static const char set_gain_docstring[] =
        "set_gain(subdev, gain=1.0, mode='relative')\n\n"
                "    set_gain sets the gain of provided subdev to the specified gain. The gain can either be a relative "
                "gain in range [0,1] or absolute gain in dB. The return value is the actual gain that was set. By "
                "default relative gain is used with a default value of 1.0 (full gain)";

static PyObject *
Usrp_set_gain(Usrp *self, PyObject *args, PyObject *kwds) {
    double gain = 1.0;
    char *gain_mode = NULL;
    char *subdev_spec = NULL;

    static char *kwlist[] = {"subdev", "gain", "mode", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|sds", kwlist,
                                     &subdev_spec, &gain, &gain_mode
    )) {
        return NULL;
    }

    pysdr_subdev_t subdev;
    subdev = subdev_from_spec(self, subdev_spec);
    if (strncmp(gain_mode, "normalized", 10) == 0) {

    }

    double checkval;
    if (subdev.mode == RX_STREAM) {
        if (!uhd_ok(uhd_usrp_set_normalized_rx_gain(*self->usrp_object, gain, (size_t) subdev.index))) {
            return NULL;
        }
        if (!uhd_ok(uhd_usrp_get_normalized_rx_gain(*self->usrp_object, (size_t) subdev.index, &checkval))) {
            return NULL;
        }
    } else if (subdev.mode == TX_STREAM) {
        if (!uhd_ok(uhd_usrp_set_normalized_tx_gain(*self->usrp_object, gain, (size_t) subdev.index))) {
            return NULL;
        }
        if (!uhd_ok(uhd_usrp_get_normalized_tx_gain(*self->usrp_object, (size_t) subdev.index, &checkval))) {
            return NULL;
        }
    }

    return PyFloat_FromDouble(checkval);
}


static const char set_rate_docstring[] =
        "set_rate(subdev, rate)\n\n"
                "    set_rate sets the sample rate of provided subdev to the specified rate. The return value is the "
                "actual rate that was set.";

static PyObject *
Usrp_set_rate(Usrp *self, PyObject *args, PyObject *kwds) {
    double rate = 1.0;
    char *subdev_spec = NULL;

    static char *kwlist[] = {"subdev", "rate", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|d", kwlist,
                                     &subdev_spec, &rate
    )) {
        return NULL;
    }

    pysdr_subdev_t subdev;
    subdev = subdev_from_spec(self, subdev_spec);
    double checkval = -1.0;
    if (subdev.mode == RX_STREAM) {
        checkval = pysdr_set_rx_rate(*self->usrp_object, self->rx_streams[subdev.index].rate, (size_t) subdev.index);
    } else if (subdev.mode == TX_STREAM) {
        checkval = pysdr_set_tx_rate(*self->usrp_object, self->tx_streams[subdev.index].rate, (size_t) subdev.index);
    } else {
        return NULL;
    }

    if (checkval < 0.0) {
        return NULL;
    }

    return PyFloat_FromDouble(checkval);
}

#endif //PYSDRUHD_RF_SETTINGS_H
