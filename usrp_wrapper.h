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

#ifndef PYSDRUHD_USRP_WRAPPER_H
#define PYSDRUHD_USRP_WRAPPER_H


#include "usrp_object.h"
#include "clock_and_time_sources.h"
#include "wrapper_helper.h"
#include "sensors.h"
#include <uhd.h>
#include <stdio.h>
#include <string.h>
#include <structmember.h>
#include <sys/time.h>
#include <numpy/ndarrayobject.h>


static PyMemberDef Usrp_members[] = {
        {NULL}  /* Sentinel */
};


static const char recv_docstring[] =
        "samples, metadata = Usrp.recv()\n\n"
                "    return an tuple of (samples, metadata). Samples are of shape (nchannels, nsamples) where nchannels matches "
                "the number of subdevs specified during construction and nsamples is the number of samples in the packet returned "
                "by UHD. Metadata is a tuple with a timespec of the first sample in this packet.";

static PyObject *
Usrp_recv(Usrp *self) {
    size_t rx_samples_count = 0;
    time_t full_secs = 0;
    double frac_secs = 0.;

    if (!uhd_ok( uhd_rx_streamer_recv(*self->rx_streamer, self->recv_buffers_ptr, self->samples_per_buffer,
                                     self->rx_metadata, 3.0, false, &rx_samples_count) )) {
        return NULL;
    }

    if (!uhd_ok( uhd_rx_metadata_time_spec(*self->rx_metadata, &full_secs, &frac_secs) )) {
        return NULL;
    }
    PyObject *metadata = PyTuple_New(2);
    PyTuple_SET_ITEM(metadata, 0, PyInt_FromSize_t((size_t)full_secs));
    PyTuple_SET_ITEM(metadata, 1, PyFloat_FromDouble(frac_secs));

    npy_intp shape[2];
    shape[0] = self->number_rx_streams;
    shape[1] = rx_samples_count;

    PyObject *return_val = PyTuple_New(2);
    PyTuple_SET_ITEM(return_val, 0, PyArray_SimpleNewFromData(2, shape, NPY_COMPLEX64, self->recv_buffers));
    PyTuple_SET_ITEM(return_val, 1, metadata);
    return return_val;
}


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


static const char send_stream_command_docstring[] =
        "send_stream_command(mode='continuous', when='now')\n\n"
                "    send_stream_command will create and issue a stream command to UHD. UHD stream commands send an rx or tx "
                "streamer a command that contains a `stream_mode` enum, `stream_now` bool, and optionall a timespec. This wrapper "
                "accepts the mode as a string and when can either be a string matching 'now' or a tuple of (full secs, fractional "
                "secs).\n\n"
                "N.B. from UHD docs: to use an on-board GPSDO use set_time_source('external') and set_clock_source('external')";
// Ooops! this isn't actually doing anything yet
static PyObject *
Usrp_send_stream_command(Usrp *self, PyObject *args, PyObject *kwds) {
    PyObject *command;

    if (!PyArg_ParseTuple(args, "O", &command)) {
        return NULL;
    }

    PyObject *requested_commands;
    requested_commands = PyDict_Keys(command);
    Py_ssize_t number_of_commands = PyList_Size(requested_commands);
    for (Py_ssize_t command_number=0; command_number < number_of_commands; ++command_number) {
        PyObject *command_key;
        command_key = PyList_GetItem(requested_commands, command_number);
        printf("got a command with key %s ", PyString_AsString(command_key));
        printf("that has a value of %s\n", PyString_AsString(PyDict_GetItem(command, command_key)));
    }
    /* This should really be made adjustable, which would make this wrapper probably
     * the only place that this is an easy way to get streaming started at a time
     * that you care about
     */
    uhd_stream_cmd_t stream_cmd = {
            .stream_mode = UHD_STREAM_MODE_START_CONTINUOUS,
            .stream_now = true,
            //.time_spec_full_secs = 1,
            //.time_spec_frac_secs= 0.,
    };
    if (!uhd_ok( uhd_usrp_set_time_source(*self->usrp_object, "internal", 0) )) {
        return NULL;
    }
    if (!uhd_ok( uhd_usrp_set_time_now(*self->usrp_object, 0, 0.0, 0) )) {
        return NULL;
    }
    if (!uhd_ok( uhd_rx_streamer_issue_stream_cmd(*self->rx_streamer, &stream_cmd) )) {
        return NULL;
    }

    return Py_None;
}


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
        if (!uhd_ok( uhd_usrp_set_rx_freq(*self->usrp_object, &tune_request, (size_t) subdev.index, &tune_result) )) {
            return NULL;
        }
        if (!uhd_ok( uhd_usrp_get_rx_freq(*self->usrp_object, (size_t) subdev.index, &checkval) )) {
            return NULL;
        }
        self->rx_streams[subdev.index].frequency = frequency;
        self->rx_streams[subdev.index].lo_offset = offset;
        frequency = tune_result.actual_rf_freq;
    } else if (subdev.mode == TX_STREAM) {
        if (!uhd_ok( uhd_usrp_set_tx_freq(*self->usrp_object, &tune_request, (size_t) subdev.index, &tune_result) )) {
            return NULL;
        }
        if (!uhd_ok( uhd_usrp_get_tx_freq(*self->usrp_object, (size_t) subdev.index, &checkval) )) {
            return NULL;
        }
        self->tx_streams[subdev.index].frequency = frequency;
        self->tx_streams[subdev.index].lo_offset = offset;
        frequency = tune_result.actual_rf_freq;
    }

    return PyFloat_FromDouble(frequency);
}


typedef enum GAIN_MODE {NORMALIZED, DB};
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


static const char send_docstring[] =
        "Usrp.send(samples, metadata)\n\n"
                "    send a vector of samples to the USRP for transmission";

static PyObject *
Usrp_send(Usrp *self, PyObject *args, PyObject *kwds) {

    static char *kwlist[] = {"samples", "metadata", NULL};
    PyObject *samples=NULL, *metadata=NULL;

    Py_ssize_t args_size = PyTuple_Size(args);
    if (args_size > 0) {
        samples = PyTuple_GetItem(args, 0);
    }
    if (args_size == 2) {
        metadata = PyTuple_GetItem(args, 1);
    }

    int ndims = PyArray_NDIM((PyArrayObject*) samples);
    size_t samples_per_buffer = 0;
    int number_channels = 0;
    npy_intp *samples_shape;
    samples_shape = PyArray_SHAPE((PyArrayObject*) samples);
    switch (ndims) {
        case 0:
            // this is an error
            break;
        case 1:
            samples_per_buffer = (size_t) samples_shape[0];
            number_channels = 1;
            break;
        case 2:
            number_channels = (int) samples_shape[0];
            samples_per_buffer = (size_t) samples_shape[1];
            break;
        default:
            // this is an error
            break;
    }
    uhd_tx_metadata_handle tx_metadata;
    uhd_tx_metadata_make(&tx_metadata, true, 0, 0.1, true, false);
    double timeout = 1.0;

    size_t rx_samples_count = 0;
    time_t full_secs = 0;
    double frac_secs = 0.;
    int tx_stream_index = 0;
    const void *samples_ptr = PyArray_DATA((PyArrayObject*) samples);

    size_t items_sent;
    uhd_tx_streamer_send(*self->tx_streamer, &samples_ptr, samples_per_buffer, &tx_metadata, timeout, &items_sent);
    uhd_tx_metadata_free(&tx_metadata);

    Py_RETURN_NONE;
}


static PyObject *
Usrp_str(Usrp *self) {

    // this is mildly dangerous. It's a reeeeal pain to keep track of char counts so let's just malloc and realloc
    // generously. yaaaaaay C!
    char *str_representation = malloc(1024);

    sprintf(str_representation, "{");
    if (self->usrp_type) {
        sprintf(str_representation, "%stype:'%s', ", str_representation, PyString_AsString(self->usrp_type));
    }
    if (self->addr) {
        sprintf(str_representation, "%s{addr:'%s', ", str_representation, PyString_AsString(self->addr));
    }
    sprintf(str_representation, "%s \n", str_representation);

    size_t current_str_len = 1024;

    // This should look something like {"subdev":{}...}
    for (unsigned int rx_stream_indx=0; rx_stream_indx < self->number_rx_streams; ++rx_stream_indx) {
        stream_config_t this_stream = self->rx_streams[rx_stream_indx];
        size_t this_str_len;

        char mode_str[5];
        if (this_stream.mode == RX_STREAM) {
            sprintf(mode_str, "%s", "RX");
        } else if (this_stream.mode == TX_STREAM) {
            sprintf(mode_str, "%s", "TX");
        }

        const char fmt_str[] = "%s '%s': {'mode': '%s', 'antenna': '%s', 'frequency': %1.2f, 'rate': %1.2f, 'gain': %1.2f},\n";
        this_str_len = strlen(fmt_str) + strlen(this_stream.antenna) + strlen(this_stream.subdev);
        str_representation = realloc(str_representation, current_str_len + this_str_len + 32);
        sprintf(str_representation, fmt_str,
                str_representation, this_stream.subdev, mode_str, this_stream.antenna,
                this_stream.frequency, this_stream.frequency, this_stream.gain);
    }
    for (unsigned int tx_stream_indx=0; tx_stream_indx < self->number_tx_streams; ++tx_stream_indx) {
        stream_config_t this_stream = self->tx_streams[tx_stream_indx];
        size_t this_str_len;

        char mode_str[5];
        if (this_stream.mode == RX_STREAM) {
            sprintf(mode_str, "%s", "RX");
        } else if (this_stream.mode == TX_STREAM) {
            sprintf(mode_str, "%s", "TX");
        }

        const char fmt_str[] = "%s '%s': {'mode': '%s', 'antenna': '%s', 'frequency': %1.2f, 'rate': %1.2f, 'gain': %1.2f},\n";
        this_str_len = strlen(fmt_str) + strlen(this_stream.antenna) + strlen(this_stream.subdev);
        str_representation = realloc(str_representation, current_str_len + this_str_len + 32);
        sprintf(str_representation, fmt_str,
                str_representation, this_stream.subdev, mode_str, this_stream.antenna,
                this_stream.frequency, this_stream.frequency, this_stream.gain);
    }

    sprintf(str_representation, "%s}", str_representation);
    PyObject *ret_str = PyString_FromString(str_representation);
    free(str_representation);

    return ret_str;
}

static PyObject *
Usrp_repr(Usrp *self) {
    // actually, I should sprintf this in a Usrp(...)
    return Usrp_str(self);
}


static PyMethodDef Usrp_methods[] = {
        {"recv",                  (PyCFunction) Usrp_recv,                  METH_NOARGS,  recv_docstring},
        {"send",                  (PyCFunction) Usrp_send,                  METH_VARARGS |
                                                                            METH_KEYWORDS,  send_docstring},
        {"sensor_names",          (PyCFunction) Usrp_sensor_names,          METH_NOARGS,  sensor_names_docstring},
        {"get_sensor",            (PyCFunction) Usrp_get_sensor,            METH_VARARGS |
                                                                            METH_KEYWORDS, get_sensor_docstring},
        {"set_master_clock_rate", (PyCFunction) Usrp_set_master_clock_rate, METH_VARARGS, set_master_clock_rate_docstring},
        {"set_time",              (PyCFunction) Usrp_set_time,              METH_VARARGS |
                                                                            METH_KEYWORDS, set_time_docstring},
        {"send_stream_command",   (PyCFunction) Usrp_send_stream_command,   METH_VARARGS |
                                                                            METH_KEYWORDS, send_stream_command_docstring},
        {"set_frequency",         (PyCFunction) Usrp_set_frequency,         METH_VARARGS |
                                                                            METH_KEYWORDS, set_frequency_docstring},
        {"set_gain",              (PyCFunction) Usrp_set_gain,              METH_VARARGS |
                                                                            METH_KEYWORDS, set_gain_docstring},
        {"set_rate",              (PyCFunction) Usrp_set_rate,              METH_VARARGS |
                                                                            METH_KEYWORDS, set_rate_docstring},
        {"get_time_sources",      (PyCFunction) Usrp_get_time_sources,      METH_NOARGS, get_time_sources_docstring},
        {"get_time_source",       (PyCFunction) Usrp_get_time_source,       METH_NOARGS, get_time_source_docstring},
        {"set_time_source",       (PyCFunction) Usrp_set_time_source,       METH_VARARGS, set_time_source_docstring},
        {"get_clock_sources",     (PyCFunction) Usrp_get_clock_sources,     METH_NOARGS, get_clock_sources_docstring},
        {"get_clock_source",      (PyCFunction) Usrp_get_clock_source,      METH_NOARGS, get_clock_source_docstring},
        {"set_clock_source",      (PyCFunction) Usrp_set_clock_source,      METH_VARARGS, set_clock_source_docstring},
        {NULL}  /* Sentinel */
};

static const char Usrp_docstring[] =
        "A friendly native python interface to USRPs. The constructor looks like this (all optional arguments): "

                "        Usrp(addr, type, streams, frequency, rate, gain)\n"
                "            addr: a string with the address of a network connected USRP\n"
                "            type: a string with the type of USRP (find with uhd_find_devices)\n"
                "            streams: a dictionary of the form {'subdev': {'frequency': <double>, 'rate': <double>, 'gain': <double>}, }\n"
                "            The keys within a subdev are optional and will take default values of the frequency, rate, and gain parameters:\n"
                "            frequency: <double> the center frequency to tune to\n"
                "            rate: <double> the requested sample rate\n"
                "            gain: <double> the requested gain\n"
                "\n"
                "        The primary function of a Usrp is to transmit and receive samples. These are handled through\n"
                "        samples, metadata = Usrp.recv()\n"
                "        Usrp.transmit(samples) (not implemented)\n"
                "\n"
                "        In both cases samples is a numpy array with shape (nchannels, nsamps)\n"
                "\n"
                "    There are several currently unsupported USRP features that are of interest:\n"
                "        * burst modes\n"
                "        * offset tuning (relatively easy to impl)\n"
                "        * settings transport args like buffer size and type conversions\n"
                "\n"
                "I'm also interested in implementing some of the as_sequence, etc methods so we can do things like "
                "reduce(frame_handler, filter(burst_detector, map(signal_processing, Usrp())))\n"
                "\n"
                "Until then, this is missing some more advances features and is crash-prone when things aren't butterflies and \n"
                "rainbows, but is at least capable of streaming 200 Msps in to python with no overhead.";


static void
Usrp_dealloc(Usrp *self) {
    Py_XDECREF(self->addr);
    Py_XDECREF(self->usrp_type);


    if (self->rx_streamer != NULL) {
        uhd_rx_streamer_free(self->rx_streamer);
        free(self->rx_streamer);
        self->rx_streamer = NULL;
    }

    if (self->rx_metadata != NULL) {
        uhd_rx_metadata_free(self->rx_metadata);
        free(self->rx_metadata);
        self->rx_metadata = NULL;
    }

    if (self->tx_streamer != NULL) {
        uhd_tx_streamer_free(self->tx_streamer);
        free(self->tx_streamer);
        self->tx_streamer = NULL;
    }

    if (self->usrp_object != NULL) {
        uhd_usrp_free(self->usrp_object);
        free(self->usrp_object);
        self->usrp_object = NULL;
    }

    if (self->rx_streams != NULL) {
        free(self->rx_streams);
        self->rx_streams = NULL;
    }

    if (self->tx_streams != NULL) {
        free(self->tx_streams);
        self->tx_streams = NULL;
    }

    if (self->recv_buffers != NULL) {
        free(self->recv_buffers);
        self->recv_buffers = NULL;
    }

    if (self->recv_buffers_ptr != NULL) {
        free(self->recv_buffers_ptr);
        self->recv_buffers_ptr = NULL;
    }

    Py_TYPE(self)->tp_free((PyObject *) self);
}

/*
 * NB: new is only ever called once. init can be called any number of times
 */
static PyObject *
Usrp_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Usrp *self = (Usrp *) type->tp_alloc(type, 0);

    if (self != NULL) {
        self->addr = NULL;
        self->usrp_type = NULL;

        self->samples_per_buffer = 0;
        self->recv_buffers_ptr = NULL;
        self->recv_buffers = NULL;

        self->usrp_object = malloc(sizeof(uhd_usrp_handle));

        self->rx_streamer = malloc(sizeof(uhd_rx_streamer_handle));
        self->rx_metadata = malloc(sizeof(uhd_rx_metadata_handle));
        self->number_rx_streams = 0;

        self->tx_streamer = malloc(sizeof(uhd_tx_streamer_handle));
        //self->tx_metadata = malloc(sizeof(uhd_tx_metadata_handle));
        self->number_tx_streams = 0;

        if (!uhd_ok(uhd_rx_streamer_make(self->rx_streamer))) {
            return NULL;
        }
        if (!uhd_ok(uhd_rx_metadata_make(self->rx_metadata))) {
            return NULL;
        }
        if (!uhd_ok(uhd_tx_streamer_make(self->tx_streamer))) {
            return NULL;
        }
    } else {
        /*
         * I'm not sure how this would happen, but something is horribly wrong.
         * The python Noddy example does this check though...
         */
        PyErr_BadArgument();
        return NULL;
    }

    return (PyObject *) self;
}


static int
Usrp_init(Usrp *self, PyObject *args, PyObject *kwds) {
    PyObject *addr = NULL, *addr2 = NULL, *tmp = NULL;
    PyObject *usrp_type = NULL;
    PyObject *streams_dict = NULL;
    double frequency_param = 910e6;
    double lo_offset_param = 0.0;
    double rate_param = 1e6;
    double gain_param = 0;

    static char *kwlist[] = {"addr", "addr2", "type", "streams", "frequency", "lo_offset", "rate", "gain", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOdddd", kwlist,
                                     &addr, &addr2, &usrp_type, &streams_dict, &frequency_param, &lo_offset_param,
                                     &rate_param, &gain_param
    )) {
        return -1;
    }

    char *device_args = malloc(40);
    memset(device_args, 0x0, 40);
    if (addr) {
        tmp = self->addr;
        Py_INCREF(addr);
        self->addr = addr;
        Py_XDECREF(tmp);
        snprintf(device_args, 18, "addr=%s,", PyString_AsString(addr));
    }
    if (addr2) {
        // Erg. what to do about the internal addr we keep around....
        char addr2string[64] = {'\0'}; // we could check the length of the python string
        snprintf(addr2string, 18, "second_addr=%s,", PyString_AsString(addr2));
        strcat(device_args, addr2string);
    }

    if (usrp_type) {
        tmp = self->usrp_type;
        Py_INCREF(usrp_type);
        self->usrp_type = usrp_type;
        Py_XDECREF(tmp);
        char typestring[64]; // we could check the length of the python string
        snprintf(typestring, 64, "type=%s", PyString_AsString(usrp_type));
        strcat(device_args, typestring);
    }

    if (!uhd_ok(uhd_usrp_make(self->usrp_object, device_args))) {
        return -1;
    }

    // Ideally interrogate the device to create an internal dict of channels/subdevs
    char usrp_pp[2048];
    uhd_usrp_get_pp_string(*self->usrp_object, usrp_pp, 2048);
    /* DEV HELPER */ printf("usrp: %s\n", usrp_pp);

    size_t channel[] = {0, 1, 2, 3, 4, 5, 6, 7};
    char rx_subdev_spec_string[64] = {'\0'};

    if (streams_dict) {
        if (PyDict_Check(streams_dict)) {
            parse_dict_to_streams_config(self, streams_dict, frequency_param, lo_offset_param, rate_param, gain_param,
                                         rx_subdev_spec_string);
        } else {
            PyErr_SetString(PyExc_TypeError, "streams argument needs to be a dict of form"
                    "    {'<DB>:<SUBDEV>': {'mode': 'RX'|'TX', 'frequency': double, 'rate': double, 'gain': double},}");
        }
    } else {
        // We didn't get a config dict, so default to create 1 RX stream on A:0
        self->rx_streams = malloc(sizeof(stream_config_t));
        self->number_rx_streams = 1;
        self->rx_streams[0].lo_offset = lo_offset_param;
        self->rx_streams[0].frequency = frequency_param;
        self->rx_streams[0].rate = rate_param;
        self->rx_streams[0].gain = gain_param;
        strncpy(self->rx_streams[0].subdev, "A:0\0", 4);
        strncpy(self->rx_streams[0].antenna, "RX2\0", 4);
    }

    // We should accept a stream_args dict
    uhd_stream_args_t stream_args = {
            .cpu_format = "fc32",
            .otw_format = "sc16",
            .args = "",
            .channel_list = channel,
            .n_channels = self->number_rx_streams
    };

    uhd_subdev_spec_handle subdev_spec;
    uhd_subdev_spec_make(&subdev_spec, rx_subdev_spec_string);
    uhd_usrp_set_rx_subdev_spec(*self->usrp_object, subdev_spec, 0);

    for (size_t rx_stream_index = 0; rx_stream_index < self->number_rx_streams; ++rx_stream_index) {
        if (!uhd_ok(uhd_usrp_set_rx_antenna(*self->usrp_object,
                                            self->rx_streams[rx_stream_index].antenna,
                                            channel[rx_stream_index]) )) {
            return -1;
        }

        PyObject *empty_arg = PyTuple_New(0);
        PyObject *rate_kws = Py_BuildValue("{s:s,s:d}", "subdev", self->rx_streams[rx_stream_index].subdev,
                                           "rate", self->rx_streams[rx_stream_index].rate);
        Usrp_set_rate(self, empty_arg, rate_kws);

        PyObject *gain_kws = Py_BuildValue("{s:s,s:d,s:s}", "subdev", self->rx_streams[rx_stream_index].subdev,
                                           "gain", self->rx_streams[rx_stream_index].gain,
                                            "mode", "normalized");
        Usrp_set_gain(self, empty_arg, gain_kws);

        PyObject *freq_kws = Py_BuildValue("{s:s,s:d,s:d}", "subdev", self->rx_streams[rx_stream_index].subdev,
                                           "center_frequency", self->rx_streams[rx_stream_index].frequency,
                                           "offset", self->rx_streams[rx_stream_index].lo_offset);
        Usrp_set_frequency(self, empty_arg, freq_kws);
    }
    if (self->number_rx_streams > 0) {
        if (!uhd_ok(uhd_usrp_get_rx_stream(*self->usrp_object, &stream_args, *self->rx_streamer))) {
            return -1;
        }

        if (!uhd_ok(uhd_rx_streamer_max_num_samps(*self->rx_streamer, &self->samples_per_buffer))) {
            return -1;
        }

        size_t buffer_size_per_channel = self->samples_per_buffer * 2 * sizeof(float);
        self->recv_buffers = malloc(self->number_rx_streams * buffer_size_per_channel);
        self->recv_buffers_ptr = malloc(sizeof(void *) * self->number_rx_streams);
        printf("making receive buffer sizes per channel %zu\n", buffer_size_per_channel);
        // watch out world, we're indexing void* 's!
        for (size_t rx_stream_index = 0; rx_stream_index < self->number_rx_streams; ++rx_stream_index) {
            self->recv_buffers_ptr[rx_stream_index] = self->recv_buffers + (rx_stream_index * buffer_size_per_channel);
        }
    }

    for (size_t tx_stream_index = 0; tx_stream_index < self->number_tx_streams; ++tx_stream_index) {
        if (!uhd_ok(uhd_usrp_set_tx_antenna(*self->usrp_object,
                                            self->tx_streams[tx_stream_index].antenna,
                                            channel[tx_stream_index]) )) {
            return -1;
        }

        PyObject *empty_arg = PyTuple_New(0);
        PyObject *rate_kws = Py_BuildValue("{s:s,s:d}", "subdev", self->tx_streams[tx_stream_index].subdev,
                                           "rate", self->tx_streams[tx_stream_index].rate);
        Usrp_set_rate(self, empty_arg, rate_kws);

        PyObject *gain_kws = Py_BuildValue("{s:s,s:d,s:s}", "subdev", self->tx_streams[tx_stream_index].subdev,
                                           "gain", self->tx_streams[tx_stream_index].gain,
                                           "mode", "normalized");

        Usrp_set_gain(self, empty_arg, gain_kws);
        PyObject *freq_kws = Py_BuildValue("{s:s,s:d,s:d}", "subdev", self->tx_streams[tx_stream_index].subdev,
                                           "center_frequency", self->tx_streams[tx_stream_index].frequency,
                                           "offset", self->tx_streams[tx_stream_index].lo_offset);
        Usrp_set_frequency(self, empty_arg, freq_kws);
    }
    if (self->number_tx_streams > 0) {
        if (!uhd_ok(uhd_usrp_get_tx_stream(*self->usrp_object, &stream_args, *self->tx_streamer))) {
            return -1;
        }

        if (!uhd_ok(uhd_tx_streamer_max_num_samps(*self->tx_streamer, &self->samples_per_buffer))) {
            return -1;
        }
    }

    if (!uhd_ok( uhd_subdev_spec_free(&subdev_spec) )) {
        return -1;
    }

    free(device_args);
    return 0;
}


static PyTypeObject UsrpType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "pysdruhd.Usrp",             /* tp_name */
        sizeof(Usrp), /* tp_basicsize */
        0,                         /* tp_itemsize */
        (destructor) Usrp_dealloc, /* tp_dealloc */
        0,       /* tp_print */
        0,                         /* tp_getattr */
        0,                         /* tp_setattr */
        0,                         /* tp_compare */
        (reprfunc) Usrp_repr,      /* tp_repr */
        0,                         /* tp_as_number */
        0,                         /* tp_as_sequence */
        0,                         /* tp_as_mapping */
        0,                         /* tp_hash */
        0,                         /* tp_call */
        (reprfunc) Usrp_str,       /* tp_str */
        0,                         /* tp_getattro */
        0,                         /* tp_setattro */
        0,                         /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT |
        Py_TPFLAGS_BASETYPE,        /* tp_flags */
        Usrp_docstring,           /* tp_doc */
        0,                         /* tp_traverse */
        0,                         /* tp_clear */
        0,                         /* tp_richcompare */
        0,                         /* tp_weaklistoffset */
        0,                         /* tp_iter */
        0,                         /* tp_iternext */
        Usrp_methods,         /* tp_methods */
        Usrp_members,         /* tp_members */
        0,                         /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        (initproc) Usrp_init,      /* tp_init */
        0,                         /* tp_alloc */
        Usrp_new,                 /* tp_new */
};

#endif //PYSDRUHD_USRP_WRAPPER_H
