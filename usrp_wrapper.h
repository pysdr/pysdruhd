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
#include "send.h"
#include "usrp_time.h"
#include "recv.h"
#include "rf_settings.h"
#include "usrp_stream_command.h"

#include <uhd.h>
#include <stdio.h>
#include <string.h>
#include <structmember.h>
#include <sys/time.h>
#include <numpy/ndarrayobject.h>


static PyMemberDef Usrp_members[] = {
        {NULL}  /* Sentinel */
};


static PyMethodDef Usrp_methods[] = {
        {"recv",                  (PyCFunction) Usrp_recv,                  METH_NOARGS,  recv_docstring},
        {"send",                  (PyCFunction) Usrp_send,                  METH_VARARGS |
                                                                            METH_KEYWORDS, send_docstring},
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
        {"get_time_sources",      (PyCFunction) Usrp_get_time_sources,      METH_NOARGS,  get_time_sources_docstring},
        {"get_time_source",       (PyCFunction) Usrp_get_time_source,       METH_NOARGS,  get_time_source_docstring},
        {"set_time_source",       (PyCFunction) Usrp_set_time_source,       METH_VARARGS, set_time_source_docstring},
        {"get_clock_sources",     (PyCFunction) Usrp_get_clock_sources,     METH_NOARGS,  get_clock_sources_docstring},
        {"get_clock_source",      (PyCFunction) Usrp_get_clock_source,      METH_NOARGS,  get_clock_source_docstring},
        {"set_clock_source",      (PyCFunction) Usrp_set_clock_source,      METH_VARARGS, set_clock_source_docstring},
        {NULL}  /* Sentinel */
};


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
    for (unsigned int rx_stream_indx = 0; rx_stream_indx < self->number_rx_streams; ++rx_stream_indx) {
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
    for (unsigned int tx_stream_indx = 0; tx_stream_indx < self->number_tx_streams; ++tx_stream_indx) {
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
                                            channel[rx_stream_index]))) {
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
                                            channel[tx_stream_index]))) {
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

    if (!uhd_ok(uhd_subdev_spec_free(&subdev_spec))) {
        return -1;
    }

    free(device_args);
    return 0;
}


static PyTypeObject UsrpType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "pysdruhd.Usrp",           /* tp_name */
        sizeof(Usrp),              /* tp_basicsize */
        0,                         /* tp_itemsize */
        (destructor) Usrp_dealloc, /* tp_dealloc */
        0,                         /* tp_print */
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
        Py_TPFLAGS_BASETYPE,       /* tp_flags */
        Usrp_docstring,            /* tp_doc */
        0,                         /* tp_traverse */
        0,                         /* tp_clear */
        0,                         /* tp_richcompare */
        0,                         /* tp_weaklistoffset */
        0,                         /* tp_iter */
        0,                         /* tp_iternext */
        Usrp_methods,              /* tp_methods */
        Usrp_members,              /* tp_members */
        0,                         /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        (initproc) Usrp_init,      /* tp_init */
        0,                         /* tp_alloc */
        Usrp_new,                  /* tp_new */
};

#endif //PYSDRUHD_USRP_WRAPPER_H
