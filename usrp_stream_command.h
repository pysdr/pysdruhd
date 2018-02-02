//
// Created by Nathan West on 2/1/18.
//

#ifndef PYSDRUHD_USRP_STREAM_COMMAND_H
#define PYSDRUHD_USRP_STREAM_COMMAND_H

#include "usrp_object.h"
#include <uhd.h>


static const char send_stream_command_docstring[] =
        "send_stream_command(mode='start', when='now', nsamples)\n\n"
                "    send_stream_command will create and issue a stream command to UHD. UHD stream commands send an rx or tx "
                "streamer a command that contains a `stream_mode` enum, `stream_now` bool, and optionall a timespec. This wrapper "
                "accepts the mode as a string and when can either be a string matching 'now' or a tuple of (full secs, fractional "
                "secs). By default this puts us in a continuous streaming mode now. nsamples will only have meaning for 'chunk' mode\n\n"
                "Examples:\n"
                "  send_stream_command(mode='start')\n"
                "  send_stream_command(mode='stop', when='now')\n"
                "  send_stream_command(mode='chunk', nsamples=4096)\n"
                "  send_stream_command(mode='start', when=(10,0.0)\n\n"
                "N.B. from UHD docs: to use an on-board GPSDO use set_time_source('external') and set_clock_source('external')";

static PyObject *
Usrp_send_stream_command(Usrp *self, PyObject *args, PyObject *kwds) {
    char *mode_str = NULL;
    PyObject *when = NULL;
    int nsamples = 0;

    static char *kwlist[] = {"mode", "when", "nsamples", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|SOi", kwlist,
                                     &mode_str, &when, &nsamples
    )) {
        return NULL;
    }

    // The default with no options will just be to start streaming continuous mode now
    uhd_stream_cmd_t stream_cmd = {
            .stream_mode = UHD_STREAM_MODE_START_CONTINUOUS,
            .stream_now = true
            //.time_spec_full_secs = 1,
            //.time_spec_frac_secs= 0.,
    };
    if (strncmp(mode_str, "stop", 4) == 0) {
        stream_cmd.stream_mode = UHD_STREAM_MODE_STOP_CONTINUOUS;
    } else if (strncmp(mode_str, "start", 5) == 0) {
        stream_cmd.stream_mode = UHD_STREAM_MODE_START_CONTINUOUS;
    } else if (strncmp(mode_str, "chunk", 5) == 0) {
        stream_cmd.stream_mode = UHD_STREAM_MODE_NUM_SAMPS_AND_DONE;
    }

    if (PyString_Check(when)) {
        char *when_str = PyString_AsString(when);
        if (strncmp(when_str, "now", 3)) {
            stream_cmd.stream_now = true;
        }
    } else if (PyTuple_Check(when)) {
        if (PyTuple_Size(when) == 2) {
            PyObject *fullSec = PyTuple_GetItem(when, 0);
            PyObject *fracSec = PyTuple_GetItem(when, 1);
            stream_cmd.time_spec_full_secs = PyInt_AsLong(fullSec);
            stream_cmd.time_spec_frac_secs = PyFloat_AsDouble(fracSec);
        } else {
            PyErr_SetString(PyExc_TypeError, "when argument requires a 2-tuple as the time-spec\n");
        }
    }
    if (!uhd_ok(uhd_rx_streamer_issue_stream_cmd(*self->rx_streamer, &stream_cmd))) {
        return NULL;
    }

    return Py_None;
}

#endif //PYSDRUHD_USRP_STREAM_COMMAND_H
