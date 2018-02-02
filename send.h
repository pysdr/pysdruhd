//
// Created by Nathan West on 2/1/18.
//

#ifndef PYSDRUHD_SEND_H
#define PYSDRUHD_SEND_H

#include "usrp_object.h"
#include <uhd.h>

static const char send_docstring[] =
        "Usrp.send(samples, metadata)\n\n"
                "    send a vector of samples to the USRP for transmission";

static PyObject *
Usrp_send(Usrp *self, PyObject *args, PyObject *kwds) {

    static char *kwlist[] = {"samples", "metadata", NULL};
    PyObject *samples = NULL, *metadata = NULL;

    Py_ssize_t args_size = PyTuple_Size(args);
    if (args_size > 0) {
        samples = PyTuple_GetItem(args, 0);
    }
    if (args_size == 2) {
        metadata = PyTuple_GetItem(args, 1);
    }

    int ndims = PyArray_NDIM((PyArrayObject *) samples);
    size_t samples_per_buffer = 0;
    int number_channels = 0;
    npy_intp *samples_shape;
    samples_shape = PyArray_SHAPE((PyArrayObject *) samples);
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
    const void *samples_ptr = PyArray_DATA((PyArrayObject *) samples);

    size_t items_sent;
    uhd_tx_streamer_send(*self->tx_streamer, &samples_ptr, samples_per_buffer, &tx_metadata, timeout, &items_sent);
    uhd_tx_metadata_free(&tx_metadata);

    Py_RETURN_NONE;
}

#endif //PYSDRUHD_SEND_H
