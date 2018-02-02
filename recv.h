//
// Created by Nathan West on 2/1/18.
//

#ifndef PYSDRUHD_RECV_H
#define PYSDRUHD_RECV_H

#include "usrp_object.h"
#include <uhd.h>
#include <numpy/ndarrayobject.h>

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

    if (!uhd_ok(uhd_rx_streamer_recv(*self->rx_streamer, self->recv_buffers_ptr, self->samples_per_buffer,
                                     self->rx_metadata, 3.0, false, &rx_samples_count))) {
        return NULL;
    }

    if (!uhd_ok(uhd_rx_metadata_time_spec(*self->rx_metadata, &full_secs, &frac_secs))) {
        return NULL;
    }
    PyObject *metadata = PyTuple_New(2);
    PyTuple_SET_ITEM(metadata, 0, PyInt_FromSize_t((size_t) full_secs));
    PyTuple_SET_ITEM(metadata, 1, PyFloat_FromDouble(frac_secs));

    npy_intp shape[2];
    shape[0] = self->number_rx_streams;
    shape[1] = rx_samples_count;

    PyObject *return_val = PyTuple_New(2);
    PyTuple_SET_ITEM(return_val, 0, PyArray_SimpleNewFromData(2, shape, NPY_COMPLEX64, self->recv_buffers));
    PyTuple_SET_ITEM(return_val, 1, metadata);
    return return_val;
}

#endif //PYSDRUHD_RECV_H
