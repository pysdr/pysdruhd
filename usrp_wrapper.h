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


#include <uhd.h>
#include <stdio.h>
#include <string.h>
#include <structmember.h>

#define MIN(x,y) x < y ? x : y

typedef enum {
    OFF,
    TX_STREAM,
    RX_STREAM,
    TX_BURST,
    RX_BURST,
} stream_mode_t;

stream_mode_t convert_string_to_stream_mode_t(PyObject *string_mode) {
    stream_mode_t mode = RX_STREAM; /* default to RX streaming */
    puts("made it to convert\n");
    fflush(stdout);
    if (string_mode != NULL && PyString_CheckExact(string_mode)) {
        size_t compare_length = (size_t) PyString_Size(string_mode);
        if (strncmp("TX\0", PyString_AsString(string_mode), MIN(compare_length, 2)) == 0) {
            mode = TX_STREAM;
        } else if (strncmp("RX\0", PyString_AsString(string_mode), MIN(compare_length, 2)) == 0) {
            mode = RX_STREAM;
        } else if (strncmp("OFF\0", PyString_AsString(string_mode), MIN(compare_length, 3)) == 0) {
            mode = OFF;
        }
    } else {
        /* mode is not one of our known modes. TODO: Panic!!!!! */
    }
    return mode;
}

typedef struct {
    stream_mode_t mode;
    double frequency;
    double rate;
    double gain;
    char subdev[6]; /* such as A:0, A:1, B:0, B:1 */
    char identifier[10]; /* such as TX/RX, RX2, RX1.... (would be nice to have a card type identifier) */
} stream_config_t;

typedef struct {
    PyObject_HEAD
    /* Type-specific fields go here. */
    PyObject *addr;
    PyObject *usrp_type;
    uhd_usrp_handle *usrp_object;
    uhd_rx_streamer_handle *rx_streamer;
    uhd_rx_metadata_handle *rx_metadata;
    stream_config_t *rx_streams;
    int number_rx_streams;

    // None of the TX stuff is implemented yet (one thing at a time)
    uhd_tx_streamer_handle *tx_streamer;
    //uhd_tx_metadata_handle *tx_metadata;
    stream_config_t *tx_streams;
    size_t number_tx_streams;

    size_t samples_per_buffer;
    void *recv_buffers;
    void **recv_buffers_ptr;

} Usrp;

static void
Usrp_dealloc(Usrp *self) {
    printf("deallocing usrp\n");
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
    /* DEBUG */ printf("new()\n");
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

        // TODO: handle the errors
        uhd_error uhd_errno;
        uhd_errno = uhd_rx_streamer_make(self->rx_streamer);
        uhd_errno = uhd_rx_metadata_make(self->rx_metadata);
        uhd_errno = uhd_tx_streamer_make(self->tx_streamer);
        //uhd_errno = uhd_tx_metadata_make(self->tx_metadata);
    } else {
        /*
         * I'm not sure how this would happen, but something is horribly wrong.
         * The python Noddy example does this check though...
         */
        puts("Usrp.__new__ failed\n");
    }

    return (PyObject *) self;
}

static int
Usrp_init(Usrp *self, PyObject *args, PyObject *kwds)
{
    PyObject *addr = NULL, *addr2 = NULL, *tmp = NULL;
    PyObject *usrp_type = NULL;
    PyObject *streams_dict = NULL;
    double frequency_param = 910e6;
    double rate_param = 1e6;
    double gain_param = 0;

    static char *kwlist[] = {"addr", "addr2", "type", "streams", "frequency", "rate", "gain", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOddd", kwlist,
                                     &addr, &addr2, &usrp_type, &streams_dict, &frequency_param, &rate_param, &gain_param
    )) {
        return -1;
    }

    uhd_error uhd_errno;
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

    printf("Opening USRP with args: \"%s\"\n", device_args);
    fflush(stdout);

    uhd_errno = uhd_usrp_make(self->usrp_object, device_args);
    size_t nmboards;
    uhd_usrp_get_num_mboards(*self->usrp_object, &nmboards);
    printf("%lu mboards present\n", nmboards);

    // Ideally interrogate the device to create an internal dict of channels/subdevs
    char usrp_pp[2048];
    uhd_usrp_get_pp_string(*self->usrp_object, usrp_pp, 2048);
    /* DEV HELPER */ printf("usrp: %s\n", usrp_pp);

    size_t channel[] = {0, 1, 2, 3};
    char rx_subdev_spec_string[64] = {'\0'};

    /*
     * We will copy the dict in to our internal struct and fill in missing values with defaults
     * If there is no dict then we use the parameter list and fill in missing values with defaults.
     * How friendly of me...
     */
    if (streams_dict) {
        /* DEV HELPER */ printf("checking streams\n");
        if (PyDict_Check(streams_dict)) {
            PyObject *subdev, *config;
            Py_ssize_t position = 0;
            while (PyDict_Next(streams_dict, &position, &subdev, &config)) {
                stream_config_t this_subdev;
                PyObject *value;

                strncpy(this_subdev.subdev, PyString_AsString(subdev), 6);
                const char mode_key[] = "mode";
                value = PyDict_GetItemString(config, mode_key);
                this_subdev.mode = convert_string_to_stream_mode_t(value);
                this_subdev.frequency = frequency_param;
                this_subdev.rate = rate_param;
                this_subdev.gain = gain_param;

                value = PyDict_GetItemString(config, "frequency\0");
                if (value != NULL) {
                    this_subdev.frequency = PyFloat_AsDouble(value);
                }
                value = PyDict_GetItemString(config, "rate\0");
                if (value != NULL) {
                    this_subdev.rate = PyFloat_AsDouble(value);
                }
                value = PyDict_GetItemString(config, "gain\0");
                if (value != NULL) {
                    this_subdev.gain = PyFloat_AsDouble(value);
                }

                if (this_subdev.mode == RX_STREAM) {
                    strncat(rx_subdev_spec_string, this_subdev.subdev, 3);
                    strncat(rx_subdev_spec_string, " \0", 1);
                    /* DEBUG */ puts("got a rx stream\n");
                    /* DEBUG */ printf("{'%s': 'frequency':%1.2e, 'rate'=%1.2e, 'gain'=%1.2e}\n", this_subdev.subdev, this_subdev.frequency, this_subdev.rate, this_subdev.gain);
                    self->rx_streams = realloc(self->rx_streams, sizeof(stream_config_t) * (self->number_rx_streams+1));
                    memcpy(self->rx_streams + self->number_rx_streams, &this_subdev, sizeof(stream_config_t));
                    self->number_rx_streams++;
                } else if (this_subdev.mode == TX_STREAM) {
                    /* DEBUG */ puts("got a tx stream\n");
                    /* DEBUG */ printf("{'%s': 'frequency':%1.2e, 'rate'=%1.2e, 'gain'=%1.2e}\n", this_subdev.subdev, this_subdev.frequency, this_subdev.rate, this_subdev.gain);
                    self->tx_streams = realloc(self->tx_streams, sizeof(stream_config_t) * (self->number_tx_streams+1));
                    memcpy(self->tx_streams + self->number_tx_streams, &this_subdev, sizeof(stream_config_t));

                    self->number_tx_streams++;
                }
            } /* Parsing provided config dict */
        } else {
            puts ("streams argument needs to be a dict of form"
                            "    {'<DB>:<SUBDEV>': {'mode': 'RX'|'TX', 'frequency': double, 'rate': double, 'gain': double},\n");
        }
    } else {
        // We didn't get a config dict, so default to create 1 RX stream
        self->rx_streams = malloc(sizeof(stream_config_t));
        self->number_rx_streams = 1;
        self->rx_streams[0].frequency = frequency_param;
        self->rx_streams[0].rate = rate_param;
        self->rx_streams[0].gain = gain_param;
        strncpy(self->rx_streams[0].subdev, "A:0\0", 4);
    }


    // We should accept a stream_args dict
    uhd_stream_args_t stream_args = {
            .cpu_format = "fc32",
            .otw_format = "sc16",
            .args = "",
            .channel_list = channel,
            .n_channels = self->number_rx_streams
    };
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
    double checkval;


    char antennas[8][4] = {"RX1", "RX2", "RX1", "RX2"}; // This unfortunately needs to be in the dict. It could be different for TWINRX, basicrx, and a every other card
    uhd_subdev_spec_handle subdev_spec;
    printf("subdev spec string: %s\n", rx_subdev_spec_string);
    uhd_subdev_spec_make(&subdev_spec, rx_subdev_spec_string);
    uhd_usrp_set_rx_subdev_spec(*self->usrp_object, subdev_spec, 0);

    for (size_t rx_stream_index=0; rx_stream_index < self->number_rx_streams; ++rx_stream_index) {
        printf("setting up rx stream %lu\n", rx_stream_index);
        uhd_usrp_set_rx_antenna(*self->usrp_object, antennas[rx_stream_index], channel[rx_stream_index]);
        uhd_errno = uhd_usrp_set_rx_rate(*self->usrp_object, self->rx_streams[rx_stream_index].rate, channel[rx_stream_index]);
        uhd_errno = uhd_usrp_get_rx_rate(*self->usrp_object, channel[rx_stream_index], &checkval);
        rate_param = checkval;

        uhd_errno = uhd_usrp_set_rx_gain(*self->usrp_object, self->rx_streams[rx_stream_index].gain, channel[rx_stream_index], "");
        uhd_errno = uhd_usrp_get_rx_gain(*self->usrp_object, channel[rx_stream_index], "", &checkval);
        gain_param = checkval;

        // definitely want to support offset tuning....
        uhd_tune_request_t tune_request = {
                .target_freq = self->rx_streams[rx_stream_index].frequency,
                .rf_freq_policy = UHD_TUNE_REQUEST_POLICY_AUTO,
                .dsp_freq_policy = UHD_TUNE_REQUEST_POLICY_AUTO,
        };
        uhd_tune_result_t tune_result;
        uhd_errno = uhd_usrp_set_rx_freq(*self->usrp_object, &tune_request, channel[rx_stream_index], &tune_result);
        uhd_errno = uhd_usrp_get_rx_freq(*self->usrp_object, channel[rx_stream_index], &checkval);
        frequency_param = tune_result.actual_rf_freq;
    }
    if (self->number_rx_streams > 0) {
        puts("make the stremer object\n");
        uhd_errno = uhd_usrp_get_rx_stream(*self->usrp_object, &stream_args, *self->rx_streamer);
        if (uhd_errno != UHD_ERROR_NONE) {
            puts("we got an error from uhd_usrp_get_rx_stream\n");
            char dbg_buf[1024];
            uhd_get_last_error(dbg_buf, 1024);
            puts(dbg_buf);
        }

        uhd_errno = uhd_rx_streamer_max_num_samps(*self->rx_streamer, &self->samples_per_buffer);
        printf("max samples per buffer is %lu\n", self->samples_per_buffer);
        size_t buffer_size_per_channel = self->samples_per_buffer * 2 * sizeof(float);
        self->recv_buffers = malloc(self->number_rx_streams * buffer_size_per_channel);
        self->recv_buffers_ptr = malloc(sizeof(void *) * self->number_rx_streams);
        // watch out world, we're indexing void* 's!
        for (size_t rx_stream_index = 0; rx_stream_index < self->number_rx_streams; ++rx_stream_index) {
            self->recv_buffers_ptr[rx_stream_index] = self->recv_buffers + (rx_stream_index * buffer_size_per_channel);
            printf("buffer for channel %lu is %p\n", rx_stream_index, self->recv_buffers_ptr[rx_stream_index]);
        }
    }
    uhd_subdev_spec_free(&subdev_spec);


    for (size_t tx_stream_index=0; tx_stream_index < self->number_tx_streams; ++tx_stream_index) {
        // TODO: set up tx streams
    }
    puts("done initing\n");
    fflush(stdout);

    free(device_args);
    if (uhd_errno == UHD_ERROR_NONE) {
        return 0;
    } else {
        return -1; // TODO: properly return a python error with the UHD error string
    }
}


static PyMemberDef Usrp_members[] = {
        {NULL}  /* Sentinel */
};


static PyObject *
Usrp_recv(Usrp *self)
{
    size_t rx_samples_count = 0;
    time_t full_secs = 0;
    double frac_secs = 0.;
    uhd_error uhd_errno;

    uhd_errno = uhd_rx_streamer_recv(*self->rx_streamer, self->recv_buffers_ptr, self->samples_per_buffer,
                                               self->rx_metadata, 3.0, false, &rx_samples_count);

    uhd_errno = uhd_rx_metadata_time_spec(*self->rx_metadata, &full_secs, &frac_secs);
    PyObject *metadata = PyTuple_New(2);
    PyTuple_SET_ITEM(metadata, 0, PyInt_FromSize_t(full_secs));
    PyTuple_SET_ITEM(metadata, 1, PyFloat_FromDouble(frac_secs));

    npy_intp shape[2];
    shape[0] = self->number_rx_streams;
    shape[1] = rx_samples_count;

    PyObject *return_val = PyTuple_New(2);
    PyTuple_SET_ITEM(return_val, 0, PyArray_SimpleNewFromData(2, shape, NPY_COMPLEX64, self->recv_buffers));
    PyTuple_SET_ITEM(return_val, 1, metadata);
    return return_val;
}

static PyObject *
Usrp_sensor_names(Usrp *self)
{
    uhd_error uhd_errno;
    uhd_string_vector_handle sensor_names;
    uhd_string_vector_make(&sensor_names);
    uhd_errno = uhd_usrp_get_mboard_sensor_names(*self->usrp_object, 0, &sensor_names);
    size_t number_sensors;
    uhd_string_vector_size(sensor_names, &number_sensors);
    PyObject *sensor_names_list = PyList_New(0);

    for (unsigned int ii=0; ii < number_sensors; ++ii) {
        char sensor_name[64];
        uhd_errno = uhd_string_vector_at(sensor_names, ii, sensor_name, 64);
        PyList_Append(sensor_names_list, PyString_FromString(sensor_name));
    }
    uhd_errno = uhd_string_vector_free(&sensor_names);
    return sensor_names_list;
}

static PyObject *
Usrp_get_sensor(Usrp *self, PyObject *args, PyObject *kwds)
{

    static char *kwlist[] = {"sensor", NULL};
    PyObject *sensor_string = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist,
                                     &sensor_string, &sensor_string
    )) {
        return NULL;
    }
    uhd_error uhd_errno;
    uhd_sensor_value_handle sensor_value;
    uhd_sensor_value_make_from_string(&sensor_value, "w", "t", "f");
    uhd_usrp_get_mboard_sensor(*self->usrp_object, PyString_AsString(sensor_string), 0, &sensor_value);
    uhd_sensor_value_data_type_t sensor_dtype;
    uhd_sensor_value_data_type(sensor_value, &sensor_dtype);
    PyObject *return_sensor_value;
    bool boolval;
    int intval;
    double doubleval;
    char charval[4096];

    switch (sensor_dtype) {
        case UHD_SENSOR_VALUE_BOOLEAN:
            uhd_sensor_value_to_bool(sensor_value, &boolval);
            return_sensor_value = PyBool_FromLong(boolval);
            break;
        case UHD_SENSOR_VALUE_INTEGER:
            uhd_sensor_value_to_int(sensor_value, &intval);
            return_sensor_value = PyInt_FromLong(intval);
            break;
        case UHD_SENSOR_VALUE_REALNUM:
            uhd_sensor_value_to_realnum(sensor_value, &doubleval);
            return_sensor_value = PyFloat_FromDouble(doubleval);
            break;
        case UHD_SENSOR_VALUE_STRING:
            uhd_sensor_value_to_pp_string(sensor_value, charval, 4096);
            return_sensor_value = PyString_FromString(charval);
            break;
        default:
            return_sensor_value = Py_None;
    }

    return return_sensor_value;
}

static PyObject *
Usrp_set_master_clock_rate(Usrp *self, PyObject *args)
{
    double clock_rate;
    if (!PyArg_ParseTuple(args, "d",
                          &clock_rate)) {
        return NULL;
    }

    uhd_usrp_set_master_clock_rate(*self->usrp_object, clock_rate, 0);
    uhd_usrp_get_master_clock_rate(*self->usrp_object, 0, &clock_rate);
    return PyFloat_FromDouble(clock_rate);
}

static PyObject *
Usrp_set_time(Usrp *self, PyObject *args, PyObject *kwds)
{
    PyObject *when=NULL, *time=NULL;
    static char *kwlist[] = {"time", "when", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", kwlist,
                                     &time, &when
    )) {
        return NULL;
    }

    int full_secs = 0;
    double fractional_secs = 0.0;

    if (time != NULL) {
        if (PyString_Check(time)) { /* if we got a string it should say gps */
            if (strncmp(PyString_AsString(time), "gps", MIN((size_t) PyString_Size(time), 3))) {
                // we want to set the time next pps to whatever the gpsdo is
                uhd_sensor_value_handle sensor_value;
                uhd_sensor_value_make_from_string(&sensor_value, "w", "t", "f");
                uhd_error uhd_errno = uhd_usrp_get_mboard_sensor(*self->usrp_object, "gps_time", 0, &sensor_value);
                if (uhd_errno == UHD_ERROR_NONE) {
                    uhd_sensor_value_data_type_t sensor_dtype;
                    uhd_sensor_value_data_type(sensor_value, &sensor_dtype);
                    full_secs = uhd_sensor_value_to_int(sensor_value, &full_secs);
                }

            }
        }
        else if (PyTuple_Check(time) && PyTuple_Size(time)==2) { /* if we got a tuple, then it's whole secs, fractional secs. a very sexy time */
            full_secs = (int) PyInt_AsLong(PyTuple_GetItem(time, 0));
            fractional_secs = PyInt_AsLong(PyTuple_GetItem(time, 1));
        }
    }

    if (when != NULL && PyString_Check(when)) {
        char *data = PyString_AsString(when);
        if (strncmp(data, "now", 3) == 0) {
            uhd_usrp_set_time_now(*self->usrp_object, full_secs, fractional_secs, 0);
        } else {
            uhd_usrp_set_time_next_pps(*self->usrp_object, full_secs, fractional_secs, 0);
        }
    } else { // Default to next pps
        uhd_usrp_set_time_next_pps(*self->usrp_object, full_secs, fractional_secs, 0);
    }

    return Py_None;
}


static PyObject *
Usrp_send_stream_command(Usrp *self, PyObject *args, PyObject *kwds)
{
    PyObject *command;

    if (!PyArg_ParseTuple(args, "O", &command)) {
        return NULL;
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
    uhd_usrp_set_time_source(*self->usrp_object, "internal", 0);
    uhd_usrp_set_time_now(*self->usrp_object, 0, 0.0, 0);
    uhd_error uhd_errno = uhd_rx_streamer_issue_stream_cmd(*self->rx_streamer, &stream_cmd);

    return Py_None;
}


static PyMethodDef Usrp_methods[] = {
        {"recv", (PyCFunction) Usrp_recv, METH_NOARGS,
                "samples, metadata = Usrp.recv() will return an ndarray of shape (nchannels, nsamples) where nchannels\
 the number of subdevs specified during construction and nsamples is the number of samples in the packet returned by UHD"},
        {"sensor_names", (PyCFunction) Usrp_sensor_names, METH_NOARGS,
                "print the sensor names"},
        {"get_sensor", (PyCFunction) Usrp_get_sensor, METH_VARARGS|METH_KEYWORDS,
                "get the value of a sensor"},
        {"set_master_clock_rate", (PyCFunction) Usrp_set_master_clock_rate, METH_VARARGS,
                "set the master clock rate. Usually has some impact on ADC/DAC rate."},
        {"set_time", (PyCFunction) Usrp_set_time, METH_VARARGS|METH_KEYWORDS,
                "set_time(when='pps', time=0).\n\n"
                        "`time` is either a tuple of (full seconds, fractional seconds) or 'gps'"
                        "`when` should be a string matching either 'now' or 'pps'"},
        {"send_stream_command", (PyCFunction) Usrp_send_stream_command, METH_VARARGS|METH_KEYWORDS,
                "send_stream_command(command={'mode':'continuous', 'now':true,}) Accepts a dict as a stream command and sends that to the USRP."},
        {NULL}  /* Sentinel */
};

static const char Usrp_docstring[] =
{"A friendly native python interface to USRPs. The constructor looks like this (all optional arguments): \n\
\n\
        Usrp(addr, type, streams, frequency, rate, gain) \n\
            addr: a string with the address of a network connected USRP \n\
            type: a string with the type of USRP (find with uhd_find_devices \n\
            streams: a dictionary of the form {'subdev': {'frequency': <double>, 'rate': <double>, 'gain': <double>}, } \n\
            The keys within a subdev are optional and will take default values of the frequency, rate, and gain parameters:\n\
            frequency: <double> the center frequency to tune to \n\
            rate: <double> the requested sample rate \n\
            gain: <double> the requested gain \n\
\n\
        The primary function of a Usrp is to transmit and receive samples. These are handled through \n\
        samples, metadata = Usrp.recv() \n\
        Usrp.transmit(samples) \n\
\n\
        In both cases samples is a numpy array with shape (nchannels, nsamps) \n\
\n\
    There are several currently unsupported USRP features that are of interest: \n\
        * burst modes \n\
        * offset tuning (relatively easy to impl) \n\
        * settings transport args like buffer size and type conversions\
\n\
    I'm also interested in implementing some of the as_sequence, etc methods so we can do things like \n\
    reduce(frame_handler, filter(burst_detector, map(signal_processing, Usrp()))) \n\
\n\
Until then, this is missing some more advances features and is crash-prone when things aren't butterflies and \n\
rainbows, but is at least capable of streaming 200 Msps in to python with no overhead. \n\
        "};

static PyTypeObject UsrpType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "pysdruhd.Usrp",             /* tp_name */
        sizeof(Usrp), /* tp_basicsize */
        0,                         /* tp_itemsize */
        (destructor) Usrp_dealloc,         /* tp_dealloc */
        0,                         /* tp_print */
        0,                         /* tp_getattr */
        0,                         /* tp_setattr */
        0,                         /* tp_compare */
        0,                         /* tp_repr */
        0,                         /* tp_as_number */
        0,                         /* tp_as_sequence */
        0,                         /* tp_as_mapping */
        0,                         /* tp_hash */
        0,                         /* tp_call */
        0,                         /* tp_str */
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
