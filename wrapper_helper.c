//
// Created by nathan on 7/26/17.
//

#include "wrapper_helper.h"
#include "usrp_object.h"

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

void parse_dict_to_streams_config(Usrp *self, PyObject *streams_dict, double frequency_param, double lo_offset_param,
                                  double rate_param, double gain_param, char *rx_subdev_spec_string) {
    PyObject *subdev, *config;
    Py_ssize_t position = 0;
    while (PyDict_Next(streams_dict, &position, &subdev, &config)) {
        stream_config_t this_subdev;
        PyObject *value;

        strncpy(this_subdev.subdev, PyString_AsString(subdev), 10);
        const char mode_key[] = "mode";
        value = PyDict_GetItemString(config, mode_key);
        this_subdev.mode = convert_string_to_stream_mode_t(value);
        this_subdev.frequency = frequency_param;
        this_subdev.lo_offset = lo_offset_param;
        this_subdev.rate = rate_param;
        this_subdev.gain = gain_param;

        value = PyDict_GetItemString(config, "antenna\0");
        if (value != NULL) {
            strncpy(this_subdev.antenna, PyString_AsString(value), 6);
        } else {
            PyErr_Format(PyExc_TypeError, "Uhd dict parameter requires each subdev to have an 'antenna' set; %s did not", this_subdev.subdev);
        }
        value = PyDict_GetItemString(config, "frequency\0");
        if (value != NULL) {
            this_subdev.frequency = PyFloat_AsDouble(value);
        }
        value = PyDict_GetItemString(config, "lo_offset\0");
        if (value != NULL) {
            this_subdev.lo_offset = PyFloat_AsDouble(value);
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
            strncat(rx_subdev_spec_string, this_subdev.subdev, 6);
            strncat(rx_subdev_spec_string, " \0", 1);
            /* DEBUG */ puts("got a rx stream\n");
            /* DEBUG */ printf("{'%s': 'frequency':%1.2e, 'rate'=%1.2e, 'gain'=%1.2e}\n", this_subdev.subdev,
                               this_subdev.frequency, this_subdev.rate, this_subdev.gain);
            self->rx_streams = realloc(self->rx_streams, sizeof(stream_config_t) * (self->number_rx_streams + 1));
            memcpy(self->rx_streams + self->number_rx_streams, &this_subdev, sizeof(stream_config_t));
            self->number_rx_streams++;
        } else if (this_subdev.mode == TX_STREAM) {
            /* DEBUG */ puts("got a tx stream\n");
            /* DEBUG */ printf("{'%s': 'frequency':%1.2e, 'rate'=%1.2e, 'gain'=%1.2e}\n", this_subdev.subdev,
                               this_subdev.frequency, this_subdev.rate, this_subdev.gain);
            self->tx_streams = realloc(self->tx_streams, sizeof(stream_config_t) * (self->number_tx_streams + 1));
            memcpy(self->tx_streams + self->number_tx_streams, &this_subdev, sizeof(stream_config_t));

            self->number_tx_streams++;
        }
    } /* Parsing provided config dict */
}