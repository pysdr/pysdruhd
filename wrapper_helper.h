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

#ifndef PYSDRUHD_WRAPPER_HELPER_H
#define PYSDRUHD_WRAPPER_HELPER_H

#include <uhd.h>
#include <stdio.h>
#include <string.h>
#include <Python.h>
#include "usrp_object.h"

#define MIN(x, y) (x) < (y) ? (x) : (y)


stream_mode_t convert_string_to_stream_mode_t(PyObject *string_mode);

void parse_dict_to_streams_config(Usrp *self, PyObject *streams_dict, double frequency_param, double lo_offset,
                                  double rate_param, double gain_param, char *rx_subdev_spec_string);


static inline bool uhd_ok(uhd_error error_value) {
    bool return_value;
    if (error_value == UHD_ERROR_NONE) {
        return_value = true;
    } else {
        char uhd_error_string[8192];
        uhd_get_last_error(uhd_error_string, 8192);
        PyErr_Format(PyExc_Exception, "UHD returned %s", uhd_error_string);
        return_value = false;
    }
    return return_value;
}


#endif //PYSDRUHD_WRAPPER_HELPER_H
