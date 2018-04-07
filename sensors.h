//
// Created by Nathan West on 1/30/18.
//

#ifndef PYSDRUHD_SENSORS_H
#define PYSDRUHD_SENSORS_H

static const char sensor_names_docstring[] =
        "names = sensor_names()\n\n"
                "    Returns a list of strings containing all of the names of the sensors on a USRP as reported by UHD.";
static const char get_sensor_docstring[] =
        "value = get_sensor(sensorname)\n\n"
                "   returns the value of a sensor with name matching the string sensorname. The datatype of a sensor value is "
                "dependent on the sensor";

static struct _object *
Usrp_sensor_names(Usrp *self) {
    uhd_string_vector_t *sensor_names;
    uhd_string_vector_make(&sensor_names);
    if (!uhd_ok(uhd_usrp_get_mboard_sensor_names(*self->usrp_object, 0, &sensor_names) )) {
        return NULL;
    }
    size_t number_sensors;
    if (!uhd_ok(uhd_string_vector_size(sensor_names, &number_sensors) )) {
        return NULL;
    }
    struct _object *sensor_names_list = PyList_New(0);

    for (unsigned int ii = 0; ii < number_sensors; ++ii) {
        char sensor_name[64];
        if (!uhd_ok(uhd_string_vector_at(sensor_names, ii, sensor_name, 64) )) {
            return NULL;
        }
        PyList_Append(sensor_names_list, PyString_FromString(sensor_name));
    }
    if (!uhd_ok(uhd_string_vector_free(&sensor_names))) {
        return NULL;
    }
    return sensor_names_list;
}

static struct _object *
Usrp_get_sensor(Usrp *self, struct _object *args, struct _object *kwds) {

    static char *kwlist[] = {"sensor", NULL};
    struct _object *sensor_string = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist,
                                     &sensor_string, &sensor_string
    )) {
        return NULL;
    }

    struct uhd_sensor_value_t *sensor_value;
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

#include "usrp_object.h"

#endif //PYSDRUHD_SENSORS_H
