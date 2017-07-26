//
// Created by nathan on 7/26/17.
//

#ifndef PYSDRUHD_USRP_OBJECT_H
#define PYSDRUHD_USRP_OBJECT_H

#include <Python.h>
#include <uhd.h>


typedef enum {
    OFF,
    TX_STREAM,
    RX_STREAM,
    TX_BURST,
    RX_BURST,
} stream_mode_t;


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
#endif //PYSDRUHD_USRP_OBJECT_H
