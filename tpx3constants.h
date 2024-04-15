//
// Header file for some constants 
//

#ifndef TPX3CONSTANTS_H
#define TPX3CONSTANTS_H

namespace tpx3{

    // tpx3 chip-related numbers 
    const int NUM_RX_CHAN = 8;
    
    // register names
    const std::string DATA_DELAY = "DATA_DELAY";
    const std::string FIFO = "FIFO";
    const std::string FIFO_SIZE = "FIFO_SIZE";

    // python fifo_readout function names
    const char* CNTR_DECODING_ERR = "get_rx_decode_error_count";
    const char* CNTR_DISCARD_ERR = "get_rx_fifo_discard_count";
    const char* FIFO_STATUS_ENA = "get_rx_en_status";
    const char* FIFO_STATUS_SYNC = "get_rx_sync_status";

};

#endif
