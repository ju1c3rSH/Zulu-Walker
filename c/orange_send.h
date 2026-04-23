#ifndef __ORANGE_SEND_H__
#define __ORANGE_SEND_H__

#include "main.h"
#include "stdint.h"

/* 协议固定帧格式：AA BB + state(int32) + deta_x(int32) + deta_y(int32) + EE */
#define ORANGE_SEND_FRAME_HEADER_1   (0xAAu)
#define ORANGE_SEND_FRAME_HEADER_2   (0xBBu)
#define ORANGE_SEND_FRAME_TAIL       (0xEEu)
#define ORANGE_SEND_FRAME_SIZE       (15u)

typedef struct
{
    int32_t state;
    int32_t deta_x;
    int32_t deta_y;
} orange_send_data_t;

void orange_send_set_input(int32_t state, int32_t deta_x, int32_t deta_y);
void orange_send_pack_latest(uint8_t tx_buf[ORANGE_SEND_FRAME_SIZE]);
HAL_StatusTypeDef orange_send_transmit(UART_HandleTypeDef *huart, uint32_t timeout);

#endif
