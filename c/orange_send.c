#include "orange_send.h"

#include "string.h"

static orange_send_data_t orange_send_data = {0};

/* 设置发包输入：state + deta_x + deta_y */
void orange_send_set_input(int32_t state, int32_t deta_x, int32_t deta_y)
{
    orange_send_data.state = state;
    orange_send_data.deta_x = deta_x;
    orange_send_data.deta_y = deta_y;
}

/* 按协议将当前输入打包为 15 字节固定帧 */
void orange_send_pack_latest(uint8_t tx_buf[ORANGE_SEND_FRAME_SIZE])
{
    if (tx_buf == NULL)
    {
        return;
    }

    tx_buf[0] = ORANGE_SEND_FRAME_HEADER_1;
    tx_buf[1] = ORANGE_SEND_FRAME_HEADER_2;

    memcpy(&tx_buf[2], &orange_send_data.state, sizeof(orange_send_data.state));
    memcpy(&tx_buf[6], &orange_send_data.deta_x, sizeof(orange_send_data.deta_x));
    memcpy(&tx_buf[10], &orange_send_data.deta_y, sizeof(orange_send_data.deta_y));

    tx_buf[14] = ORANGE_SEND_FRAME_TAIL;
}

/* 通过指定串口发送当前最新数据帧 */
HAL_StatusTypeDef orange_send_transmit(UART_HandleTypeDef *huart, uint32_t timeout)
{
    uint8_t tx_buf[ORANGE_SEND_FRAME_SIZE];

    if (huart == NULL)
    {
        return HAL_ERROR;
    }

    orange_send_pack_latest(tx_buf);

    return HAL_UART_Transmit(huart, tx_buf, ORANGE_SEND_FRAME_SIZE, timeout);
}
