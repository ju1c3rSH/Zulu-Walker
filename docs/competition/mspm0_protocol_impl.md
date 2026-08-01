# MSPM0G3519 — 协议帧实现方案

> 对应 Python 侧 [`protocol.md`](protocol.md)（帧结构）和 [`protocol_content.md`](protocol_content.md)（内容定义）
> MCU: TI MSPM0G3519 (Cortex-M0+ @80MHz)
> 工具链: Keil MDK µVision + SysConfig

---

## 1. UART 资源分配

### 现有占用

| 实例 | Syscfg 别名 | 引脚 | 波特率 | 用途 |
|------|-------------|------|--------|------|
| UART0 | UART_0 | PB1 RX | 921600 | HWT101 陀螺仪 |
| UART1 | UART_1 | PB4 TX, PB5 RX | 921600 | TOF 测距 |
| UART3 | UART_3 | PB2 TX, PB3 RX | 921600 | Emm_V5 步进电机 |
| UART7 | UART_2 | PA23 TX, PA24 RX | 1M | VOFA+ 调试输出 |

### 可用实例

MSPM0G3519 有 7 个 UART 实例（UART2 不存在），以下 3 个空闲：

| 实例 | IRQn | IRQ # | 基地址 | 建议用途 |
|------|------|-------|--------|---------|
| **UART4** | `UART4_INT_IRQn` | 14 | `0x40502000` | OP 通信（推荐，IRQ 优先级冲突最小） |
| UART5 | `UART5_INT_IRQn` | 23 | `0x40504000` | 备选 |
| UART6 | `UART6_INT_IRQn` | 29 | `0x40506000` | 备选 |

### 新 UART 参数

| 参数 | 值 |
|------|-----|
| 实例 | UART4 |
| 波特率 | 921600 |
| 数据位 | 8 |
| 校验 | 无 |
| 停止位 | 1 |
| RX FIFO 阈值 | **4~8 字节**（v3.0 建议） |
| 引脚 | 需确认原理图选择空闲 GPIO |

> ⚠ **v3.0 建议**：921600 波特率下每字节 86.8µs（8N1），推荐 RX FIFO 阈值
> 4~8 字节以减少中断次数。详见 [`master_slave_protocol.md`](./master_slave_protocol.md) §8.5。

---

## 2. 文件组织

```
User/OP_UART/
├── op_uart.h          — 公开接口
└── op_uart.c          — 状态机 + CRC16 + 帧构建/解析 + 回调分发
```

### 影响的其他文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `keil/empty.syscfg` | 修改 | 新增 UART4 实例及引脚配置 |
| `keil/ti_msp_dl_config.c` | 自动生成 | syscfg 触发生成 |
| `keil/ti_msp_dl_config.h` | 自动生成 | syscfg 触发生成 |
| `Control/Interrupt.c` | 修改 | 新增 `UART4_IRQHandler` |
| `empty.c` | 修改 | 调用 `op_uart_init()` + 注册回调 |

---

## 3. CRC16-CCITT 实现

多项式 `0x1021`，初始值 `0xFFFF`，无反射，无最终异或，大端序。
与 Python 侧 `crc16_ccitt()` 完全一致。

查表法（512 字节 ROM 表，~0.2μs/帧 @80MHz）：

```c
static const uint16_t crc16_table[256] = {
    0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50A5, 0x60C6, 0x70E7,
    0x8108, 0x9129, 0xA14A, 0xB16B, 0xC18C, 0xD1AD, 0xE1CE, 0xF1EF,
    /* ... 完整 256 项 */
};

uint16_t op_crc16(const uint8_t *data, uint16_t len) {
    uint16_t crc = 0xFFFF;
    for (uint16_t i = 0; i < len; i++) {
        crc = (crc << 8) ^ crc16_table[((crc >> 8) ^ data[i]) & 0xFF];
    }
    return crc;
}
```

> **注**：MSPM0G3519 有硬件 CRC 外设（支持 CRC16-CCITT），后期可切换以节省 ROM 表空间。

---

## 4. 帧格式（与 Python 侧一致）

```
| SOF1(0xAA) | SOF2(0x55) | Length | Type | Payload | CRC16_HI | CRC16_LO |
|     1      |     1      |   1    |  1   |  0~252  |    1     |    1     |
```

- **Length** = Type(1) + Payload(n) + CRC16(2) = n + 3
- **CRC16** = CRC16-CCITT(Type + Payload)，大端序
- 最小帧 = 6 字节，最大帧 = 258 字节

---

## 5. 解析状态机

```
WAITING_SOF  ──[0xAA]──▶  GOT_SOF1
                             │
                    ┌────────┼──────────┐
                    ▼        ▼          ▼
                 [0x55]   [0xAA]     [其他]
                    │        │          │
                    ▼        ▼          ▼
                 GOT_SOF  重武装     WAITING_SOF
                    │
               [Length]
                    │
                    ▼
                 GOT_LEN
                    │
          [收满 expected_size]
                    │
                    ▼
              CRC16 校验 ──[失败]──▶ WAITING_SOF
                    │
               [成功]
                    │
                    ▼
             查表回调 ──▶ WAITING_SOF
```

```c
typedef enum {
    OP_WAITING_SOF,
    OP_GOT_SOF1,
    OP_GOT_SOF,
    OP_GOT_LEN,
} op_parse_state_t;

typedef struct {
    op_parse_state_t state;
    uint8_t buf[258];
    uint8_t pos;
    uint8_t expected_len;
} op_parser_t;
```

---

## 6. 帧构建

```c
uint16_t op_build_frame(uint8_t type, const uint8_t *payload,
                         uint8_t payload_len, uint8_t *out) {
    uint8_t content_len = 1 + payload_len;            // Type + Payload
    uint8_t total_len   = 1 + payload_len + 2;        // Type + Payload + CRC16

    // CRC16 covers Type + Payload
    uint8_t crc_buf[256];
    crc_buf[0] = type;
    memcpy(&crc_buf[1], payload, payload_len);
    uint16_t crc = op_crc16(crc_buf, content_len);

    out[0] = 0xAA;                                    // SOF1
    out[1] = 0x55;                                    // SOF2
    out[2] = total_len;                                // Length
    out[3] = type;                                     // Type
    memcpy(&out[4], payload, payload_len);              // Payload
    out[4 + payload_len + 0] = (crc >> 8) & 0xFF;      // CRC16_HI
    out[4 + payload_len + 1] = crc & 0xFF;              // CRC16_LO

    return 6 + payload_len;                             // total bytes
}
```

---

## 7. 回调分发

```c
#define OP_MAX_CALLBACKS 16

typedef void (*op_frame_callback_t)(const uint8_t *payload, uint8_t len);

static struct {
    uint8_t type;
    op_frame_callback_t cb;
} op_callbacks[OP_MAX_CALLBACKS];

static uint8_t op_callback_count = 0;

void op_uart_register_callback(uint8_t type, op_frame_callback_t cb) {
    if (op_callback_count < OP_MAX_CALLBACKS) {
        op_callbacks[op_callback_count].type = type;
        op_callbacks[op_callback_count].cb   = cb;
        op_callback_count++;
    }
}

static void op_dispatch(uint8_t type, const uint8_t *payload, uint8_t len) {
    for (uint8_t i = 0; i < op_callback_count; i++) {
        if (op_callbacks[i].type == type) {
            op_callbacks[i].cb(payload, len);
            return;
        }
    }
}
```

---

## 8. 中断集成

```c
// Control/Interrupt.c
#include "op_uart.h"

void UART4_IRQHandler(void) {
    if (DL_UART_Main_getRawInterruptStatus(
            UART_4_INST, UART_MAIN_IIDX_RX)) {
        uint8_t byte = DL_UART_Main_receiveDataBlocking(UART_4_INST);
        op_uart_feed_byte(byte);
    }
}
```

> 注意：SysConfig 中须开启 UART4 RX 中断，中断函数名必须与 startup 文件中向量表名称 `UART4_IRQHandler` 完全一致。

---

## 9. 初始化流程（`empty.c`）

```c
#include "op_uart.h"

// ⚠ v3.0 回调声明（v2.1 旧回调已移除，见 master_slave_protocol.md §8.1）
static void on_cmd_ack(const uint8_t *payload, uint8_t len);
static void on_cmd_nack(const uint8_t *payload, uint8_t len);
static void on_data_stream(const uint8_t *payload, uint8_t len);
static void on_emergency_stop(const uint8_t *payload, uint8_t len);

int main(void) {
    SYSCFG_DL_init();    // 包含 SYSCFG_DL_UART_4_init()

    op_uart_init();      // 初始化解析器

    // 注册 v3.0 回调
    op_uart_register_callback(0x21, on_cmd_ack);          // TYPE_CMD_ACK
    op_uart_register_callback(0x22, on_cmd_nack);         // TYPE_CMD_NACK
    op_uart_register_callback(0x24, on_data_stream);      // TYPE_DATA_STREAM
    op_uart_register_callback(0x18, on_emergency_stop);   // TYPE_EMERGENCY_STOP

    while (1) {
        // 业务循环
    }
}
```

---

## 10. 发送示例（MCU → OP 方向）

```c
// v3.0 发送示例（MSPM0 主机 → MaixCAM2 从机）

// 请求订阅 line position 数据流
uint8_t req_payload[3] = {0x01, 0x00, 0x00};  // data_type, min_interval_ms, reserved
op_uart_send(0x20, req_payload, 3);            // TYPE_CMD_REQUEST

// 停止当前数据流
op_uart_send(0x23, NULL, 0);                   // TYPE_CMD_STOP (空 payload)

// 紧急停止
uint8_t stop_reason = 1;
op_uart_send(0x18, &stop_reason, 1);           // TYPE_EMERGENCY_STOP
```

---

## 11. 帧类型对照表 (v3.0)

| Type | 名称 | Payload | 方向 |
|:---|:---|:---|:---:|
| 0x01 | `TYPE_ERROR` | `error_type(1B) + error_value(2B LE)` | 双向 |
| 0x18 | `TYPE_EMERGENCY_STOP` | `reason(1B)` | 双向 |
| 0x20 | `TYPE_CMD_REQUEST` | `data_type(1B) + min_interval_ms(1B) + reserved(1B)` | MSPM0→Maix |
| 0x21 | `TYPE_CMD_ACK` | `data_type(1B) + max_freq_hz(1B) + payload_size(1B)` | Maix→MSPM0 |
| 0x22 | `TYPE_CMD_NACK` | `data_type(1B) + reason(1B)` | Maix→MSPM0 |
| 0x23 | `TYPE_CMD_STOP` | (空) | MSPM0→Maix |
| 0x24 | `TYPE_DATA_STREAM` | `seq(1B) + data_type(1B) + sub_payload(var)` | Maix→MSPM0 |

| 已废弃 (DEPRECATED) |
| 0x15 | `TYPE_HEARTBEAT` | — | — |
| 0x17 | `TYPE_VISUAL_SERVO_DATA` | — | — |

> 完整 v3.0 定义见 [`master_slave_protocol.md`](./master_slave_protocol.md) §3.1。
> 旧类型 0x02/0x03/0x04/0x10/0x11/0x12/0x13/0x14/0x16 已移除。

---

## 12. 实施顺序

```
Step 1 — 基础设施
  ├ keil/empty.syscfg: 新增 UART4，配置 921600 8N1，选空闲引脚
  └ 编译验证: 自动生成 ti_msp_dl_config.c/h

Step 2 — 新建 OP_UART 模块
  ├ User/OP_UART/op_uart.h — 公开接口声明
  ├ User/OP_UART/op_uart.c — 状态机 + CRC16 + 构建 + 分发
  └ 编译验证

Step 3 — 中断集成
  ├ Control/Interrupt.c: 添加 UART4_IRQHandler
  └ 编译验证

Step 4 — 应用层接线
  ├ empty.c: op_uart_init() + 注册 v3.0 回调（见 §9）
  └ 端到端验证：
       MSPM0 发 TYPE_CMD_REQUEST(0x20) → MaixCAM2 回复 TYPE_CMD_ACK(0x21)
       → MaixCAM2 持续推送 TYPE_DATA_STREAM(0x24) → MSPM0 发 TYPE_CMD_STOP(0x23)

Step 5 — 数据处理对接
  └ 在 on_data_stream 回调中解析各 data_type 子负载（见 master_slave_protocol.md §8.4）
```
