# MSPM0G3519 — 协议帧实现方案

> 对应 Python 侧 [`protocol.md`](protocol.md)（帧结构）和 [`protocol_content.md`](protocol_content.md)（内容定义）
> MCU: TI MSPM0G3519 (Cortex-M0+ @80MHz)
> 工具链: Keil MDK µVision + SysConfig

---

## 1. UART 资源分配

### 现有占用

| 实例 | Syscfg 别名 | 引脚 | 波特率 | 用途 |
|------|-------------|------|--------|------|
| UART0 | UART_0 | PB1 RX | 115200 | HWT101 陀螺仪 |
| UART1 | UART_1 | PB4 TX, PB5 RX | 115200 | TOF 测距 |
| UART3 | UART_3 | PB2 TX, PB3 RX | 115200 | Emm_V5 步进电机 |
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
| RX FIFO 阈值 | 1 字节 |
| 引脚 | 需确认原理图选择空闲 GPIO |

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

// 回调函数声明
static void on_heartbeat(const uint8_t *payload, uint8_t len);
static void on_servo_data(const uint8_t *payload, uint8_t len);
static void on_emergency_stop(const uint8_t *payload, uint8_t len);

int main(void) {
    SYSCFG_DL_init();    // 包含 SYSCFG_DL_UART_4_init()

    op_uart_init();      // 初始化解析器

    // 注册回调
    op_uart_register_callback(0x15, on_heartbeat);      // TYPE_HEARTBEAT
    op_uart_register_callback(0x17, on_servo_data);     // TYPE_VISUAL_SERVO_DATA
    op_uart_register_callback(0x18, on_emergency_stop); // TYPE_EMERGENCY_STOP

    while (1) {
        // 业务循环
    }
}
```

---

## 10. 发送示例（MCU → OP 方向）

```c
// 发送 TYPE_ARRIVED zone=3
uint8_t arrived_payload = 3;
op_uart_send(0x02, &arrived_payload, 1);

// 发送 TYPE_ACTION_DONE action_id=1 result=0
uint8_t action_payload[2] = {1, 0};
op_uart_send(0x14, action_payload, 2);

// 发送 TYPE_CMD_FROM_MCU cmd=CMD_START_QR
uint8_t cmd_payload = 0x01;
op_uart_send(0x10, &cmd_payload, 1);

// 发送 TYPE_HEARTBEAT seq=42, mission_state=5, visual_state=0
uint8_t hb_payload[3] = {42, 5, 0};
op_uart_send(0x15, hb_payload, 3);
```

---

## 11. 帧类型对照表

| Type | 名称 | Payload | 方向 |
|:---|:---|:---|:---:|
| 0x01 | `TYPE_ERROR` | `error_type(1B) + error_value(2B LE)` | OP→MCU |
| 0x02 | `TYPE_ARRIVED` | `zone_id(1B)` | **MCU→OP** |
| 0x03 | `TYPE_PICK` | `zone_id(1B)` | **MCU→OP** |
| 0x04 | `TYPE_SET` | `zone_id(1B)` | **MCU→OP** |
| 0x10 | `TYPE_CMD_FROM_MCU` | `cmd_id(1B) + args` | **MCU→OP** |
| 0x11 | `TYPE_STATUS_FROM_VISION` | `mission_state + visual_state + flags + cargo_count` | OP→MCU |
| 0x12 | `TYPE_QR_RESULT` | `len(1B) + ascii[len]` | OP→MCU |
| 0x13 | `TYPE_COLOR_RESULT` | `color_id(1B) + confidence(1B)` | OP→MCU |
| 0x14 | `TYPE_ACTION_DONE` | `action_id(1B) + result(1B)` | **MCU→OP** |
| 0x15 | `TYPE_HEARTBEAT` | `seq + mission_state + visual_state` | 双向 |
| 0x16 | `TYPE_REQUEST_SYNC` | `requested_state(1B)` | 双向 |
| 0x17 | `TYPE_VISUAL_SERVO_DATA` | `error_x(2B) + error_y(2B) + flags(1B) + state(1B)` | OP→MCU |
| 0x18 | `TYPE_EMERGENCY_STOP` | `reason(1B)` | 双向 |

> 完整内容定义见 [`protocol_content.md`](protocol_content.md)。

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
  ├ empty.c: op_uart_init() + 回调注册
  └ 端到端环回测试（OP ↔ MCU 互发心跳验证链路）

Step 5 — 业务帧对接
  └ 逐个接入 ARRIVED / ACTION_DONE / CMD / SERVO_DATA 等帧
```
