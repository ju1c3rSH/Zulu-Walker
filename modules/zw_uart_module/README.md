# zw_uart_module

UART protocol module for the 2026 GKS near-end MaixCAM vision slave
(**panzer_links_protocol v2.1**, talking to the panzer_Brain USART1 master).

See `README_CN.md` (Chinese, authoritative) and
`docs/competition/gks_protocol_crosscheck.md` for the four-source cross-check
(md spec ↔ serial_protocol C library ↔ panzer copy ↔ panzer_Brain usage).

Frame: `AA 55 | Length | Type | Payload[0..252] | CRC16_HI CRC16_LO`,
CRC16-CCITT (poly 0x1021, init 0xFFFF, big-endian on wire), payload fields
little-endian. Types: 0x20 REQUEST / 0x21 ACK / 0x22 NACK / 0x23 STOP /
0x24 DATA_STREAM (payload always 10B = seq + data_id + 8B sub-payload).
data_ids on this link: 0x01 EXPLOSIVE / 0x02 TARGET / 0x03 RESCUE.

Key semantics: zero-value heartbeat while subscribed (Brain 1s staleness),
idempotent re-subscription keeps vision latches, NACK unknown data_ids,
parser is a 1:1 port of the C `Protocol_ParserFeed` state machine
(AA-AA-55 resync, whole-frame drop on CRC error).
