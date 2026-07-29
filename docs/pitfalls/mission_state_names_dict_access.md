# MissionStateNames 坑记：'dict' object has no attribute 'PICK_RAW'

> 最后更新: 2026-07-15
> 现象: `_AlignRawState.on_execute` 内 `MissionStateNames.PICK_RAW` 抛出 `AttributeError: 'dict' object has no attribute 'PICK_RAW'`

---

## 1. 现象描述

- `_AlignRawState.on_execute` 返回 `MissionStateNames.PICK_RAW` 跳转状态
- 运行时抛出 `AttributeError: 'dict' object has no attribute 'PICK_RAW'`
- `MissionStateMachine` 类内的 `_setup_states`、`_setup_transitions` 等能正常工作，不受影响

---

## 2. 根因：dict 用点号属性访问

### 2.1 定义

`mission_state_machine.py:561` — `MissionStateNames` 是 **dict**：

```python
MissionStateNames = {
    MissionState.IDLE: "IDLE",         # {0: "IDLE"}
    MissionState.PICK_RAW: "PICK_RAW", # {5: "PICK_RAW"}
    ...
}
```

### 2.2 误用

各个 state 子类（`_AlignRawState`、`_CheckLoadState`、`_AlignRoughState` 等）内部使用 **点号属性访问**：

```python
# ❌ dict 没有 .PICK_RAW 属性
return MissionStateNames.PICK_RAW
```

### 2.3 正确用法

```python
# ✅ dict 键访问，返回字符串 "PICK_RAW"
return MissionStateNames[MissionState.PICK_RAW]
```

---

## 3. 为什么 `_MissionStateMachine` 类内没问题？

`MissionStateMachine` 类（继承 `BaseStateMachine`）的方法全部使用的是 **正确的 dict 键访问**：

```python
self.register_state(MissionStateNames[MissionState.PICK_RAW], _PickRawState())
    #                       ↑  dict[MissionState.值]  ← 正确
```

而 state 子类是独立的 `State` 子类，写在文件底部，作者误用了 Python 的 dot notation 去访问 dict。

---

## 4. 涉及范围

| State 类 | 错误写法 | 修复 |
|---------|---------|------|
| `_AlignRawState` | `MissionStateNames.PICK_RAW` / `.ERROR` | `MissionStateNames[MissionState.PICK_RAW]` / `[MissionState.ERROR]` |
| `_CheckLoadState` | `.ALIGN_RAW` / `.ALIGN_ROUGH` / `.NAV_TO_ROUGH` / `.NAV_TO_TEMP` / `.ERROR` | 同上 |
| `_RingDiscoveryState` | `.ALIGN_ROUGH` / `.ALIGN_TEMP` / `.ERROR` | 同上 |
| `_AlignRoughState` | `.PICK_ROUGH` / `.PLACE_ROUGH` | 同上 |
| `_PlaceRoughState` | `.ALIGN_ROUGH` (×2) | 同上 |
| `_AlignTempState` | `.PLACE_TEMP` | 同上 |
| `_PlaceTempState` | `.ALIGN_TEMP` / `.RETURN_HOME` / `.NAV_TO_RAW_SECOND` / `.RETURN_HOME` | 同上 |
| `_ReturnHomeState` | `.FINISHED` | 同上 |

共 22 处。

---

## 5. 排查方法

搜索 `MissionStateNames.` 后跟大写字母的模式：

```bash
rg 'MissionStateNames\.\w+' context/mission_state_machine.py
```

合法的结果应只包含：
- `MissionStateNames.items()` — dict 方法
- `MissionStateNames.get(...)` — dict 方法
- `MissionStateNames[MissionState.XXX]` — dict 键访问

---

## 6. 教训

- `on_execute` 返回的是 **state name 字符串**（如 `"PICK_RAW"`），不是 state ID 整数
- `MissionStateNames` 是 `Dict[int, str]`，`MissionState.XXX` 是 int key，通过 `MissionStateNames[MissionState.XXX]` 拿到 str value
- **dict 不支持 dot notation 属性访问**（除非是 dict 方法如 `.items()`）
- 所有 state 子类的返回应与 `MissionStateMachine._setup_states()` 注册的 key 保持一致的 key 来源
