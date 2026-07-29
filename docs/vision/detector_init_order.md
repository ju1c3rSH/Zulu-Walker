# Detector 初始化顺序规范

## 问题

EdgeDrawing 系列检测器（CargoDetector、RingDetector）在 `__init__`
中调用 `_init_edge_drawing()` 时，内部会调用 `_update_ed_params()`，
该方法需要读取 `self.ed_min_path_length` 等 ED 参数。

若参数赋值在 `_init_edge_drawing()` 调用**之后**，触发：

```
AttributeError: 'RingDetector' object has no attribute 'ed_min_path_length'
```

## 崩溃链

```
detector __init__ crash
  → processor 创建失败
  → Camera._setup_tasks() 抛异常
  → Camera 创建失败
  → CameraManager.load_config() 失败
  → zw_opencv_module.start() 返回 False
  → 处理线程未启动，config 为 None
  → 预览窗口不出现，Cam FPS 不显示
```

## 正确写法

`ed_*` 参数必须在 `_init_edge_drawing()` 之前赋值：

```python
def __init__(self):
    ...
    self.ed = None
    # ▼ 必须先赋值
    self.ed_min_path_length = 50
    self.ed_gradient_threshold = 36
    self.ed_nfa_validation = True
    self.edge_morph_kernel = 3
    self.edge_morph_iterations = 1
    # ▼ 再调用
    self._init_edge_drawing()
    ...
```

## 受影响文件

| 文件 | 修复状态 |
|------|---------|
| `detectors/cargo_detector/__init__.py` | 已修复 |
| `detectors/ring_detector/__init__.py`  | 已修复 |

## 检查清单

新增检测器时确认：

- [ ] `ed_min_path_length` 是否在 `_init_edge_drawing()` 之前赋值
- [ ] 所有 `_update_ed_params()` 读取的属性均已提前赋值

---

## 关联约束：`_setup_tasks` 总是实例化处理器

`Camera._setup_tasks()` (`vision_manager.py`) 即使任务配置了 `enabled: false`，也会创建对应的处理器对象。

禁用只控制 `task.execute()` 是否执行，但处理器 `__init__` 仍会被调用。

因此：**某个处理器的依赖模块初始化失败时，即使该任务被禁用，也会导致整个 Camera 创建失败。** 崩溃链同上。
