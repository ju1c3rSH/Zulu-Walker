# 软件 AEC（自动曝光控制）

## 动机

MaixCAM2 内置 AE 在暗光下会自动拉长曝光时间，导致高速钢球产生拖影，YOLO11n 检测失效。解决方案：**固定短曝光消除拖影，软件 PI 控制器动态调 gain 补偿亮度**。

## 架构

```
project_config.yaml
  → Machine.create()              ← 转发 exposure_us/gain/aec
    → CameraHub.open(**kwargs)
      → create_camera(**kwargs)   ← pop 三个新字段
        → MaixCam2Camera(..., exposure_us=3000, gain=200, aec={...})
          → cam.exposure(3000)    ← 锁定手动曝光 (AeMode.Manual)
          → cam.gain(200)         ← 初始增益

VisionManager._process_loop (单线程, 60fps):
  [每帧]
    pipe.process_frame() → cam.read_raw() → 缓存 BGR ndarray
    _update_display_frame()
  [每30帧]
    _adjust_exposure()
      → cam.last_frame (已缓存 BGR)
      → ROI裁剪 → BGR2GRAY → np.mean
      → EMA平滑 → PI控制器 → set_gain(clamped)
```

## 数据流

- `read()` / `read_raw()` 都缓存 `self._last_frame`（BGR ndarray）
- AEC 读取已缓存的 `last_frame`，**无额外 camera I/O**
- `set_gain()` / `set_exposure()` 有独立 try/except + None 守卫
- `set()` 委托给 `set_gain()`/`set_exposure()`，保持 `_last_gain` 同步

## 算法

```
ROI灰度均值 → EMA(α=0.1) → err = target - ema
→ |err| ≤ deadband? → return
→ P-only 预钳位 candidate_gain
→ gain未饱和? → conditional I-term 累加
→ delta = kp*err + ki*I → gain = clamp(last + delta, min, max)
→ set_gain(gain)
```

- **固定 exposure**：消除拖影（双方程中占主导的副作用）
- **只调 gain**：传感器响应快，代价（噪点）对 CNN 检测器相对鲁棒
- **条件积分 (anti-windup)**：gain 饱和时暂停累加，防止积分饱和
- **EMA 平滑**：时间域等效于论文 arXiv:1705.05685 的 Gaussian 空间域平滑

## 配置参数 (`project_config.yaml`)

```yaml
cameras:
  - camera_id: "main"
    exposure_us: 3000            # 固定曝光 us, 3ms
    gain: 200                    # 初始增益
    aec:
      enabled: true              # false = 固定 gain 模式
      adjust_interval_frames: 30 # ~0.5s @60fps
      target_mean: 80            # ROI 目标亮度, 0-255, 越低越暗
      gain_min: 50               # 增益下限
      gain_max: 600              # 增益上限 (噪点容忍)
      kp: 0.5                    # 比例系数
      ki: 0.05                   # 积分系数
      max_i: 100                 # 积分钳位 (anti-windup)
      deadband: 8                # 死区, |err|≤此值跳过
      roi: [0.4, 0.7, 0.0, 1.0] # [y0_frac, y1_frac, x0_frac, x1_frac]
      ema_alpha: 0.1            # EMA 平滑系数
```

## 文件改动

| 文件 | 改动 |
|------|------|
| `project_config.yaml` | 新增 exposure_us / gain / aec 配置段 |
| `framework/hal/platforms/maixcam2/camera.py` | 构造函数扩展 exposure_us/gain/aec 参数；set_gain/set_exposure 方法；last_gain/last_frame/aec_config 属性；set() 委托给 set_gain/set_exposure |
| `framework/hal/platforms/maixcam2/__init__.py` | create_camera() pop 三个新字段 |
| `framework/hal/machine.py` | hub.open() 转发三个新字段 |
| `modules/zw_opencv_module/vision_manager.py` | AEC 状态初始化；start() 从 camera 加载配置；_process_loop 每30帧调用；_adjust_exposure PI 控制器 |

## 关键设计决策

| 决策 | 理由 |
|------|------|
| AEC 逻辑放在 VisionManager | HAL 保持原语层，算法层与帧管道共用数据 |
| `set_gain`/`last_gain` 不在 Camera Protocol | AEC 是 MaixCAM2 专有功能，用 getattr/hasattr 守卫 |
| 条件积分（非饱和时才累加） | 防止 gain 钳位后积分饱和导致 overshoot |
| 移除 `any_fresh` 守卫 | `read_raw()` 第一帧后永远返回非 None，守卫是死代码 |
| 存储 camera_id 而非硬编码 "main" | 允许配置不同 camera_id 不失效 |
| 构造函数 post-init 分离 | Camera() 成功后 exp/gain 失败不会留下 dangling handle |

## 性能评估 (MaixCAM2)

- **AEC 每次执行**：~0.5-3ms（含 set_gain I2C 写入）
- **触发间隔**：每 30 帧 (0.5s) → 均摊 <0.1ms/帧
- **看门狗**：超时 10s，限频 1 Hz → 3000x 安全裕度
- **GC 压力**：AEC 分配 136KB/0.5s（0.27MB/s），对比相机帧 81MB/s → 0.33%
- **60fps 预算**：单帧 16.67ms，推理+绘制 ~12ms，AEC 最多 +3ms → 仍有裕量

## 参考

- arXiv:1705.05685 — "Active Control of Camera Parameters for Object Detection Algorithms" (Wu & Tsotsos)
  - 验证主动相机参数控制可提升 CNN 检测器 mAP +6%
  - 论文使用离线查表法 + Gaussian 平滑；本项目使用在线 PI 反馈法
- GitHub: Fzzzhan/Autoexposure-of-PI-camera — PI 控制器 + HSV 直方图参考实现
