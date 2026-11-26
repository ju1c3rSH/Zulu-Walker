# MaixPy 绘制后端的真实机制（maix.image / maix.display）

> 生成日期：2026-11-26
> 目的：为「跨平台统一 UI」的 Canvas 抽象提供事实依据——MaixPy 到底怎么画、画在哪、贵在哪。
> 证据来源：
> - 上游实现（抓取自 github.com/sipeed/MaixCDK `main`，与板上固件版本可能略有差异）：
>   `components/vision/src/maix_image.cpp`、`components/vision/src/maix_display.cpp`
> - 本机 `F:\MaixPy-main`：`docs/doc/zh/vision/{image_ops,opencv,dual_buff}.md`、`components/maix/{gen_api_cpp.py,CMakeLists.txt}`
> - 本仓：`modules/zw_opencv_module/vision_manager.py`、`modules/zw_opencv_module/detectors/pendulum_calibrator/debug/debug_device.py`
> - 配套文档：`docs/architecture/maixcdk_perf_findings.md`（基于 MaixCDK 源码实读的性能问题候选与提 PR 建议）

---

## 1. 一句话结论

**MaixPy 的绘制 = OpenCV 在 Image 自己那块内存上直接画。** 没有中间画布、没有绘制指令队列、没有 GPU 光栅化；`maix.image.Image` 就是「一块 4KB 对齐的裸内存 + 格式描述符」，每个 `draw_*` 方法把它包成一个零拷贝 `cv::Mat` 视图，交给 OpenCV 写回。

推论：
- **几何图元（rect/line/circle）很便宜**，因为就是 OpenCV 的 `cv::rectangle`/`cv::line`/`cv::circle`；
- **文本很贵**（抗锯齿 + 字形光栅化 + 每帧重绘）；
- **格式错了很贵**（`show()` 内部整帧转换）；
- **颜色传错会静默失败**（Python 元组 vs C++ `Color` 类型不匹配，异常被上层兜底吞掉）。

---

## 2. 调用链：Python → C++ → OpenCV

```
img.draw_rect(...)            # Python
  ↓ pybind11 自动生成的 wrapper
maix::image::Image::draw_rect(...)   # C++（MaixCDK components/vision）
  ↓
cv::rectangle(cv::Mat(_height,_width,ch_format,_data), ...)  # OpenCV 直接写像素
```

**MaixPy 仓库里没有绘制实现**，只有绑定生成器：

- `F:\MaixPy-main\components\maix\gen_api_cpp.py` 解析 MaixCDK 头文件，生成 `maixpy_wrapper.cpp`（`PYBIND11_MODULE(_maix, m)`）；
- `components/maix/CMakeLists.txt` 里 `gen_api_cpp.py --sdk_path ${SDK_PATH}`；
- `components/maix/component.py` 的 `requires` 里有 `vision`——**真正的实现在 MaixCDK**（`github.com/sipeed/MaixCDK`），需 `MAIXCDK_PATH` 环境变量。

所以：`pip install maix` 装到的是「绑定 + MaixCDK 内核」，内核必须按平台编译——这决定了「桌面 Linux 也用 maix.*」不是随手可得的方案。

---

## 3. Image 是什么

```cpp
// maix_image.cpp :: Image::_create_image
_actual_data = malloc(_data_size + 0x1000);
_data = (void *)(((uint64_t)_actual_data + 0x1000) & ~0xFFF);   // 4KB 对齐
```

- 一块 `malloc` 的裸内存 + `_format`/`_width`/`_height`/`_data_size`；
- **4KB 对齐**——这是能零拷贝喂给硬件 VO / NPU 的前提；
- 常用格式：`FMT_RGB888`（内存里 R,G,B 逐像素 packed）、`FMT_BGR888`、`FMT_RGBA8888`、`FMT_BGRA8888`、`FMT_GRAYSCALE`、`FMT_YUV420SP`、`FMT_JPEG`。

---

## 4. 绘制原语：全部走 OpenCV，且是零拷贝视图

所有 `draw_*` 的前两行长得一模一样：

```cpp
int ch_format = 0; cv::Scalar cv_color;
_get_cv_format_color(_format, color, &ch_format, cv_color);
cv::Mat img(_height, _width, ch_format, _data);   // 视图：包住自己的 _data，不拷贝
```

然后：

| 原语 | OpenCV 实现 |
|---|---|
| `draw_rect` | `cv::rectangle`（`thickness=-1` 填充） |
| `draw_line` | `cv::line` |
| `draw_circle` | `cv::circle` |
| `draw_ellipse` | `cv::ellipse` |
| `draw_arrow` / `draw_cross` / `draw_keypoints` | 同族几何图元 |
| `draw_string` | `cv::freetype::FreeType2::putText` 或 `cv::putText`（见 §5） |
| `draw_image` | 图像贴图（sprite blit） |
| `to_format` | `cv::cvtColor` |
| `load` | `cv::imread` + `cv::cvtColor` |

**代价模型**：几何图元 ≈ 一次 OpenCV 调用 + 少量内存写；文本 = 逐字形光栅化 + 抗锯齿混合；`to_format` = 整帧。

---

## 5. 文本绘制（最贵的原语，细节最多）

```cpp
// maix_image.cpp :: _put_text
if (font_id == -1) {                                  // 加载了 TTF/OTF
    cv::Ptr<cv::freetype::FreeType2> ft2 = fonts_info[font_name];
    cv::Point point_tmp(point.x, point.y + ft2->getTextSize(...).height);
    ft2->putText(img, text, point_tmp, scale * fonts_size_info[font_name],
                 color, thickness, cv::LINE_AA, true);
} else {                                              // 内置 Hershey 字体
    cv::Point point_tmp(point.x, point.y + cv::getTextSize(...).height);
    cv::putText(img, text, point_tmp, font_id, scale, color, thickness, cv::LINE_AA, false);
}
```

要点：

1. **字体栈 = OpenCV 的 freetype 模块（FT2）**。`image.load_font(name, path, size)` 加载字体，`fonts_info[font_name]` 缓存 `cv::Ptr<FreeType2>`；`image.set_default_font(name)` 设默认。
2. **坐标系补偿**：MaixPy 的 `(x, y)` 是**文字左上角**，OpenCV 是**基线左下** → 内部统一补偿 `y += 首个字符高度`。**我们的 Canvas 抽象必须沿用「左上原点」语义**，否则两个后端的文字会错位一截。
3. **抗锯齿恒开**（`cv::LINE_AA`）→ 文本逐像素混合，成本远高于几何图元。
4. `image.string_size(text, scale, thickness)` 用同一字体缓存算尺寸，**与 draw_string 保持一致的 scale/thickness** 才有意义。
5. 默认字体**只支持英文**；中文必须 `load_font`（板载 `/maixapp/share/font/SourceHanSansCN-Regular.otf`）。
6. **RGBA 画布上 draw_string 的 alpha 不可靠**（本仓 `_make_rgba_text_sprite` 注释自述），所以透明底文字的惯用做法是：RGBA sprite 上画完，再用 `get_pixel/set_pixel` 逐像素补 alpha（`vision_manager.py:630` 附近）。

---

## 6. 颜色：一个必须踩准的坑

```cpp
// maix_image.cpp :: _get_cv_format_color
if (color->format != _format) { color = color->to_format2(_format); color_alloc = true; }
switch (_format) {
case image::FMT_RGB888: *ch_format = CV_8UC3; cv_color = cv::Scalar(color->r, color->g, color->b); break;
case image::FMT_BGR888: *ch_format = CV_8UC3; cv_color = cv::Scalar(color->b, color->g, color->r); break;
case image::FMT_GRAYSCALE: *ch_format = CV_8UC1; cv_color = cv::Scalar(color->gray); break;
case image::FMT_BGRA8888: *ch_format = CV_8UC4; cv_color = cv::Scalar(color->b, color->g, color->r, color->alpha*255); break;
case image::FMT_RGBA8888: *ch_format = CV_8UC4; cv_color = cv::Scalar(color->r, color->g, color->b, color->alpha*255); break;
}
```

- C++ 侧签名是 `const image::Color &color`。**Python 侧必须传 `image.Color.from_rgb(...)` 或 `image.COLOR_*`**；传 RGB 元组 → pybind11 参数类型不匹配 → 抛 `TypeError` → 被上层 `except` 吞掉 → **界面静默空白**（本仓三个 tuner 的头注释都记了这条坑）。
- `Color` 的 `format` 与画布 `format` 不一致时会 `to_format2`（一次堆分配 + 释放）。
- 注意 RGB888 / BGR888 的通道顺序在 C++ 里被正确 swap——**画布是 RGB888 时颜色语义就是 RGB**。

---

## 7. show()：白名单决定快慢（慢路径的根源）

```cpp
// maix_display.cpp :: Display::show(image::Image &img, image::Fit fit)
if (img_trans) img_trans->send_image(img);            // 转发给 MaixVision（IDE 预览通道）

#if defined(PLATFORM_MAIXCAM) || defined(PLATFORM_MAIXCAM2)
if (show_img_format != FMT_RGB888 && != FMT_YVU420SP && != FMT_YUV420SP
    && != FMT_BGRA8888 && != FMT_GRAYSCALE) {
    image::Image *show_img = img.to_format(FMT_RGB888);   // ⚠ 整帧 cvtColor + 堆分配
    _impl->show(*show_img, fit);
    delete show_img;                                     // ⚠ 每帧分配/释放
} else {
    _impl->show(img, fit);                               // ✅ 零拷贝直通
}
```

- **白名单内**（RGB888 / YUV420SP / YVU420SP / BGRA8888 / GRAYSCALE）→ 零拷贝直通硬件 VO；
- **白名单外（典型是 BGR888）→ 整帧格式转换 + 每帧堆分配**，这就是本仓 `framework/display/types.py` 把 BGR888 标成「慢路径」的出处，也是「cv2image 转换在 AX630C 上 10~20ms vs 原生 2~6ms」实测的根源；
- **平台实现**（按 `#ifdef` 切换）：`PLATFORM_LINUX` → `maix_display_sdl.hpp`（**SDL**）；`PLATFORM_MAIXCAM` → `maix_display_mmf.hpp`；`PLATFORM_MAIXCAM2` → `maix_display_maixcam2.hpp`；另有 `maix_display_fb.hpp`（framebuffer）；
- 每次 `show()` 都会 `img_trans->send_image(img)`——MaixVision 预览通道，线上有额外开销；
- 桌面 Linux 分支走 SDL + crop/resize/`to_format`，**说明「maix.* 跑在桌面 Linux」在实现上是可行的**（前提是自己编 MaixCDK 的 linux 轮子）。

---

## 8. OpenCV 互转：copy=True/False 的差别很大

（`docs/doc/zh/vision/opencv.md`）

```python
img_bgr = image.image2cv(img, ensure_bgr=True,  copy=True)   # 拷贝一份，慢但安全
img_rgb = image.image2cv(img, ensure_bgr=False, copy=False)  # 借用 img 的内存，零拷贝
img2    = image.cv2image(img_rgb, bgr=False, copy=False)      # 同样借用，零拷贝
```

- `copy=False` 时 `img_rgb` 与 `img` **共用内存**：改一边另一边也变；`img_rgb` 存活期间 `img` 不能被销毁，否则崩溃。
- `ensure_bgr=False` 拿到的是 **RGB** 图，而 OpenCV API 一律按 BGR 解释 → 通道会反。
- `cv2image(bgr=True)` 默认产出 BGR888 → **正好掉进 show() 的慢路径白名单外**，等于「转换一次 + show 时再转一次」，双重开销。

---

## 9. 还有一条更省的路径：maix.pipeline

- `Display::push(pipeline::Frame *frame, image::Fit fit)` 存在于 `maix_display.cpp`；
- `maix.pipeline` 提供 **Frame / Buffer 的生产者-消费者通道**（相机侧有 channel，display 侧有 consumer），帧在缓冲池里流转而不是每帧 `copy()`；
- 同类思路还有 NPU 的 `dual_buff`（`docs/doc/zh/vision/dual_buff.md`：CPU 预处理与 NPU 推理重叠，帧率翻倍，代价是多一份缓冲内存、结果有 1 帧延迟）。

> 对我们的意义：当前 `_update_display_frame()` 每帧 `raw_img.copy()`（1280×352×3 ≈ 1.29MB）+ 全量重绘，是「重建窗口」节流存在的原因。改用 pipeline 缓冲池有机会把这次拷贝去掉。

---

## 10. 我们项目目前实际怎么画的

`vision_manager.py`（全部画在 `raw_img.copy()` 出来的 RGB888 Image 上）：

| 用法 | 位置 | 说明 |
|---|---|---|
| `draw_line` / `draw_circle` | `_draw_rail` `:535 :537 :554` | 轨道轴、原点、球投影 |
| `draw_rect` | 横幅 `:668 :682 :698`、按钮底框 `:998 :1015 :1032 :1048` | 告警横幅 + 按钮描边 |
| `draw_string` | 横幅 `:669 :683 :699`、检测列表 `:919 :951 :967` | `scale=1.5, thickness=2` 的每目标文本是热点 |
| `draw_image` | sprite 贴图 `:591 :845 :857 :871 :883` | **文字 sprite 缓存**：FPS / 时间 / 测试 id / cm 刻度标签 |
| `string_size` | `:605 :624` | sprite 尺寸计算，与 draw_string 同字体同 scale |

文字加速的两条既有实践（可直接进 Canvas 设计）：
1. **不透明黑底 sprite**：`_make_rgb_text_sprite` —— 黑底白字直接贴在黑色 header 上，避开 RGBA alpha 路径；
2. **透明 sprite 需手工补 alpha**：`_make_rgba_text_sprite` —— 逐像素 `get_pixel/set_pixel` 写 alpha（因为 OpenCV freetype 在 RGBA 画布上 alpha 不可靠）。

反向的坑：`detectors/pendulum_calibrator/debug/debug_device.py:303` 用 `maix.image.cv2image(combined, bgr=True)` → 得到 BGR888 → `show()` 时再整帧转 RGB888，**双重转换**；同文件 `:404` 用 `image2cv(raw, ensure_bgr=False, copy=True)` 是拷贝而非借用。

---

## 11. 对 Canvas 抽象（跨平台统一）的直接结论

1. **原语集够用**：`maix.image` 的 `rect/line/circle/string/blit` + `string_size` 正好覆盖 D2 定的 7 原语，不需要更多。
2. **统一「左上原点」语义**：MaixPy 已在 C++ 内部把 OpenCV 的基线语义补偿成左上原点；`CvCanvas` 用 `cv2.putText` 时**必须做同样的 y 偏移**，否则两后端文字错位。
3. **统一颜色入口**：Canvas 内部一律 `(r,g,b,a)` 元组，`MaixCanvas` 负责转 `image.Color.from_rgb`，**绝不把元组漏到 C++ 边界**（否则静默空白）。
4. **文本必须缓存**：静态文案在 Canvas 层提供 `text_sprite(text)` 能力，两后端各自缓存；动态文本才直接画。
5. **像素容器必须 RGB888**：设备端只有 RGB888 走 show() 快路径。桌面 `CvCanvas` 也要保持 RGB 语义（cv2 内部按 BGR 画，最后一次 `cvtColor` 或直接用 RGB 通道顺序绘制），否则桌面端「画得快但上屏慢」。
6. **格式检查前置**：Canvas 层提供 `ensure_native_format()`，避免白名单外的格式一路走到 show() 才被整帧转换。
7. **缓冲可复用**：`Image` 支持自建缓冲，Canvas 层可维护环形缓冲，把每帧 `copy()` 换成缓冲复用（配合 `maix.pipeline` 更彻底）。
