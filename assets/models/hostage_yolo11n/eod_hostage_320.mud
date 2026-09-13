[basic]
type = axmodel
model_npu = eod_hostage_320.axmodel

[extra]
model_type = yolo11
type = detector
input_type = rgb
labels = yao_gu, yuan_zhu, yuan_tai

input_cache = true
output_cache = true
input_cache_flush = false
output_cache_inval = true

mean = 0,0,0
scale = 0.00392156862745098, 0.00392156862745098, 0.00392156862745098
