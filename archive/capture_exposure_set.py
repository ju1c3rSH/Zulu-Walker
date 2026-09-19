import os
import sys
import time

from maix import camera, image

BASE_DIR = "./captures"

CONFIGS = [
    (500,  100, 25),
    (1500, 200, 25),
    (3000, 300, 25),
    (5000, 500, 25),
]

WIDTH = 1280
HEIGHT = 352
FPS = 60
JPEG_QUALITY = 95
STABILIZE_DELAY = 0.5
INTERVAL = 0.5


def main():
    run_label = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE_DIR, run_label)
    os.makedirs(run_dir, exist_ok=True)

    try:
        cam = camera.Camera(WIDTH, HEIGHT, image.Format.FMT_RGB888, fps=FPS, buff_num=3)
    except Exception as e:
        print(f"[FATAL] Camera init failed: {e}", file=sys.stderr)
        sys.exit(1)

    cam.skip_frames(30)

    total_shots = sum(c for _, _, c in CONFIGS)
    shot_no = 0
    cfg_count = len(CONFIGS)

    for ci, (exp, gain, count) in enumerate(CONFIGS, 1):
        group_dir = os.path.join(run_dir, f"expo_{exp}_gain_{gain}")
        os.makedirs(group_dir, exist_ok=True)

        input(f"\n=== 第 {ci}/{cfg_count} 组: expo={exp} gain={gain} x{count} 张 ===\n"
              f"    保存路径: {group_dir}\n"
              f"    调整好光照环境后按 Enter 键开始采集...")

        cam.exposure(exp)
        cam.gain(gain)
        time.sleep(STABILIZE_DELAY)

        for i in range(1, count + 1):
            img = cam.read()
            if img is None:
                print(f"[WARN] frame lost at config {ci}/{cfg_count} frame {i}/{count}")
                continue

            fname = f"expo_{exp}_gain_{gain}_{i:03d}.jpg"
            fpath = os.path.join(group_dir, fname)
            img.save(fpath)

            shot_no += 1
            print(f"[{shot_no}/{total_shots}] cfg={ci}/{cfg_count} "
                  f"expo={exp} gain={gain}  {i}/{count} -> {fname}")

            if i < count:
                time.sleep(INTERVAL)

    print(f"\nDone! {shot_no} images -> {run_dir}")


if __name__ == "__main__":
    main()
