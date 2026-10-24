# 测: cv2 编码是否释放 GIL? (决定"换线程"到底有没有用)
import cv2, numpy as np, threading, time

img = np.random.randint(0,256,(480,640,3),np.uint8)

def busy_worker(stop, counter):
    # 纯 Python 忙循环: 只有拿到 GIL 才能推进
    n = 0
    while not stop.is_set():
        n += 1
    counter.append(n)

def measure(encode_fn, label):
    stop = threading.Event(); counter = []
    t = threading.Thread(target=busy_worker, args=(stop,counter)); t.start()
    time.sleep(0.05)
    base = counter[0]
    t0 = time.perf_counter()
    for _ in range(30): encode_fn()
    el = time.perf_counter()-t0
    stop.set(); t.join()
    progressed = counter[0]-base
    print(f"{label:28s} encode {el/30*1000:6.2f}ms/次   后台Python推进 {progressed:>10,}")

print("=== 编码期间后台 Python 线程能否推进（能推进 = 编码释放了 GIL）===")
measure(lambda: cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY,80]), "cv2.imencode JPEG q80")
measure(lambda: cv2.resize(img,(552,368)), "cv2.resize")
def pure_py():
    s=0
    for i in range(200000): s+=i
measure(pure_py, "纯 Python 循环(对照)")
