# -*- coding: utf-8 -*-
from queue import Queue
from threading import Thread
import numpy as np
import cv2
import sys



class CameraStream:
    """
    摄像头初始化的时候，会先检测是否支持V4L2接口（Linux专用），如果支持则使用V4L2接口进行视频捕获，这通常会提供更好的性能和更低的延迟。如果不支持V4L2接口，则回退到默认的视频捕获方式。摄像头参数设置部分也进行了错误处理，以确保在某些参数无法设置时不会导致程序崩溃。帧的读取和更新通过一个独立的线程进行，使用队列来存储最新的帧，确保读取时不会阻塞。
    设置完捕获方式后，会开始设置相关摄像头参数，如分辨率、帧率、缓冲区大小和视频编码格式。最后，摄像头捕获线程会持续运行，直到调用`release`方法来停止线程并释放摄像头资源。
    
    """
    def __init__(self, source=0, width=640, height=480):
        try:
            self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        except AttributeError:
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera source {source}")

        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        except: pass
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        except: pass
        try:
            self.cap.set(cv2.CAP_PROP_FPS, 60)
        except: pass
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except: pass
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except: pass

        self.queue = Queue(maxsize=2)
        self.running = True
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            #print(f"CameraStream: Read frame - ret={ret}, frame_shape={frame.shape if ret else 'N/A'}")
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                self.running = False
                break

            """
            这里使用非阻塞式队列来更新每一帧的数据
            如果队列已满，则先尝试移除旧的帧以腾出空间，然后再将新的帧放入队列中。这种方式确保了读取最新帧时不会被旧帧阻塞，同时也避免了内存占用过多的问题。
            """        
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except:
                    pass

            try:
                self.queue.put_nowait(frame)
            except:
                pass

    def read_frame(self):
        try:
            return self.queue.get_nowait()
        except:
            return None

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.cap.release()

#usage :camera = CameraStream(0)