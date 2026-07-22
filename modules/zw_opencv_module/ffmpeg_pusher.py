# -*- coding: utf-8 -*-
import asyncio
import subprocess
import cv2
import numpy as np
import threading
from typing import Optional
from utils.log_util import log_print


class FFmpegPusher:
    def __init__(self, rtmp_url, fps=30, width=1280, height=720, use_hardware_accel=False):
        self.rtmp_url = rtmp_url
        self.fps = fps
        self.width = width
        self.height = height
        self.process: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._stopped = False
        self.use_hardware_accel = use_hardware_accel
        self.fallback_to_software = False # 添加一个标志来跟踪是否已经尝试过硬件编码
        self.hardware_encode_error = False # 检测到硬件编码错误
        self._loop = None           # 后台事件循环
        self._loop_thread = None    # 运行循环的线程
        self._loop_lock = threading.Lock()  # 线程安全锁

    async def start(self):
        """启动FFmpeg进程"""
        if self.process is not None:
            return
        self._stopped = False
        self.hardware_encode_error = False  # 重置硬件错误标志

        # 构建命令：输入参数 -> 编码器参数 -> 输出格式 -> URL
        command = ['ffmpeg', '-y']

        # 1. 输入参数
        command.extend([
            '-f', 'rawvideo',        # 输入格式
            '-vcodec', 'rawvideo',   # 输入编解码器
            '-pix_fmt', 'bgr24',     # OpenCV默认格式
            '-s', f'{self.width}x{self.height}',  # 分辨率
            '-r', str(self.fps),     # 输入帧率
            '-i', '-',               # 从stdin读取
        ])

        # 2. 编码器参数（必须在输出格式之前）
        if self.use_hardware_accel and not self.fallback_to_software:
            command.extend(self._get_hardware_accel_params())
        else:
            command.extend(self._get_software_encoder_params())

        # 3. 输出格式和URL
        command.extend([
            '-f', 'flv',             # 输出格式
            '-flvflags', 'no_duration_filesize',  # FLV标志
            self.rtmp_url
        ])

        log_print(f"启动FFmpeg命令: {' '.join(command)}")
        
        
        self.process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        await asyncio.sleep(0.1)
        if self.process.returncode is not None:
            # 进程已退出，可能是参数不兼容
            stderr_output = await self.process.stderr.read()
            error_msg = stderr_output.decode('utf-8', errors='ignore')[:500]
            log_print(f"FFmpeg进程立即退出 (返回码: {self.process.returncode})")
            log_print(f"错误输出: {error_msg}")

            if self.use_hardware_accel and not self.fallback_to_software:
                log_print("硬件编码器可能不兼容，尝试回退到软件编码...")
                self.fallback_to_software = True
                self.process = None
                # 重新启动使用软件编码
                return await self.start()
            else:
                raise RuntimeError(f"FFmpeg进程启动失败: {error_msg}")

        # 启动输出读取任务
        self._reader_task = asyncio.create_task(self._read_output())

    async def _read_output(self):
        """读取FFmpeg的输出，防止缓冲区阻塞"""
        if self.process is None:
            return

        async def read_stream(stream, prefix):
            """逐行读取流输出"""
            while not self._stopped:
                try:
                    line = await stream.readline()
                    if not line:
                        break
                    text = line.decode('utf-8', errors='ignore').strip()
                    if text:
                        log_print(f"[{prefix}] {text}")
                        # 检测硬件编码错误
                        error_keywords = ['failed', 'error', 'Invalid', 'unsupported', 'not found', 'Connection refused']
                        if any(keyword.lower() in text.lower() for keyword in error_keywords):
                            log_print(f"[FFmpegPusher] 检测到可能的错误: {text[:200]}")
                            if 'h264_rkmpp' in text.lower() or 'rkmpp' in text.lower():
                                self.hardware_encode_error = True
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    log_print(f"Error reading {prefix}: {e}")
                    break

        # 同时启动 stdout 和 stderr 读取任务
        stdout_task = asyncio.create_task(read_stream(self.process.stdout, "FFmpeg-out"))
        stderr_task = asyncio.create_task(read_stream(self.process.stderr, "FFmpeg-err"))

        # 等待两个任务完成
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

    def _get_software_encoder_params(self):
        """获取软件编码器参数"""
        return [
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-crf', '23',
            '-g', str(self.fps),
            '-b:v', '2000k',
            '-bufsize', '1000k',
            '-maxrate', '2000k',
        ]
        
        
    def _get_hardware_accel_params(self):
        """获取硬件加速编码器参数"""
        params = [
                '-c:v' , 'h264_rkmpp',
                '-pix_fmt', 'nv12'
            ]
            
        total_pixels = self.width * self.height
        if total_pixels <= 640 * 480:
            bitrate = '1000k'
        elif total_pixels <= 1280 * 720:
            bitrate = '2000k'
        else:
            bitrate = '4000k'
            
        params.extend(['-b:v', bitrate,
                        '-g', str(min(self.fps,30)),
                        '-profile:v', 'baseline',
                        '-tune', 'zerolatency',
                        '-bf', '0',
                        '-qp_min', '18',
                        '-qp_max', '28',
                        '-preset', 'ultrafast',
                        ])
            
        if total_pixels <= 1280 * 720:
                params.extend(['-bufsize', '500k'])
            
        return params
    
    
    
    async def push_frame(self, frame):
        """推送一帧到FFmpeg"""
        if frame is None:
            log_print("Warning: Received an empty frame. Skipping.")
            return False

        if self.hardware_encode_error and self.use_hardware_accel and not self.fallback_to_software:
            log_print("检测到硬件编码错误，尝试回退到软件编码...")
            self.fallback_to_software = True
            if self.process:
                await self.close()
            await self.start()

        if self.process is None or self.process.stdin is None:
            log_print("FFmpeg process not started or stdin closed")
            return False

        if self.process.returncode is not None:
            log_print(f"FFmpeg process has exited with code {self.process.returncode}")
            await self.close()
            return False

        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            log_print(f"Warning: Frame size {frame.shape[:2]} doesn't match expected {self.height}x{self.width}")
            frame = cv2.resize(frame, (self.width, self.height))

        try:
            data = frame.tobytes()
            # log_print(f"[FFmpegPusher] Writing {len(data)} bytes to FFmpeg stdin")
            self.process.stdin.write(data)
            #log_print(f"[FFmpegPusher] Data written, calling drain...")
            await self.process.stdin.drain()
            #log_print(f"[FFmpegPusher] Drain completed successfully")
            return True
        except BrokenPipeError:
            log_print("Error pushing frame: Broken pipe - FFmpeg process may have terminated")
            await self.close()

            # 检查是否需要回退到软件编码
            if self.use_hardware_accel and not self.fallback_to_software:
                log_print("硬件编码失败，尝试回退到软件编码...")
                self.fallback_to_software = True
                # 重新启动使用软件编码
                await self.start()
                return await self.push_frame(frame)
            return False
        except Exception as e:
            log_print(f"Error pushing frame: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            await self.close()

            # 检查是否需要回退到软件编码
            if self.use_hardware_accel and not self.fallback_to_software:
                log_print("硬件编码失败，尝试回退到软件编码...")
                self.fallback_to_software = True
                # 重新启动使用软件编码
                await self.start()
                # 重试推送当前帧
                return await self.push_frame(frame)
            return False

    async def close(self):
        """关闭FFmpeg进程"""
        if self._stopped:
            return

        self._stopped = True

        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self.process:
            if self.process.stdin:
                self.process.stdin.close()
                await self.process.stdin.wait_closed()

            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                log_print("FFmpeg process didn't terminate in time, forcing kill")
                self.process.kill()
                await self.process.wait()

            self.process = None

    def _get_or_create_loop(self):
        """获取或创建后台事件循环（线程安全）"""
        with self._loop_lock:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                self._loop_thread = threading.Thread(
                    target=self._run_loop,
                    args=(self._loop,),
                    daemon=True
                )
                self._loop_thread.start()
            return self._loop

    def _run_loop(self, loop):
        """在后台线程中运行事件循环"""
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def start_sync(self):
        """同步启动 FFmpeg（使用后台事件循环）"""
        try:
            loop = self._get_or_create_loop()
            future = asyncio.run_coroutine_threadsafe(self.start(), loop)
            return future.result(timeout=10.0)
        except Exception as e:
            log_print(f"Error in start_sync: {e}")
            return False
    def push_frame_sync(self, frame):
        """同步推送帧（使用后台事件循环，支持高频调用）"""
        try:
            loop = self._get_or_create_loop()
            future = asyncio.run_coroutine_threadsafe(
                self.push_frame(frame), loop
            )
            # 5秒超时，避免无限等待
            return future.result(timeout=5.0)
        except Exception as e:
            log_print(f"Error in push_frame_sync: {e}")
            return False

    def close_sync(self):
        """同步关闭（包含事件循环清理）"""
        try:
            # 先关闭 FFmpeg 进程
            if self._loop and not self._loop.is_closed():
                future = asyncio.run_coroutine_threadsafe(
                    self.close(), self._loop
                )
                try:
                    future.result(timeout=10.0)
                except Exception as e:
                    log_print(f"Error during close: {e}")

            # 停止事件循环
            with self._loop_lock:
                if self._loop and not self._loop.is_closed():
                    self._loop.call_soon_threadsafe(self._loop.stop)
                    # 等待线程结束
                    if self._loop_thread and self._loop_thread.is_alive():
                        self._loop_thread.join(timeout=2.0)
                    self._loop.close()
                    self._loop = None
                    self._loop_thread = None
        except Exception as e:
            log_print(f"Error in close_sync: {e}")

    def __del__(self):
        """析构函数，确保资源清理"""
        if self.process is not None and not self._stopped:
            log_print("Warning: FFmpegPusher not properly closed. Forcing cleanup.")
            # 注意：在析构函数中无法运行异步代码，所以只能尽力而为
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
                else:
                    asyncio.run(self.close())
            except:
                pass