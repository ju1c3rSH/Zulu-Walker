# -*- coding: utf-8 -*-
import asyncio
import subprocess
import cv2
import numpy as np
from typing import Optional

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

    async def start(self):
        """启动FFmpeg进程"""
        if self.process is not None:
            return
        self._stopped = False
        self.hardware_encode_error = False  # 重置硬件错误标志
        
        
        command = [                                                                                                                                                                                                                                                                                                                'ffmpeg',
            '-y',                    # 无害（输出覆盖）                                                                                                                                                                                                                                                                      
            '-f', 'rawvideo',        # 输入格式
            '-vcodec', 'rawvideo',   # 输入编解码器（一致）
            '-pix_fmt', 'bgr24',     # OpenCV默认格式
            '-s', f'{self.width}x{self.height}',  # 分辨率
            '-r', str(self.fps),     # 输入帧率
            '-i', '-',               # 从stdin读取
            '-f', 'flv',             # 输出格式
            '-flvflags', 'no_duration_filesize',  # FLV标志
            self.rtmp_url
        ]
        #the up is the base command
        
        
        if self.use_hardware_accel and not self.fallback_to_software:
            command.extend(self._get_hardware_accel_params())
        else:
            command.extend(self._get_software_encoder_params())

        print(f"启动FFmpeg命令: {' '.join(command)}")
        
        
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
            print(f"FFmpeg进程立即退出 (返回码: {self.process.returncode})")
            print(f"错误输出: {error_msg}")

            if self.use_hardware_accel and not self.fallback_to_software:
                print("硬件编码器可能不兼容，尝试回退到软件编码...")
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

        # 同时读取stdout和stderr
        stdout_task = asyncio.create_task(self.process.stdout.read())
        stderr_task = asyncio.create_task(self.process.stderr.read())

        while not self._stopped:
            done, pending = await asyncio.wait(
                [stdout_task, stderr_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            if self._stopped:
                break

            for task in done:
                try:
                    data = await task
                    if data:
                        # 打印FFmpeg的输出，特别是stderr中的错误信息
                        try:
                            text = data.decode('utf-8', errors='ignore').strip()
                            if text:
                                print(f"FFmpeg output: {text}")
                                # 检测硬件编码错误
                                error_keywords = ['h264_rkmpp', 'RKMPP', 'Hardware', 'failed', 'error', 'Invalid', 'unsupported', 'not found']
                                if any(keyword.lower() in text.lower() for keyword in error_keywords):
                                    print(f"检测到可能的硬件编码错误: {text[:100]}")
                                    self.hardware_encode_error = True
                        except:
                            # 如果无法解码，忽略
                            pass
                except Exception as e:
                    print(f"Error reading FFmpeg output: {e}")

            # 重新启动已完成的任务
            if stdout_task in done:
                stdout_task = asyncio.create_task(self.process.stdout.read())
            if stderr_task in done:
                stderr_task = asyncio.create_task(self.process.stderr.read())

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
            print("Warning: Received an empty frame. Skipping.")
            return False

        # 检查是否检测到硬件编码错误并需要回退
        if self.hardware_encode_error and self.use_hardware_accel and not self.fallback_to_software:
            print("检测到硬件编码错误，尝试回退到软件编码...")
            self.fallback_to_software = True
            if self.process:
                await self.close()
            await self.start()
            # 继续推送帧

        if self.process is None or self.process.stdin is None:
            print("FFmpeg process not started or stdin closed")
            return False

        # 检查帧尺寸是否匹配
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            print(f"Warning: Frame size {frame.shape[:2]} doesn't match expected {self.height}x{self.width}")
            # 调整帧尺寸
            frame = cv2.resize(frame, (self.width, self.height))

        try:
            # 写入帧数据
            self.process.stdin.write(frame.tobytes())
            await self.process.stdin.drain()
            return True
        except BrokenPipeError:
            print("Error pushing frame: Broken pipe - FFmpeg process may have terminated")
            await self.close()

            # 检查是否需要回退到软件编码
            if self.use_hardware_accel and not self.fallback_to_software:
                print("硬件编码失败，尝试回退到软件编码...")
                self.fallback_to_software = True
                # 重新启动使用软件编码
                await self.start()
                # 重试推送当前帧
                return await self.push_frame(frame)
            return False
        except Exception as e:
            print(f"Error pushing frame: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            await self.close()

            # 检查是否需要回退到软件编码
            if self.use_hardware_accel and not self.fallback_to_software:
                print("硬件编码失败，尝试回退到软件编码...")
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

            # 等待进程结束
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                print("FFmpeg process didn't terminate in time, forcing kill")
                self.process.kill()
                await self.process.wait()

            self.process = None

    # 同步包装器，便于在同步代码中使用
    def push_frame_sync(self, frame):
        """同步推送帧（在已有事件循环中运行）"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果事件循环已在运行，我们需要在另一个线程中运行
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(lambda: asyncio.run(self.push_frame(frame)))
                    return future.result()
            else:
                # 没有运行的事件循环，直接运行
                return asyncio.run(self.push_frame(frame))
        except Exception as e:
            print(f"Error in push_frame_sync: {e}")
            return False

    def close_sync(self):
        """同步关闭"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(lambda: asyncio.run(self.close()))
                    return future.result()
            else:
                asyncio.run(self.close())
        except Exception as e:
            print(f"Error in close_sync: {e}")

    def __del__(self):
        """析构函数，确保资源清理"""
        if self.process is not None and not self._stopped:
            print("Warning: FFmpegPusher not properly closed. Forcing cleanup.")
            # 注意：在析构函数中无法运行异步代码，所以只能尽力而为
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
                else:
                    asyncio.run(self.close())
            except:
                pass