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
        self.fallback_to_software = False # Whether a software fallback was selected.
        self.hardware_encode_error = False # Set when a log line reports encoder failure.
        self._loop = None           # Background event loop.
        self._loop_thread = None    # Thread running the loop.
        self._loop_lock = threading.Lock()  # Guards loop creation and teardown.

    async def start(self):
        """Start the FFmpeg subprocess."""
        if self.process is not None:
            return
        self._stopped = False
        self.hardware_encode_error = False  # Reset the hardware error flag.

        # Build the command in order: input, encoder, output format, URL.
        command = ['ffmpeg', '-y']

        # 1. Input parameters.
        command.extend([
            '-f', 'rawvideo',        # Input container.
            '-vcodec', 'rawvideo',   # Input codec.
            '-pix_fmt', 'bgr24',     # OpenCV default pixel format.
            '-s', f'{self.width}x{self.height}',  # Frame size.
            '-r', str(self.fps),     # Input frame rate.
            '-i', '-',               # Read frames from stdin.
        ])

        # 2. Encoder parameters (must precede the output format).
        if self.use_hardware_accel and not self.fallback_to_software:
            command.extend(self._get_hardware_accel_params())
        else:
            command.extend(self._get_software_encoder_params())

        # 3. Output format and destination.
        command.extend([
            '-f', 'flv',             # Output container.
            '-flvflags', 'no_duration_filesize',  # FLV flags.
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
            # Process exited during startup, likely incompatible options.
            stderr_output = await self.process.stderr.read()
            error_msg = stderr_output.decode('utf-8', errors='ignore')[:500]
            log_print(f"FFmpeg进程立即退出 (返回码: {self.process.returncode})")
            log_print(f"错误输出: {error_msg}")

            if self.use_hardware_accel and not self.fallback_to_software:
                log_print("硬件编码器可能不兼容，尝试回退到软件编码...")
                self.fallback_to_software = True
                self.process = None
                # Restart with the software encoder.
                return await self.start()
            else:
                raise RuntimeError(f"FFmpeg进程启动失败: {error_msg}")

        # Start draining the process output.
        self._reader_task = asyncio.create_task(self._read_output())

    async def _read_output(self):
        """Drain FFmpeg output so the pipes cannot fill and block the process."""
        if self.process is None:
            return

        async def read_stream(stream, prefix):
            """Log a stream line by line and flag encoder failures."""
            while not self._stopped:
                try:
                    line = await stream.readline()
                    if not line:
                        break
                    text = line.decode('utf-8', errors='ignore').strip()
                    if text:
                        log_print(f"[{prefix}] {text}")
                        # Detect hardware encoder failures in the log.
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

        # Drain stdout and stderr concurrently.
        stdout_task = asyncio.create_task(read_stream(self.process.stdout, "FFmpeg-out"))
        stderr_task = asyncio.create_task(read_stream(self.process.stderr, "FFmpeg-err"))

        # Wait for both streams to finish.
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

    def _get_software_encoder_params(self):
        """Return the software encoder parameters."""
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
        """Return the hardware-accelerated encoder parameters."""
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
        """Write one frame to FFmpeg's stdin."""
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
            self.process.stdin.write(data)
            await self.process.stdin.drain()
            return True
        except BrokenPipeError:
            log_print("Error pushing frame: Broken pipe - FFmpeg process may have terminated")
            await self.close()

            # Switch to software encoding if hardware encoding failed.
            if self.use_hardware_accel and not self.fallback_to_software:
                log_print("硬件编码失败，尝试回退到软件编码...")
                self.fallback_to_software = True
                # Restart with the software encoder.
                await self.start()
                return await self.push_frame(frame)
            return False
        except Exception as e:
            log_print(f"Error pushing frame: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            await self.close()

            # Switch to software encoding if hardware encoding failed.
            if self.use_hardware_accel and not self.fallback_to_software:
                log_print("硬件编码失败，尝试回退到软件编码...")
                self.fallback_to_software = True
                # Restart with the software encoder.
                await self.start()
                # Retry the current frame.
                return await self.push_frame(frame)
            return False

    async def close(self):
        """Close the FFmpeg process and cancel its reader task."""
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
        """Return the background loop, creating it under lock if needed."""
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
        """Run an event loop on the background thread."""
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def start_sync(self):
        """Start FFmpeg synchronously using the background loop."""
        try:
            loop = self._get_or_create_loop()
            future = asyncio.run_coroutine_threadsafe(self.start(), loop)
            return future.result(timeout=10.0)
        except Exception as e:
            log_print(f"Error in start_sync: {e}")
            return False
    def push_frame_sync(self, frame):
        """Push a frame synchronously using the background loop."""
        try:
            loop = self._get_or_create_loop()
            future = asyncio.run_coroutine_threadsafe(
                self.push_frame(frame), loop
            )
            # Bound each push so a stalled process cannot block the caller.
            return future.result(timeout=5.0)
        except Exception as e:
            log_print(f"Error in push_frame_sync: {e}")
            return False

    def close_sync(self):
        """Close synchronously and tear down the background loop."""
        try:
            # Stop FFmpeg before tearing down the loop.
            if self._loop and not self._loop.is_closed():
                future = asyncio.run_coroutine_threadsafe(
                    self.close(), self._loop
                )
                try:
                    future.result(timeout=10.0)
                except Exception as e:
                    log_print(f"Error during close: {e}")

            # Stop the event loop.
            with self._loop_lock:
                if self._loop and not self._loop.is_closed():
                    self._loop.call_soon_threadsafe(self._loop.stop)
                    # Wait for the loop thread to exit.
                    if self._loop_thread and self._loop_thread.is_alive():
                        self._loop_thread.join(timeout=2.0)
                    self._loop.close()
                    self._loop = None
                    self._loop_thread = None
        except Exception as e:
            log_print(f"Error in close_sync: {e}")

    def __del__(self):
        """Best-effort cleanup when close_sync() was not called."""
        if self.process is not None and not self._stopped:
            log_print("Warning: FFmpegPusher not properly closed. Forcing cleanup.")
            # Async code cannot be awaited reliably here, so cleanup is best effort.
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
                else:
                    asyncio.run(self.close())
            except:
                pass