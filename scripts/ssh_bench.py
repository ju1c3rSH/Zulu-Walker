# -*- coding: utf-8 -*-
"""SSH helper for MaixCAM bench debugging (paramiko).

设备端 sshd 对 exec 通道的 EOF/exit-status 上报不可靠，
所以这里"尽力读"：读到 exit status 或超时为止，把手头拿到的都返回。
"""
import sys
import time
import paramiko

HOST, USER, PWD = "192.168.0.101", "root", "root"


def run(cmd: str, timeout: int = 30) -> str:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, 22, USER, PWD, timeout=8, banner_timeout=15)
    try:
        ch = cli.get_transport().open_session()
        ch.settimeout(2.0)
        ch.exec_command(cmd)
        buf = []
        t0 = time.time()
        rc = None
        while time.time() - t0 < timeout:
            while ch.recv_ready():
                buf.append(ch.recv(65536).decode("utf-8", "replace"))
            while ch.recv_stderr_ready():
                buf.append(ch.recv_stderr(65536).decode("utf-8", "replace"))
            if ch.exit_status_ready():
                rc = ch.recv_exit_status()
                # status 就绪后再把残余读完
                time.sleep(0.3)
                while ch.recv_ready():
                    buf.append(ch.recv(65536).decode("utf-8", "replace"))
                while ch.recv_stderr_ready():
                    buf.append(ch.recv_stderr(65536).decode("utf-8", "replace"))
                break
            time.sleep(0.2)
        out = "".join(buf)
        return f"$ {cmd}\n[rc={rc if rc is not None else 'timeout'}]\n{out}"
    finally:
        cli.close()


if __name__ == "__main__":
    cmd = sys.argv[1]
    t = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    print(run(cmd, t))
