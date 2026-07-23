import fcntl
import json
import os
import pty
import select
import signal
import struct
import sys
import termios
import time


def drain(master, output, duration):
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        ready, _, _ = select.select([master], [], [], min(0.05, deadline - time.monotonic()))
        if not ready:
            continue
        try:
            chunk = os.read(master, 65536)
        except OSError:
            return False
        if not chunk:
            return False
        output.extend(chunk)
    return True


config = json.load(sys.stdin)
pid, master = pty.fork()
if pid == 0:
    os.chdir(config["cwd"])
    os.execvpe(config["command"], [config["command"], *config.get("args", [])], config["env"])

columns = int(config.get("columns", 100))
rows = int(config.get("rows", 30))
fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))
output = bytearray()
start = time.monotonic()
timeout = int(config.get("timeoutMs", 5000)) / 1000

try:
    drain(master, output, 0.8)
    for action in config.get("actions", []):
        drain(master, output, int(action.get("afterMs", 0)) / 1000)
        os.write(master, action.get("send", "").encode("utf-8"))
    remaining = max(0.1, timeout - (time.monotonic() - start))
    drain(master, output, min(remaining, 1.0))
finally:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    drain(master, output, 0.3)
    try:
        _, status = os.waitpid(pid, 0)
        exit_code = os.waitstatus_to_exitcode(status)
    except ChildProcessError:
        exit_code = None
    os.close(master)

print(json.dumps({
    "output": output.decode("utf-8", errors="replace"),
    "exitCode": exit_code,
}))
