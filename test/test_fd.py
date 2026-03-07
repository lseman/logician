import subprocess
import os
import sys

# Close fd 0 to simulate textual closing it
os.close(0)

try:
    subprocess.run(["echo", "hello"], capture_output=True, text=True)
    print("subprocess default OK")
except Exception as e:
    print("subprocess default Error:", e)

try:
    subprocess.run(["echo", "hello"], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    print("subprocess DEVNULL OK")
except Exception as e:
    print("subprocess DEVNULL Error:", e)
