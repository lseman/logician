import sys
import subprocess
try:
    subprocess.run(["echo", "hello"], pass_fds=[-1])
except Exception as e:
    print("Caught:", type(e), e)
