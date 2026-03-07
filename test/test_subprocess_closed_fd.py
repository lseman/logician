import sys, os, subprocess

def test_closed_fd(fd_to_close):
    # duplicate the fd to restore later if needed, but here we just exit
    try:
        os.close(fd_to_close)
    except:
        pass
    try:
        subprocess.run(["echo", "hello"], capture_output=True)
        print(f"FD {fd_to_close} OK")
    except Exception as e:
        print(f"FD {fd_to_close} Error: {e}")

test_closed_fd(0)
test_closed_fd(1)
test_closed_fd(2)
