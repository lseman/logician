#!/usr/bin/env python3
"""
Test to reproduce the thinking message cropping issue.
"""

import subprocess
import time
import threading
import json
from typing import List, Optional


def start_bridge():
    """Start the logician bridge in background."""
    cmd = [
        "python3", "-m", "logician_bridge",
        "--log-level", "debug"
    ]
    print(f"Starting bridge with: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    return proc


def send_thinking_tokens(bridge_url: str, num_messages: int, tokens_per_message: int) -> None:
    """Send thinking tokens to the bridge."""
    import urllib.request
    import urllib.error

    for msg_idx in range(num_messages):
        print(f"Sending thinking message {msg_idx + 1}/{num_messages}")
        for tok_idx in range(tokens_per_message):
            tok = f"Token {tok_idx + 1} of message {msg_idx + 1} - thinking content"

            req = urllib.request.Request(
                bridge_url,
                data=json.dumps({
                    "type": "event",
                    "payload": {
                        "token": tok
                    }
                }).encode(),
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    resp.read()
            except Exception as e:
                print(f"Error sending token: {e}")

            time.sleep(0.01)  # Small delay between tokens


def main():
    print("Starting test for thinking message cropping issue...")

    # Start the bridge
    bridge_proc = start_bridge()

    # Give it time to start
    time.sleep(2)

    try:
        # Send multiple thinking messages
        num_messages = 3
        tokens_per_message = 20

        # Create a thread to send tokens
        def send_tokens():
            send_thinking_tokens(
                "http://localhost:8080/event",
                num_messages,
                tokens_per_message
            )

        send_thread = threading.Thread(target=send_tokens)
        send_thread.start()

        # Keep running for a bit to see the output
        time.sleep(10)
        send_thread.join(timeout=2)

        print("Test completed. Check the bridge output for issues.")

    except KeyboardInterrupt:
        print("Test interrupted")
    finally:
        bridge_proc.terminate()
        bridge_proc.wait()


if __name__ == "__main__":
    main()
