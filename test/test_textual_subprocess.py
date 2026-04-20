import subprocess

from textual.app import App


class TestApp(App):
    def on_mount(self):
        with open("test_out.txt", "w") as f:
            try:
                subprocess.run(["echo", "hello"], capture_output=True, text=True)
                f.write("subprocess ok\n")
            except Exception as e:
                f.write(f"subprocess error: {e}\n")
        self.exit()

if __name__ == "__main__":
    TestApp().run()
