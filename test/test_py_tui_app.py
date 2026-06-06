import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
PY_TUI_PATH = os.path.join(ROOT, "py-tui")
sys.path.insert(0, PY_TUI_PATH)

pytest.importorskip("textual")

from logician_tui.app import LogicianTUI
from logician_tui.widgets.input_composer import InputComposer, SlashCommand
from logician_tui.widgets.message_row import MessageRow


def test_update_status_bar_no_status_bar_does_not_crash() -> None:
    app = LogicianTUI(python_cmd="python3")
    app._status_bar = None

    # Should not raise if the status bar is not yet mounted.
    app._update_status_bar()


def test_show_error_screen_assigns_status_bar(monkeypatch) -> None:
    from unittest.mock import patch, MagicMock
    from logician_tui.widgets.transcript import MessageTranscript

    app = LogicianTUI(python_cmd="python3")
    app._status_bar = None

    # Mock Screen and Container in the app module so we don't need
    # an active Textual app context. Also mock push_screen and mount
    # to avoid Textual's widget lifecycle requirements.
    mock_screen = MagicMock()
    mock_container = MagicMock()

    def _noop_mount(self, *widgets, **kwargs):
        pass

    with patch("logician_tui.app.Screen", return_value=mock_screen), \
         patch("logician_tui.app.Container", return_value=mock_container), \
         patch.object(app, "push_screen"), \
         patch.object(MessageTranscript, "mount", _noop_mount):
        # Should not crash and should assign a status bar reference.
        app._show_error_screen("Bridge failed")

    assert app._status_bar is not None


def test_message_row_render_returns_text() -> None:
    msg = MessageRow(role="assistant", text="Hello world")
    rendered = msg.render()

    assert "assistant" in rendered
    assert "Hello world" in rendered


def test_show_slash_popup_markup_without_color(monkeypatch) -> None:
    composer = InputComposer()
    composer._slash_commands = [
        SlashCommand(command="/help", usage="/help", description="Show help", dispatch="local"),
        SlashCommand(command="/version", usage="/version", description="Version info", dispatch="local"),
    ]
    composer._slash_selected = 0

    class DummyPopup:
        def __init__(self) -> None:
            self.text = ""

        def set_class(self, *_args, **_kwargs) -> None:
            pass

        def update(self, text: str) -> None:
            self.text = text

    popup = DummyPopup()
    monkeypatch.setattr(composer, "query_one", lambda selector, *args, **kwargs: popup)

    composer._show_slash_popup("/")

    assert "[]•" not in popup.text
    assert "[/]" not in popup.text
    assert "• /help" in popup.text


def test_input_submitted_clears_and_focuses(monkeypatch) -> None:
    composer = InputComposer()
    composer._slash_commands = []

    class DummyInput:
        def __init__(self):
            self.value = "hello"
            self.cursor_position = 5
            self.focused = False

        def focus(self) -> None:
            self.focused = True

    class DummyPopup:
        def __init__(self):
            self.hidden = False

        def set_class(self, visible: bool, _class: str) -> None:
            self.hidden = not visible

    input_widget = DummyInput()
    popup_widget = DummyPopup()

    def query_one(selector, *args, **kwargs):
        if selector == "#slash-popup":
            return popup_widget
        if selector == "#input-area":
            return input_widget
        raise RuntimeError(selector)

    monkeypatch.setattr(composer, "query_one", query_one)
    posted = []
    composer.post_message = lambda message: posted.append(message)

    composer.on_input_submitted(type("E", (), {"value": "hello"})())

    assert len(posted) == 1
    assert posted[0].value == "hello"
    assert input_widget.value == ""
    assert input_widget.cursor_position == 0
    assert input_widget.focused is True
    assert popup_widget.hidden is True
