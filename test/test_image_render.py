"""Tests for image auto-render config in logician_bridge.py."""
from logician_bridge import _should_emit_image


class TestImageAutoRenderConfig:
    """Test image auto-render config defaults and toggles."""

    def test_default_is_disabled(self):
        """Image rendering should be disabled by default."""
        assert _should_emit_image({}) is False

    def test_explicit_false(self):
        assert _should_emit_image({"image_auto_render_enabled": False}) is False

    def test_explicit_true(self):
        assert _should_emit_image({"image_auto_render_enabled": True}) is True

    def test_zero_is_false(self):
        assert _should_emit_image({"image_auto_render_enabled": 0}) is False

    def test_nonzero_is_true(self):
        assert _should_emit_image({"image_auto_render_enabled": 1}) is True

    def test_other_keys_ignored(self):
        """Other config keys should not affect image setting."""
        cfg = {
            "prompt_rag_context_enabled": False,
            "thinking_level": "high",
            "tool_cache_enabled": True,
        }
        assert _should_emit_image(cfg) is False
