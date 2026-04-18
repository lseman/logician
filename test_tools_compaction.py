"""Tests for tool compaction and plugin cache modules."""

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from src.tools.compaction import (
    ContentReplacementState,
    FileBasedContentReplacementState,
    compact_result,
    DEFAULT_MAX_INLINE_CHARS,
)
from src.tools.plugin_cache import (
    PluginCacheManager,
    PluginInstallationEntry,
    PluginInfo,
    create_plugin_id,
    parse_plugin_id,
    migrate_legacy_cache,
)


class TestContentReplacementState:
    """Tests for ContentReplacementState."""

    def test_register_compaction(self):
        """Test registering a compacted result."""
        state = ContentReplacementState()
        state.register_compaction(
            result_id="test_123",
            file_path="/tmp/test.txt",
            preview_text="First 5000 chars...",
            original_size=100000,
        )

        assert state.has_replacement("test_123")
        assert state.get_replacement("test_123") is not None
        assert state.get_preview_text("test_123") == "First 5000 chars..."

    def test_get_replacement_nonexistent(self):
        """Test getting replacement for nonexistent ID."""
        state = ContentReplacementState()
        assert not state.has_replacement("nonexistent")
        assert state.get_replacement("nonexistent") is None
        assert state.get_preview_text("nonexistent") is None

    def test_clear_all(self):
        """Test clearing all replacements."""
        state = ContentReplacementState()
        state.register_compaction(
            result_id="test_123",
            file_path="/tmp/test.txt",
            preview_text="preview",
            original_size=100,
        )
        state.clear()

        assert not state.has_replacement("test_123")
        assert len(state.replacements) == 0

    def test_clear_specific(self):
        """Test clearing a specific replacement."""
        state = ContentReplacementState()
        state.register_compaction(
            result_id="test_123",
            file_path="/tmp/test.txt",
            preview_text="preview",
            original_size=100,
        )
        state.register_compaction(
            result_id="test_456",
            file_path="/tmp/test2.txt",
            preview_text="preview2",
            original_size=200,
        )
        state.clear("test_123")

        assert not state.has_replacement("test_123")
        assert state.has_replacement("test_456")

    def test_total_compacted_bytes(self):
        """Test total_compacted_bytes tracking."""
        state = ContentReplacementState()
        state.register_compaction(
            result_id="test_1",
            file_path="/tmp/test1.txt",
            preview_text="preview1",
            original_size=1000,
        )

        assert state.total_compacted_bytes == 0  # Not updated until file exists

        # Create the file
        Path("/tmp/test1.txt").write_text("x" * 1000)

        # Now the size should be updated
        assert state.total_compacted_bytes == 1000


class TestFileBasedContentReplacementState:
    """Tests for FileBasedContentReplacementState."""

    def setup_method(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "compaction_state.json"

    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_save_and_load(self):
        """Test saving and loading state from disk."""
        state = FileBasedContentReplacementState(str(self.storage_path))
        state.register_compaction(
            result_id="test_1",
            file_path="/tmp/test.txt",
            preview_text="preview",
            original_size=100,
        )

        state.save()
        state.clear()  # Clear in-memory

        # Load from disk
        state2 = FileBasedContentReplacementState(str(self.storage_path))
        state2.load()

        assert state2.has_replacement("test_1")

    def test_load_nonexistent(self):
        """Test loading nonexistent file."""
        state = FileBasedContentReplacementState(str(self.storage_path))
        state.load()

        assert len(state.replacements) == 0

    def test_save_creates_parent_dirs(self):
        """Test that save creates parent directories."""
        # Remove storage path if it exists
        if self.storage_path.exists():
            self.storage_path.unlink()

        state = FileBasedContentReplacementState(str(self.storage_path))
        state.register_compaction(
            result_id="test_1",
            file_path="/tmp/test.txt",
            preview_text="preview",
            original_size=100,
        )
        state.save()

        assert self.storage_path.exists()


class TestCompactResult:
    """Tests for compact_result function."""

    def setup_method(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_compact_large_result(self):
        """Test compacting a result that exceeds max_chars."""
        large_text = "x" * (DEFAULT_MAX_INLINE_CHARS + 1000)
        result, metadata = compact_result(large_text)

        assert result.startswith("[COMPACTED:")
        assert metadata is not None
        assert "file_path" in metadata
        assert "preview" in metadata
        assert "original_size" in metadata

    def test_inline_small_result(self):
        """Test that small results are returned inline."""
        small_text = "x" * 100
        result, metadata = compact_result(small_text)

        assert result == small_text
        assert metadata is None

    def test_inline_small_dict_preserves_structure(self):
        """Test that small structured results stay structured."""
        payload = {"status": "ok", "value": "x" * 10}
        result, metadata = compact_result(payload)

        assert result == payload
        assert metadata is None

    def test_compact_result_metadata(self):
        """Test that compacted results have correct metadata."""
        large_text = "x" * 150000
        result, metadata = compact_result(large_text)

        assert metadata["original_size"] == 150000
        assert len(metadata["preview"]) <= 5000


class TestPluginCacheManager:
    """Tests for PluginCacheManager."""

    def setup_method(self):
        """Set up temporary cache directory."""
        self.cache_dir = Path(tempfile.mkdtemp())
        self.manager = PluginCacheManager(str(self.cache_dir))

    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.cache_dir)

    def test_register_plugin(self):
        """Test registering a plugin."""
        self.manager.register_plugin(
            plugin_id="test-plugin@marketplace",
            scope="user",
            install_path="/tmp/test-plugin",
            version="1.0.0",
        )

        installations = self.manager.get_installed_plugins()
        assert "test-plugin@marketplace" in installations
        assert len(installations["test-plugin@marketplace"]) == 1

    def test_register_plugin_creates_cache(self):
        """Test that registering a plugin creates cache directory."""
        self.manager.register_plugin(
            plugin_id="test-plugin@marketplace",
            scope="user",
            install_path="/tmp/test-plugin",
            version="1.0.0",
        )

        cache_path = self.manager.get_versioned_cache_path(
            "test-plugin@marketplace", "1.0.0"
        )
        assert cache_path.exists()

    def test_unregister_plugin(self):
        """Test unregistering a plugin."""
        self.manager.register_plugin(
            plugin_id="test-plugin@marketplace",
            scope="user",
            install_path="/tmp/test-plugin",
            version="1.0.0",
        )

        assert self.manager.is_plugin_installed("test-plugin@marketplace")
        self.manager.unregister_plugin("test-plugin@marketplace", "user")
        assert not self.manager.is_plugin_installed("test-plugin@marketplace")

    def test_is_plugin_globally_installed(self):
        """Test is_plugin_globally_installed."""
        self.manager.register_plugin(
            plugin_id="test-plugin@marketplace",
            scope="user",
            install_path="/tmp/test-plugin",
            version="1.0.0",
        )

        assert self.manager.is_plugin_globally_installed("test-plugin@marketplace")

    def test_parse_plugin_id(self):
        """Test parsing plugin ID."""
        name, marketplace = parse_plugin_id("plugin@marketplace")
        assert name == "plugin"
        assert marketplace == "marketplace"

        # Handle local plugins
        name, marketplace = parse_plugin_id("local-plugin")
        assert name == "local-plugin"
        assert marketplace == "local"


class TestPluginInfo:
    """Tests for PluginInfo dataclass."""

    def test_plugin_info_to_dict(self):
        """Test PluginInfo to_dict conversion."""
        info = PluginInfo(
            name="test-plugin",
            version="1.0.0",
            description="Test plugin",
            author="Test Author",
        )

        result = info.to_dict()
        assert result["name"] == "test-plugin"
        assert result["version"] == "1.0.0"
        assert result["description"] == "Test plugin"
        assert result["author"] == "Test Author"


class TestPluginInstallationEntry:
    """Tests for PluginInstallationEntry dataclass."""

    def test_entry_to_dict(self):
        """Test PluginInstallationEntry to_dict conversion."""
        entry = PluginInstallationEntry(
            scope="user",
            install_path="/tmp/test",
            version="1.0.0",
            installed_at="2026-01-01T00:00:00Z",
            git_commit_sha="abc123",
            project_path="/tmp/project",
        )

        result = entry.to_dict()
        assert result["scope"] == "user"
        assert result["install_path"] == "/tmp/test"
        assert result["version"] == "1.0.0"
        assert result["git_commit_sha"] == "abc123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
