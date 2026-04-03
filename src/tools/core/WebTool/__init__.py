"""Package-style wrapper for web and package metadata tools."""

from .tool import fetch_url, github_read_file, pypi_info, web_search

__all__ = ["fetch_url", "github_read_file", "pypi_info", "web_search"]
