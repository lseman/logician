# -*- coding: utf-8 -*-
"""Plugin manager for logician — marketplace add, install, list, remove, update."""

from .manager import PluginManager
from .state import InstalledPluginsRegistry as PluginState

__all__ = ["PluginManager", "PluginState"]
