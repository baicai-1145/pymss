"""Lightweight plugin management shipped with pymss.

Importing this package does not import the separation runtime. Hosts can inspect
their plugin store even when model dependencies cannot be initialized.
"""

from .service import PluginContext, PluginPolicy, PluginService

API_VERSION = 1

__all__ = ["API_VERSION", "PluginContext", "PluginPolicy", "PluginService"]
