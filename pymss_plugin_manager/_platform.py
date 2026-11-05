"""Native architecture without Windows Management Instrumentation queries."""
import os
import platform
import sys


def machine() -> str:
    if sys.platform == "win32":
        return os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get("PROCESSOR_ARCHITECTURE", "")
    return platform.machine()
