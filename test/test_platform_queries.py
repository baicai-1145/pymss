import os
import platform
import sys
from unittest import mock

from pymss.operations.command import current_target
from pymss_plugin_manager.context import environment_identity


def test_windows_plugin_identity_and_native_target_do_not_query_wmi():
    with mock.patch.object(sys, "platform", "win32"), \
            mock.patch.dict(os.environ, {"PROCESSOR_ARCHITECTURE": "AMD64", "PROCESSOR_ARCHITEW6432": ""}), \
            mock.patch.object(platform, "machine", side_effect=AssertionError("WMI queried")):
        assert environment_identity()["architecture"] == "amd64"
        assert current_target() == "win32-x86_64"


def test_windows_native_architecture_takes_precedence_over_emulated_process():
    with mock.patch.object(sys, "platform", "win32"), \
            mock.patch.dict(os.environ, {"PROCESSOR_ARCHITECTURE": "x86", "PROCESSOR_ARCHITEW6432": "ARM64"}), \
            mock.patch.object(platform, "machine", side_effect=AssertionError("WMI queried")):
        assert environment_identity()["architecture"] == "arm64"
        assert current_target() == "win32-arm64"


def test_posix_native_target_keeps_existing_architecture_mapping():
    for name, machine, expected in [("darwin", "arm64", "arm64"), ("linux", "aarch64", "arm64"), ("linux", "x86_64", "x86_64")]:
        with mock.patch.object(sys, "platform", name), mock.patch.object(platform, "machine", return_value=machine):
            assert environment_identity()["architecture"] == machine
            assert current_target() == f"{name}-{expected}"
