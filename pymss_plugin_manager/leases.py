"""Cross-process execution leases; lock files deliberately outlive their owners."""
from __future__ import annotations

import atexit
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import os
from pathlib import Path
import threading
import time

from .locks import RuntimeLockBusy

_HELD: ContextVar[dict | None] = ContextVar("plugin_binding_leases", default=None)
_PROCESS_LEASES: dict[tuple[int, str], object] = {}
_PROCESS_GUARD = threading.RLock()
_CANDIDATES: ContextVar[frozenset[str]] = ContextVar("plugin_candidate_probes", default=frozenset())


def is_candidate(root: Path):
    return os.path.normcase(str(Path(root).resolve())) in _CANDIDATES.get()


@contextmanager
def candidate_snapshot(root: Path):
    """Used only by the supervised child whose parent holds the store lease."""
    token = _CANDIDATES.set(_CANDIDATES.get() | {os.path.normcase(str(Path(root).resolve()))})
    try:
        yield
    finally:
        _CANDIDATES.reset(token)


def _owner():
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return os.getpid(), threading.get_ident(), id(task)


@contextmanager
def binding_lease(root: Path, *, exclusive: bool = False, _track_context: bool = True,
                  timeout: float = 0, cancelled=lambda: False):
    key = os.path.normcase(str(Path(root).resolve()))
    held = (_HELD.get() or {}) if _track_context else {}
    existing = held.get(key)
    if existing is not None and existing[1] == _owner():
        if exclusive and not existing[0]:
            raise RuntimeLockBusy("PLUGIN_STORE_BUSY: This execution is using the plugin binding")
        yield
        return
    root = Path(root).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    path = root.parent / f".{root.name}.execution.lock"
    with path.open("a+b") as stream:
        deadline = time.monotonic() + timeout

        def retry():
            if cancelled():
                raise InterruptedError("Plugin management was cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeLockBusy("PLUGIN_STORE_BUSY: The plugin binding is in use")
            time.sleep(min(.05, remaining))

        if os.name == "nt":
            import ctypes
            import msvcrt
            from ctypes import wintypes

            class Overlapped(ctypes.Structure):
                _fields_ = [("internal", ctypes.c_size_t), ("internal_high", ctypes.c_size_t),
                            ("offset", wintypes.DWORD), ("offset_high", wintypes.DWORD),
                            ("event", wintypes.HANDLE)]

            api = ctypes.WinDLL("kernel32", use_last_error=True)
            api.LockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                      wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(Overlapped)]
            api.LockFileEx.restype = wintypes.BOOL
            api.UnlockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                        wintypes.DWORD, ctypes.POINTER(Overlapped)]
            api.UnlockFileEx.restype = wintypes.BOOL
            handle = msvcrt.get_osfhandle(stream.fileno())
            overlapped = Overlapped()
            while not api.LockFileEx(handle, 1 | (2 if exclusive else 0), 0, 1, 0, ctypes.byref(overlapped)):
                error = ctypes.get_last_error()
                if error in {32, 33, 158}:
                    retry()
                else:
                    raise ctypes.WinError(error)

            def unlock():
                api.UnlockFileEx(handle, 0, 1, 0, ctypes.byref(overlapped))
        else:
            import fcntl
            while True:
                try:
                    fcntl.flock(stream, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    retry()

            def unlock():
                fcntl.flock(stream, fcntl.LOCK_UN)
        token = _HELD.set({**held, key: (exclusive, _owner())}) if _track_context else None
        try:
            yield
        finally:
            try:
                if token is not None:
                    _HELD.reset(token)
            except ValueError:
                # Process leases can be released by atexit on another thread.
                pass
            finally:
                unlock()


def hold_runtime(root: Path):
    key = os.getpid(), os.path.normcase(str(Path(root).resolve()))
    with _PROCESS_GUARD:
        if key not in _PROCESS_LEASES:
            lease = binding_lease(root, _track_context=False)
            lease.__enter__()
            _PROCESS_LEASES[key] = lease
            return True
        return False


def release_binding(root: Path):
    key = os.getpid(), os.path.normcase(str(Path(root).resolve()))
    with _PROCESS_GUARD:
        lease = _PROCESS_LEASES.pop(key, None)
        if lease is not None:
            lease.__exit__(None, None, None)


def runtime_roots():
    with _PROCESS_GUARD:
        return [root for (pid, root) in _PROCESS_LEASES if pid == os.getpid()]


def release_runtime():
    with _PROCESS_GUARD:
        for key, lease in list(_PROCESS_LEASES.items())[::-1]:
            if key[0] == os.getpid():
                del _PROCESS_LEASES[key]
                lease.__exit__(None, None, None)


atexit.register(release_runtime)
