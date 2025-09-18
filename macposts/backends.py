"""Array backend management for macposts.

This module centralizes the logic for selecting the array library used by the
Python helper utilities. It allows macposts to take advantage of GPU capable
libraries such as CuPy when available while keeping CPU based NumPy as a
fallback.

The default behaviour honours a couple of environment variables:

``MACPOSTS_BACKEND``
    Can be set to ``"numpy"``/``"cpu"`` or ``"cupy"``/``"gpu"`` to force a
    backend. The special value ``"auto"`` (default) will pick the CPU backend
    unless GPU acceleration is explicitly preferred.

``MACPOSTS_PREFER_GPU``
    When ``MACPOSTS_BACKEND`` is ``"auto"`` a truthy value for this variable
    (``1``, ``true``, ``yes`` or ``on``) will attempt to use the GPU backend and
    fall back to the CPU backend if that fails.

``MACPOSTS_RETURN_DEVICE_ARRAYS``
    When truthy, skip converting GPU arrays back to NumPy arrays. This keeps the
    arrays on the GPU for downstream processing. The flag is ignored when the
    selected backend does not support GPUs.

Users can also control the backend at runtime via :func:`configure_array_backend`
or the :func:`use_array_backend` context manager.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import threading
import warnings
from dataclasses import dataclass
from types import ModuleType
from typing import Iterator, Optional

__all__ = [
    "ArrayBackend",
    "BackendState",
    "BackendUnavailable",
    "configure_array_backend",
    "get_array_module",
    "get_backend_state",
    "is_gpu_enabled",
    "reset_array_backend",
    "to_device_array",
    "to_output_array",
    "use_array_backend",
]


@dataclass(frozen=True)
class ArrayBackend:
    """Container for information about an array backend."""

    name: str
    module: ModuleType
    uses_gpu: bool


@dataclass(frozen=True)
class BackendState:
    """Runtime configuration for array processing."""

    backend: ArrayBackend
    return_device_arrays: bool


class BackendUnavailable(RuntimeError):
    """Raised when the requested backend cannot be loaded."""


_backend_lock = threading.RLock()
_STATE: BackendState


def _env_flag(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_backend(name: str) -> ArrayBackend:
    normalized = (name or "").strip().lower()
    try:
        if normalized in {"numpy", "cpu", "np"}:
            module = importlib.import_module("numpy")
            return ArrayBackend("numpy", module, False)
        if normalized in {"cupy", "gpu"}:
            module = importlib.import_module("cupy")
            return ArrayBackend("cupy", module, True)
    except ImportError as exc:  # pragma: no cover - tested via public API
        raise BackendUnavailable(str(exc)) from exc
    raise ValueError(f"Unknown backend: {name!r}")


def _select_state(
    name: str = "auto",
    return_device_arrays: Optional[bool] = None,
    *,
    prefer_gpu: Optional[bool] = None,
    from_env: bool = False,
) -> BackendState:
    normalized = (name or "auto").strip().lower()
    backend: ArrayBackend
    if normalized in {"auto", ""}:
        prefer_gpu = bool(prefer_gpu) or _env_flag("MACPOSTS_PREFER_GPU")
        if prefer_gpu:
            try:
                backend = _load_backend("cupy")
            except BackendUnavailable as exc:
                if from_env:
                    warnings.warn(
                        "GPU backend requested but CuPy is not available; using NumPy",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                backend = _load_backend("numpy")
        else:
            backend = _load_backend("numpy")
    else:
        backend = _load_backend(normalized)

    if return_device_arrays is None:
        # Default to host arrays to preserve historical behaviour.
        return_device_arrays = False

    if return_device_arrays and not backend.uses_gpu:
        warnings.warn(
            "return_device_arrays requested but the active backend does not "
            "use a GPU; keeping NumPy arrays instead",
            RuntimeWarning,
            stacklevel=2,
        )
        return_device_arrays = False

    return BackendState(backend=backend, return_device_arrays=return_device_arrays)


def _initial_state() -> BackendState:
    backend_name = os.getenv("MACPOSTS_BACKEND", "auto")
    return_device_arrays = _env_flag("MACPOSTS_RETURN_DEVICE_ARRAYS")
    return _select_state(
        backend_name,
        return_device_arrays=return_device_arrays,
        prefer_gpu=None,
        from_env=True,
    )


def _set_state(state: BackendState) -> BackendState:
    global _STATE
    with _backend_lock:
        _STATE = state
    return state


_set_state(_initial_state())


def get_backend_state() -> BackendState:
    """Return the current backend configuration."""

    return _STATE


def get_array_module() -> ModuleType:
    """Return the array module (NumPy/CuPy) for the active backend."""

    return get_backend_state().backend.module


def is_gpu_enabled() -> bool:
    """Return whether the current backend uses a GPU."""

    return get_backend_state().backend.uses_gpu


def configure_array_backend(
    name: str = "auto",
    *,
    prefer_gpu: Optional[bool] = None,
    return_device_arrays: Optional[bool] = None,
) -> BackendState:
    """Select a new array backend.

    Parameters
    ----------
    name:
        Backend identifier. ``"numpy"``/``"cpu"`` forces the CPU backend,
        ``"cupy"``/``"gpu"`` forces the GPU backend and ``"auto"`` selects
        automatically (default).
    prefer_gpu:
        When ``name`` is ``"auto"`` a truthy value will attempt to use the GPU
        backend first and fall back to the CPU backend if it is unavailable.
    return_device_arrays:
        When ``True`` and the selected backend supports GPUs, arrays produced by
        helper functions will remain on the device.
    """

    state = _select_state(
        name,
        return_device_arrays=return_device_arrays,
        prefer_gpu=prefer_gpu,
        from_env=False,
    )
    return _set_state(state)


def reset_array_backend() -> BackendState:
    """Reset the backend configuration using the environment defaults."""

    return _set_state(_initial_state())


@contextlib.contextmanager
def use_array_backend(
    name: str = "auto",
    *,
    prefer_gpu: Optional[bool] = None,
    return_device_arrays: Optional[bool] = None,
) -> Iterator[BackendState]:
    """Context manager to temporarily switch the array backend."""

    previous = get_backend_state()
    state = configure_array_backend(
        name,
        prefer_gpu=prefer_gpu,
        return_device_arrays=return_device_arrays,
    )
    try:
        yield state
    finally:
        _set_state(previous)


def to_device_array(obj, *, dtype=None, copy: bool = False):
    """Convert *obj* to the active backend array type."""

    xp = get_array_module()
    return xp.array(obj, dtype=dtype, copy=copy)


def to_output_array(array, state: Optional[BackendState] = None):
    """Convert *array* to the appropriate output type for the backend."""

    state = state or get_backend_state()
    backend = state.backend
    xp = backend.module

    if backend.uses_gpu and not state.return_device_arrays:
        if hasattr(xp, "asnumpy") and isinstance(array, xp.ndarray):
            return xp.asnumpy(array)
    return array
