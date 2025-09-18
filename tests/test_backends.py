import importlib

import pytest

import macposts.backends as backends


@pytest.fixture(autouse=True)
def _restore_backend():
    previous = backends.get_backend_state()
    yield
    backends.configure_array_backend(
        previous.backend.name,
        return_device_arrays=previous.return_device_arrays,
    )


def test_configure_numpy_backend():
    state = backends.configure_array_backend("numpy")
    assert state.backend.name == "numpy"
    assert not state.backend.uses_gpu
    assert not state.return_device_arrays
    xp = backends.get_array_module()
    assert xp.__name__.startswith("numpy")


def test_return_device_arrays_flag_cpu():
    with pytest.warns(RuntimeWarning):
        state = backends.configure_array_backend(
            "numpy", return_device_arrays=True
        )
    assert not state.return_device_arrays


def test_use_array_backend_context_manager():
    base_state = backends.configure_array_backend("numpy")
    with backends.use_array_backend("numpy") as state:
        assert state.backend.name == "numpy"
        assert backends.get_backend_state() == state
    assert backends.get_backend_state() == base_state


def test_gpu_backend_missing(monkeypatch):
    original_import = backends.importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "cupy":
            raise ImportError("No module named 'cupy'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(backends.importlib, "import_module", fake_import)
    with pytest.raises(backends.BackendUnavailable):
        backends.configure_array_backend("gpu")


@pytest.mark.skipif(
    importlib.util.find_spec("cupy") is None, reason="CuPy is not available"
)
def test_gpu_backend_available():
    state = backends.configure_array_backend("gpu")
    assert state.backend.uses_gpu
    xp = backends.get_array_module()
    assert xp.__name__.startswith("cupy")
    # Keep device arrays when requested explicitly.
    state = backends.configure_array_backend("gpu", return_device_arrays=True)
    assert state.return_device_arrays
