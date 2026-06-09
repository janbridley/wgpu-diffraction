import pytest
import numpy as np


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_gpu: requires a WebGPU device")


def pytest_collection_modifyitems(config, items):
    try:
        from wgpu_diffraction import sf3d

        p = np.zeros((2, 3), dtype=np.float32)
        k = np.array([[1, 0, 0]], dtype=np.float32)
        sf3d(p, k)
    except Exception:
        skip = pytest.mark.skip(reason="No WebGPU device available")
        for item in items:
            item.add_marker(skip)
