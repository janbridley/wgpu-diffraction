import pytest
import numpy as np


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_gpu: requires an OpenCL GPU device")


def pytest_collection_modifyitems(config, items):
    try:
        import glsf

        p = np.zeros((2, 3), dtype=np.float32)
        k = np.array([[1, 0, 0]], dtype=np.float32)
        glsf.sf3d(p, k)
    except Exception:
        skip = pytest.mark.skip(reason="No OpenCL GPU device available")
        for item in items:
            item.add_marker(skip)
