"""Test _version.py."""

import sys
from importlib import reload
import nitransforms


def test_version_scm0(monkeypatch):
    """Retrieve the version via setuptools_scm."""

    class _version:
        __version__ = "10.0.0"

    monkeypatch.setitem(sys.modules, "nitransforms._version", _version)
    reload(nitransforms)
    assert nitransforms.__version__ == "10.0.0"


def test_version_fallback(monkeypatch):
    """Check version could not be interpolated."""
    monkeypatch.setitem(sys.modules, "nitransforms._version", None)

    reload(nitransforms)
    assert nitransforms.__version__ == "0+unknown"
