"""Test _version.py."""
import sys
from collections import namedtuple
from pkg_resources import DistributionNotFound
from importlib import reload
import nitransforms


def test_version_scm0(monkeypatch):
    """Retrieve the version via setuptools_scm."""
    class _version:
        __version__ = "10.0.0"

    monkeypatch.setitem(sys.modules, 'nitransforms._version', _version)
    reload(nitransforms)
    assert nitransforms.__version__ == "10.0.0"


def test_version_scm1(monkeypatch):
    """Retrieve the version via pkg_resources."""
    monkeypatch.setitem(sys.modules, 'nitransforms._version', None)

    def _dist(name):
        Distribution = namedtuple("Distribution", ["name", "version"])
        return Distribution(name, "success")

    monkeypatch.setattr('pkg_resources.get_distribution', _dist)
    reload(nitransforms)
    assert nitransforms.__version__ == "success"


def test_version_scm2(monkeypatch):
    """Check version could not be interpolated."""
    monkeypatch.setitem(sys.modules, 'nitransforms._version', None)

    def _raise(name):
        raise DistributionNotFound("No get_distribution mock")

    monkeypatch.setattr('pkg_resources.get_distribution', _raise)
    reload(nitransforms)
    assert nitransforms.__version__ == "unknown"
