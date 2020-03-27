"""Prepare package for distribution."""
from pathlib import Path
from setuptools import setup
from toml import loads

if __name__ == "__main__":
    scm_cfg = loads(
        (Path(__file__).parent / "pyproject.toml").read_text()
    )["tool"]["setuptools_scm"]
    setup(
        name="nitransforms",
        use_scm_version=scm_cfg,
    )
