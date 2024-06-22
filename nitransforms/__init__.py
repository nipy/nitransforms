# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Geometric transforms.

.. currentmodule:: nitransforms

.. autosummary::
   :toctree: ../generated

   transform
"""
from . import linear, manip, nonlinear, surface
from .linear import Affine, LinearTransformsMapping
from .nonlinear import DenseFieldTransform
from .manip import TransformChain

try:
    from ._version import __version__
except ModuleNotFoundError:
    from pkg_resources import get_distribution, DistributionNotFound

    try:
        __version__ = get_distribution("nitransforms").version
    except DistributionNotFound:
        __version__ = "unknown"
    del get_distribution
    del DistributionNotFound

__packagename__ = "nitransforms"
__copyright__ = "Copyright (c) 2021 The NiPy developers"

__all__ = [
    "surface",
    "linear",
    "manip",
    "nonlinear",
    "Affine",
    "LinearTransformsMapping",
    "DenseFieldTransform",
    "TransformChain",
    "__copyright__",
    "__packagename__",
    "__version__",
]
