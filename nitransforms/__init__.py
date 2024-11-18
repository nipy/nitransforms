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
from .resampling import apply

try:
    from ._version import __version__
except ModuleNotFoundError:
    __version__ = "0+unknown"

__packagename__ = "nitransforms"
__copyright__ = "Copyright (c) 2021 The NiPy developers"

__all__ = [
    "apply",
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
