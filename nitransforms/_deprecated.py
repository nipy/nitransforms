# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Deprecation machinery, from NiBabel, against the version of NiTransforms."""

from functools import partial

from nibabel.deprecator import Deprecator
from nibabel.pkg_info import cmp_pkg_version

try:
    from ._version import __version__
except ModuleNotFoundError:
    __version__ = "0+unknown"

deprecate_with_version = Deprecator(
    partial(cmp_pkg_version, pkg_version_str=__version__)
)
