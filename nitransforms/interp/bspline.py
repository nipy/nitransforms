# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Interpolate with 3D tensor-product B-Spline basis."""

import numpy as np
import nibabel as nb
from scipy.interpolate import NdBSpline
from scipy.sparse import coo_array, csr_array

from nitransforms._deprecated import deprecate_with_version


@deprecate_with_version(
    "grid_bspline_weights is deprecated. "
    "Please use `scipy.interpolate.NdBSpline.design_matrix` instead.",
    since="26.0.0",
    until="28.0.0",
)
def grid_bspline_weights(target_grid, ctrl_grid):
    r"""
    Evaluate tensor-product B-Spline weights on a grid.

    For each of the :math:`N` input locations :math:`\mathbf{x} = (x_i, x_j, x_k)`
    and :math:`K` control points or *knots* :math:`\mathbf{c} =(c_i, c_j, c_k)`,
    the tensor-product cubic B-Spline kernel weights are calculated:

    .. math::
        \Psi^3(\mathbf{x}, \mathbf{c}) =
        \beta^3(x_i - c_i) \cdot \beta^3(x_j - c_j) \cdot \beta^3(x_k - c_k),
        \label{eq:bspline_weights}\tag{1}

    where each :math:`\beta^3` represents the cubic B-Spline for one dimension.
    The resulting weights matrix :math:`\Psi^3(\mathbf{k}, \mathbf{s})` can be used
    as the design matrix for approximation of data.

    Parameters
    ----------
    target_grid : :class:`~nitransforms.base.ImageGrid` or :class:`~nibabel.spatialimages.SpatialImage`
        Regular grid of :math:`N` locations at which tensor B-Spline basis will be evaluated.
    ctrl_grid : :class:`~nitransforms.base.ImageGrid` or :class:`~nibabel.spatialimages.SpatialImage`
        Regular grid of :math:`K` control points (knot) where B-Spline basis are defined.

    Returns
    -------
    weights : :class:`scipy.sparse.coo_array` (:math:`K \times N`)
        A sparse matrix of interpolating weights :math:`\Psi^3(\mathbf{k}, \mathbf{s})`
        for the *N* voxels of the target EPI, for each of the total *K* knots.
        This sparse matrix can be directly used as design matrix for the fitting
        step of approximation/extrapolation.

    """  # noqa: E501
    shape = target_grid.shape[:3]
    ctrl_shape = ctrl_grid.shape[:3]
    # Target voxel indices, in the index space of the control points
    vox2vox = np.linalg.inv(ctrl_grid.affine) @ target_grid.affine
    ijk = nb.affines.apply_affine(vox2vox, np.indices(shape).reshape(3, -1).T)

    # Four zero coefficients per side keep the weights at 0 beyond the lattice,
    # and are dropped from the result.
    knots = tuple(np.arange(-6, n + 6, dtype="float64") for n in ctrl_shape)
    padded_shape = [n + 8 for n in ctrl_shape]
    padded = np.ravel_multi_index(
        np.indices(ctrl_shape).reshape(3, -1) + 4, padded_shape
    )

    weights = NdBSpline.design_matrix(ijk, knots, 3)
    # design_matrix stops at the last column with a nonzero weight
    weights = csr_array(
        (weights.data, weights.indices, weights.indptr),
        shape=(len(ijk), np.prod(padded_shape)),
    )
    return coo_array(weights.tocsc()[:, padded].T, dtype="float32")
