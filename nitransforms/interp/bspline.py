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
from scipy.sparse import csr_matrix, kron


def _cubic_bspline(d, order=3):
    """Evaluate the cubic bspline at distance d from the center."""
    if order != 3:
        raise NotImplementedError

    return np.piecewise(
        d,
        [d < 1.0, d >= 1.0],
        [
            lambda d: (4.0 - 6.0 * d ** 2 + 3.0 * d ** 3) / 6.0,
            lambda d: (2.0 - d) ** 3 / 6.0,
        ],
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
    The 1D B-Spline kernel implementation uses :obj:`numpy.piecewise`, and is based on the
    closed-form given by Eq. (6) of [Unser1999]_.

    By iterating over dimensions, the data samples that fall outside of the compact
    support of the tensor-product kernel associated to each control point can be filtered
    out and dismissed to lighten computation.

    Finally, the resulting weights matrix :math:`\Psi^3(\mathbf{k}, \mathbf{s})`
    can be easily identified in Eq. :math:`\eqref{eq:1}` and used as the design matrix
    for approximation of data.

    Parameters
    ----------
    target_grid : :obj:`~nitransforms.base.ImageGrid` or :obj:`nibabel.spatialimages`
        Regular grid of :math:`N` locations at which tensor B-Spline basis will be evaluated.
    ctrl_grid : :obj:`~nitransforms.base.ImageGrid` or :obj:`nibabel.spatialimages`
        Regular grid of :math:`K` control points (knot) where B-Spline basis are defined.

    Returns
    -------
    weights : :obj:`numpy.ndarray` (:math:`K \times N`)
        A sparse matrix of interpolating weights :math:`\Psi^3(\mathbf{k}, \mathbf{s})`
        for the *N* voxels of the target EPI, for each of the total *K* knots.
        This sparse matrix can be directly used as design matrix for the fitting
        step of approximation/extrapolation.

    """
    shape = target_grid.shape[:3]
    ctrl_sp = nb.affines.voxel_sizes(ctrl_grid.affine)[:3]
    ras2ijk = np.linalg.inv(ctrl_grid.affine)
    # IJK index in the control point image of the first index in the target image
    origin = nb.affines.apply_affine(ras2ijk, [tuple(target_grid.affine[:3, 3])])[0]

    wd = []
    for i, (o, n, sp) in enumerate(
        zip(origin, shape, nb.affines.voxel_sizes(target_grid.affine)[:3])
    ):
        # Locations of voxels in target image in control point image
        locations = np.arange(0, n, dtype="float16") * sp / ctrl_sp[i] + o
        knots = np.arange(0, ctrl_grid.shape[i], dtype="float16")
        distance = np.abs(locations[np.newaxis, ...] - knots[..., np.newaxis])

        within_support = distance < 2.0
        d_vals, d_idxs = np.unique(distance[within_support], return_inverse=True)
        bs_w = _cubic_bspline(d_vals)
        weights = np.zeros_like(distance, dtype="float32")
        weights[within_support] = bs_w[d_idxs]
        wd.append(csr_matrix(weights))

    return kron(kron(wd[0], wd[1]), wd[2])
