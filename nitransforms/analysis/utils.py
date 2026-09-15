# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities to aid in performing and evaluating image registration.

This module provides functions to compute displacements of image coordinates
under a transformation, useful for assessing the accuracy of image registration
processes.

References
----------
.. [Power2012] Power, JD. et al. (2012). "Spurious but systematic correlations in functional
   connectivity MRI networks arise from subject motion." NeuroImage, 59(3):2142-2154.
   doi:`10.1016/j.neuroimage.2011.10.018 <https://doi.org/10.1016/j.neuroimage.2011.10.018>`__.

"""

from __future__ import annotations


import math
from dataclasses import dataclass
from itertools import product

import nibabel as nb
import numpy as np
from scipy.spatial.transform import Rotation as R

from nitransforms.base import TransformBase
from nitransforms.linear import Affine, LinearTransformsMapping


DEFAULT_FD_RADIUS = 50.0
"""
Default radius (in mm) of a sphere where framewise displacements are calculated.
The choice was proposed by [Power2012]_, and it represents approximately the mean
distance from the cerebral cortex to the center of the head.
"""

MOTION_FORMAT_AFNI = "afni"
"""AFNI motion parameter format identifier."""
MOTION_FORMAT_FSL = "fsl"
"""FSL motion parameter format identifier."""

TRANSLATIONS_ERROR_MSG = "'translations' must have shape (T, 3)."
"""Translations shape error message"""
ROTATIONS_SHAPE_ERROR_MSG = "'rotations' must have shape (R, 3)."
"""Rotations shape error message."""
MOTION_PARAMS_SHAPE_ERROR_MSG = "'motion_parameters' must have shape (T, 6)."
"""Motion parameters shape error message"""
MOTION_PARAMS_FMT_ERROR_MSG = (
    "Unsupported motion parameter format: '{fmt}'. "
    f"Supported formats are: '{MOTION_FORMAT_AFNI}', '{MOTION_FORMAT_FSL}'."
)
"""Unsupported motion parameter format error message."""
MOTION_PARAMS_INST_ERROR_MSG = "'motion_parameters' must be a 'MotionParameters' instance."
"""Motion parameters instance error message."""

AFFINE_TYPE_ERROR_MSG = "Affine input must be a valid array, Affine, or LinearTransformsMapping."
"""Affine input type error message."""
AFFINE_SHAPE_ERROR = "Affine input must have shape (4, 4)."
"""Affine shape error message."""
AFFINE_SEQ_SHAPE_ERROR = "Affine input must have shape (4, 4) or (T, 4, 4)."
"""Affine sequence error message."""


@dataclass(frozen=True)
class MotionParameters:
    """Representation for translation and rotation motion parameters.

    Examples
    --------
    >>> import numpy as np
    >>> params = MotionParameters(
    ...     translations=np.zeros((3, 3)),
    ...     rotations=np.zeros((3, 3))
    ... )
    >>> params.translations.shape
    (3, 3)
    """

    translations: np.ndarray
    """Translational motion parameters with shape ``(T, 3)`` in mm."""
    rotations: np.ndarray
    """Rotational motion parameters with shape ``(T, 3)``."""

    def __post_init__(self):
        object.__setattr__(self, "translations", np.asarray(self.translations, dtype=float))
        object.__setattr__(self, "rotations", np.asarray(self.rotations, dtype=float))
        if self.translations.ndim != 2 or self.translations.shape[1] != 3:
            raise ValueError(TRANSLATIONS_ERROR_MSG)
        if self.rotations.ndim != 2 or self.rotations.shape[1] != 3:
            raise ValueError(ROTATIONS_SHAPE_ERROR_MSG)

    def __iter__(self):
        return iter((self.translations, self.rotations))


def extract_motion_parameters(
    motion_parameters: np.ndarray, fmt: str | None = None
) -> MotionParameters:
    """Extract translation and rotation parameters into :class:`MotionParameters`.

    Parameters
    ----------
    motion_parameters : :obj:`~numpy.ndarray`
        An ``(T, 6)`` array of motion parameters.
    fmt : :obj:`str`, optional
        Parameter format specification. Supported formats:

        - `:data:`~nitransforms.analysis.utils.MOTION_FORMAT_AFNI`:
          Assumes standard AFNI 6-column output where translations are
          in the first three columns (in mm) and rotations are in the
          subsequent three columns (converted from degrees to radians).
        - :data:`~nitransforms.analysis.utils.MOTION_FORMAT_FSL` (or :obj:`None`):
          Standard FSL-style 6-column format where translations are in
          the first three columns and rotations are in the last three
          columns.

    Returns
    -------
    :class:`~nitransforms.analysis.utils.MotionParameters`
        Structured translations and rotations.

    Raises
    ------
    :exc:`ValueError`
        If ``motion_parameters`` does not have shape ``(T, 6)`` or if ``fmt``
        is unrecognized.

    Examples
    --------
    >>> import numpy as np
    >>> raw_params = np.zeros((10, 6))
    >>> params = extract_motion_parameters(raw_params, fmt=MOTION_FORMAT_FSL)
    >>> params.translations.shape
    (10, 3)
    >>> params.rotations.shape
    (10, 3)

    >>> # Using AFNI format (rotations in degrees are converted to radians)
    >>> afni_params = np.array([[1.0, 2.0, 3.0, 180.0, 0.0, 0.0]])
    >>> params_afni = extract_motion_parameters(afni_params, fmt=MOTION_FORMAT_AFNI)
    >>> np.allclose(params_afni.rotations[0, 0], np.pi)
    True
    """
    arr = np.asarray(motion_parameters, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(MOTION_PARAMS_SHAPE_ERROR_MSG)

    translations = arr[:, :3]
    rotations = arr[:, 3:]

    if fmt is not None:
        fmt = fmt.lower()
        if fmt == MOTION_FORMAT_AFNI:
            # AFNI outputs rotations in degrees; convert to radians for
            # internal consistency
            rotations = np.deg2rad(rotations)
        elif fmt == MOTION_FORMAT_FSL:
            pass
        else:
            raise ValueError(MOTION_PARAMS_FMT_ERROR_MSG.format(fmt=fmt))

    return MotionParameters(translations=translations, rotations=rotations)


def affine_to_motion_params(
    affine: Affine | LinearTransformsMapping | np.ndarray
) -> MotionParameters:
    """Convert an affine transformation or sequence of transformations into a :class:`MotionParameters`.

    Parameters
    ----------
    affine : :obj:`~nitransforms.linear.Affine`, :obj:`~nitransforms.linear.LinearTransformsMapping`, or :obj:`~numpy.ndarray`
        A single ``(4, 4)`` affine matrix, an :class:`~nitransforms.linear.Affine`
        instance, a :class:`~nitransforms.linear.LinearTransformsMapping` sequence,
        or a batch array of shape ``(T, 4, 4)``.

    Returns
    -------
    :class:`~nitransforms.analysis.utils.MotionParameters`
        Structured translations and rotations.

    Examples
    --------
    >>> import numpy as np
    >>> # Convert a single 4x4 affine matrix
    >>> mat = np.eye(4)
    >>> params = affine_to_motion_params(mat)
    >>> params.translations.shape
    (1, 3)

    >>> # Convert a batch of 3 affine matrices (shape: T, 4, 4)
    >>> batch_mat = np.tile(np.eye(4), (3, 1, 1))
    >>> params_batch = affine_to_motion_params(batch_mat)
    >>> params_batch.translations.shape
    (3, 3)
    """
    # Check LinearTransformsMapping before Affine
    if isinstance(affine, LinearTransformsMapping):
        matrices = affine.matrix
    elif isinstance(affine, Affine):
        matrices = affine.matrix[np.newaxis, :, :]
    else:
        try:
            matrix = np.asarray(affine, dtype=float)
        except (ValueError, TypeError) as e:
            raise TypeError(AFFINE_TYPE_ERROR_MSG) from e

        if matrix.ndim == 2:
            if matrix.shape != (4, 4):
                raise ValueError(AFFINE_SHAPE_ERROR)
            matrices = matrix[np.newaxis, :, :]
        elif matrix.ndim == 3:
            if matrix.shape[1:] != (4, 4):
                raise ValueError(AFFINE_SEQ_SHAPE_ERROR)
            matrices = matrix
        else:
            raise ValueError(AFFINE_SEQ_SHAPE_ERROR)

    T = matrices.shape[0]
    translations = np.zeros((T, 3))
    rotations = np.zeros((T, 3))

    for i in range(T):
        M = matrices[i]
        translations[i] = M[:3, 3]
        rotations[i] = nb.eulerangles.mat2euler(M[:3, :3])

    return MotionParameters(translations=translations, rotations=rotations)


def motion_params_to_affine(motion_parameters: MotionParameters) -> LinearTransformsMapping:
    """Convert a :class:`MotionParameters` object into a :class:`LinearTransformsMapping`.

    Parameters
    ----------
    motion_parameters : :class:`~nitransforms.analysis.utils.MotionParameters`
        A :class:`MotionParameters` object holding translations and rotations.

    Returns
    -------
    :class:`~nitransforms.linear.LinearTransformsMapping`
        A mapping representing the resolved sequence of affine transforms.

    Examples
    --------
    >>> import numpy as np
    >>> # Create a MotionParameters record for 3 frames
    >>> params = MotionParameters(
    ...     translations=np.zeros((3, 3)),
    ...     rotations=np.zeros((3, 3))
    ... )
    >>> mapping = motion_params_to_affine(params)
    >>> len(mapping)
    3
    """
    if not isinstance(motion_parameters, MotionParameters):
        raise TypeError(MOTION_PARAMS_INST_ERROR_MSG)

    translations = motion_parameters.translations
    rotations = motion_parameters.rotations
    T = translations.shape[0]

    transforms = []
    for i in range(T):
        R = nb.eulerangles.euler2mat(*rotations[i])
        t = translations[i]
        mat = nb.affines.from_matvec(R, t)
        transforms.append(Affine(mat))

    return LinearTransformsMapping(transforms)


def compute_fd_from_motion(
    motion_parameters: MotionParameters,
    *,
    radius: float = DEFAULT_FD_RADIUS,
) -> np.ndarray:
    """Compute framewise displacement (FD) from motion parameters.

    The framewise displacement is the sum of the magnitudes of the translational
    and rotational motion, computed from the frame-to-frame differences along
    the three spatial axes [Power2012]_.

    Parameters
    ----------
    motion_parameters : :obj:`MotionParameters`
        Either a :class:`MotionParameters` instance, a `(T, 3)` array of translations,
        or a legacy `(T, 6)` combined array.
    radius : :obj:`float`, optional
        Radius (in mm) of a sphere mimicking the size of a typical human brain.

    Returns
    -------
    :obj:`~numpy.ndarray`
        The framewise displacement (FD) at each timepoint as the L1 norm of
        frame-to-frame displacement across translations and rotation-derived
        displacements.

    Raises
    ------
    exc:`TypeError`
        If ``motion_parameters`` is not a :class:`MotionParameters` instance.

    Examples
    --------
    >>> import numpy as np
    >>> params = MotionParameters(
    ...     translations=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
    ...     rotations=np.zeros((2, 3))
    ... )
    >>> fd = compute_fd_from_motion(params)
    >>> fd.shape
    (2,)
    """
    if not isinstance(motion_parameters, MotionParameters):
        raise TypeError(MOTION_PARAMS_INST_ERROR_MSG)

    translations = motion_parameters.translations
    rotations = motion_parameters.rotations

    displacements = np.hstack((
        np.diff(translations, axis=0, prepend=np.zeros((1, 3))),
        np.diff(rotations * radius, axis=0, prepend=np.zeros((1, 3)))
    ))

    # FD is the L1 norm (sum of absolute values)
    return np.linalg.norm(displacements, ord=1, axis=1)


def compute_fd_from_transform(
    img: nb.spatialimages.SpatialImage,
    xfm: TransformBase,
    xfm_prev: TransformBase | None = None,
    radius: float = DEFAULT_FD_RADIUS,
    n_vertices: int = 8,
) -> float:
    """
    Compute the framewise displacement (FD) for a given transformation.

    This implementation varies with respect to the original formulation by [Power2012]_
    in that the FD is computed as the average across a number of vertices sampled over the
    sphere. See :func:`~nitransforms.analysis.utils.sample_unit_sphere` for details
    about the vertex sampling method.

    For ``n_vertices == 1``, FD is computed from rigid-body parameter increments
    (translation L1 + radius-scaled rotation L1) instead of averaging displacements
    over sampled sphere points. See :func:`compute_fd_from_motion` for direct comparability.

    Parameters
    ----------
    img : :obj:`~nibabel.spatialimages.SpatialImage`
        The reference image. Used to extract the center coordinates.
    xfm : :obj:`~nitransforms.base.TransformBase`
        The transformation to test. Applied to coordinates around the image center.
    xfm_prev : :obj:`~nitransforms.base.TransformBase`, optional
        A previous transformation to compare with. If ``None``, the identity
        transformation is assumed (no transformation).
    radius : :obj:`float`, optional
        The radius (in mm) of the spherical neighborhood around the center of the image.
    n_vertices : :obj:`int`, optional
        The number of vertices to sample on the sphere.

    Returns
    -------
    :obj:`float`
        The average framewise displacement (FD) for the test transformation.

    Raises
    ------
    exc:`ValueError`
        If ``n_vertices < 1``.

    Examples
    --------
    >>> import numpy as np
    >>> import nibabel as nb
    >>> from nitransforms.linear import Affine
    >>> img = nb.Nifti1Image(np.zeros((5, 5, 5)), np.eye(4))
    >>> xfm = Affine()
    >>> fd = compute_fd_from_transform(img, xfm)
    >>> isinstance(fd, float)
    True
    """
    if n_vertices < 1:
        raise ValueError("n_vertices must be >= 1")

    # For a single vertex, use rigid-body parameter increments (L1 translation + radius-scaled L1 rotation)
    # instead of point sampling to avoid dependence on an arbitrary sphere vertex.
    if n_vertices == 1:
        # Relative transform from previous frame to current frame
        rel = np.linalg.inv(xfm_prev.matrix) @ xfm.matrix

        d_t = rel[:3, 3]
        d_r = R.from_matrix(rel[:3, :3]).as_euler("xyz", degrees=False)

        return float(np.linalg.norm(d_t, ord=1) + radius * np.linalg.norm(d_r, ord=1))

    xfm_prev = Affine() if xfm_prev is None else xfm_prev

    affine = img.affine
    # Compute the center of the image in voxel space
    center_ijk = 0.5 * (np.array(img.shape[:3]) - 1)
    # Convert to world coordinates
    center_xyz = nb.affines.apply_affine(affine, center_ijk)
    # Generate coordinates of points at radius distance from center
    fd_coords = sample_unit_sphere(n_points=n_vertices) * radius + center_xyz
    # Compute the average displacement from the test transformation
    return np.mean(np.linalg.norm(xfm.map(fd_coords) - xfm_prev.map(fd_coords), ord=1, axis=-1))


def displacements_within_mask(
    mask_img: nb.spatialimages.SpatialImage,
    xfm: TransformBase,
    xfm_prev: TransformBase | None = None,
) -> np.ndarray:
    """
    Compute the distance between voxel coordinates mapped through two transforms.

    Parameters
    ----------
    mask_img : :obj:`~nibabel.spatialimages.SpatialImage`
        A mask image that defines the region of interest. Voxel coordinates
        within the mask are transformed.
    xfm : :obj:`~nitransforms.base.TransformBase`
        The transformation to test. This transformation is applied to the
        voxel coordinates.
    xfm_prev : :obj:`~nitransforms.base.TransformBase`, optional
        A previous (reference) transformation to compare with. If ``None``, the identity
        transformation is assumed (no transformation).

    Returns
    -------
    :obj:`~numpy.ndarray`
        An array of displacements (in mm) for each voxel within the mask.

    Examples
    --------
    >>> import numpy as np
    >>> import nibabel as nb
    >>> from nitransforms.linear import Affine
    >>> mask_img = nb.Nifti1Image(np.ones((3, 3, 3)), np.eye(4))
    >>> xfm = Affine()
    >>> disps = displacements_within_mask(mask_img, xfm)
    >>> disps.shape
    (27,)
    """
    # Mask data as boolean (True for voxels inside the mask)
    maskdata = np.asanyarray(mask_img.dataobj) > 0
    # Convert voxel coordinates to world coordinates using affine transform
    xyz = nb.affines.apply_affine(
        mask_img.affine,
        np.argwhere(maskdata),
    )
    # Apply the test transformation
    targets = xfm.map(xyz)

    # Compute the difference (displacement) between the test and reference transformations
    diffs = targets - xyz if xfm_prev is None else targets - xfm_prev.map(xyz)
    return np.linalg.norm(diffs, axis=-1)


def sample_unit_sphere(n_points: int = 8) -> np.ndarray:
    """Returns :math:`N` evenly distributed points on a unit-radius sphere.

    This function returns a **deterministic**, **quasi-uniform** point set on the
    surface of the unit sphere :math:`S^2 \\subset \\mathbb{R}^3`.

    Notes
    -----
    - There is no unique notion of "evenly distributed" for arbitrary :math:`N` on
      a sphere. This function uses:
        * **Platonic solids** for certain small :math:`N` (high symmetry; e.g.
          :math:`N = 6` gives the :math:`\\pm` axis points).
        * A **Fibonacci / golden-angle spiral** otherwise (fast, simple, good coverage).

    Parameters
    ----------
    n_points : :obj:`int`
        Number of points on the sphere.

    Returns
    -------
    :obj:`~numpy.ndarray`
        An array of shape ``(n_points, 3)`` whose rows have unit norm.

    Raises
    ------
    :exc:`TypeError`
        If ``n_points`` is a boolean or not an integer type.
    :exc:`ValueError`
        If ``n_points < 1``.

    Examples
    --------
    Basic shape + unit norm:

    >>> import numpy as np
    >>> values = (1, 2, 8, 10, 12, 20)
    >>> for n_pts in values:
    ...     X = sample_unit_sphere(n_pts)
    ...     X.shape, bool(np.allclose(np.linalg.norm(X, axis=1), 1.0))
    ((1, 3), True)
    ((2, 3), True)
    ((8, 3), True)
    ((10, 3), True)
    ((12, 3), True)
    ((20, 3), True)

    Visualization of sampled points for each case:

    .. plot::
       :context: close-figs
       :include-source: true

       import numpy as np
       import matplotlib.pyplot as plt
       import mpl_toolkits.mplot3d

       from nitransforms.analysis.utils import sample_unit_sphere

       values = (1, 2, 8, 10, 12, 20)

       fig = plt.figure(figsize=(10, 6))
       for i, n_pts in enumerate(values, start=1):
           X = sample_unit_sphere(n_pts)
           ax = fig.add_subplot(2, 3, i, projection="3d")
           ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=30)
           ax.set_title(f"n={n_pts}")
           ax.set_xlabel("x")
           ax.set_ylabel("y")
           ax.set_zlabel("z")
           ax.set_box_aspect((1, 1, 1))
       fig.tight_layout()

    For :math:`N = 6`, return the :math:`\\pm` axis points (octahedron vertices):

    >>> X = sample_unit_sphere(6)
    >>> # Each row has exactly one coordinate with magnitude 1, others 0
    >>> bool(np.all((np.abs(X) == 1.0).sum(axis=1) == 1))
    True
    >>> bool(np.all((np.abs(X) == 1.0).sum(axis=0) == 2))  # each axis appears twice (±)
    True

    For :math:`N = 4`, the tetrahedron has constant pairwise dot product -1/3
    off-diagonal:

    >>> X = sample_unit_sphere(4)
    >>> D = X @ X.T
    >>> off = D[~np.eye(4, dtype=bool)]
    >>> bool(np.allclose(off, -1/3))
    True

    For a quasi-uniform set, the second moment matrix is close to ``I/3``, and the
    minimum angular separation is non-trivial:

    >>> X = sample_unit_sphere(200)
    >>> M = (X.T @ X) / len(X)
    >>> bool(np.allclose(M, np.eye(3) / 3, atol=1e-3))
    True
    >>> def min_angle_rad(Y):
    ...     dots = np.clip(Y @ Y.T, -1.0, 1.0)
    ...     np.fill_diagonal(dots, 1.0)
    ...     ang = np.arccos(dots)
    ...     np.fill_diagonal(ang, np.inf)
    ...     return float(ang.min())
    >>> min_angle_rad(X) > 0.18
    True

    Improper inputs:

    >>> sample_unit_sphere(True)
    Traceback (most recent call last):
    ...
    TypeError: n_points must be a positive integer

    >>> sample_unit_sphere(0)
    Traceback (most recent call last):
    ...
    ValueError: n_points must be 1 or greater

    """
    if isinstance(n_points, (bool, np.bool_)) or not isinstance(
        n_points, (int, np.integer)
    ):
        raise TypeError("n_points must be a positive integer")
    if n_points < 1:
        raise ValueError("n_points must be 1 or greater")

    def _normalize(X):
        X = np.asarray(X, dtype=float)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    # --- Highly symmetric small-N cases (Platonic solids / degeneracies) ---
    if n_points == 1:
        return np.array([[0.0, 0.0, 1.0]])
    if n_points == 2:
        return np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    if n_points == 4:  # tetrahedron (4 cube corners with even number of minus signs)
        X = np.array(
            [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
        )
        return _normalize(X)
    if n_points == 6:  # octahedron: ±axes
        return np.vstack([np.eye(3), -np.eye(3)]).astype(float)
    if n_points in (8, 20):  # hexahedron (cube) & base for dodecahedron
        X = np.array(list(product([-1.0, 1.0], repeat=3)), dtype=float)
        if n_points == 8:
            return _normalize(X)

        # Dodecahedron, add 12 vertices
        extra = []
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        invphi = 1.0 / phi
        for a in (-1.0, 1.0):
            for b in (-1.0, 1.0):
                extra.extend(
                    [
                        [0.0, a * invphi, b * phi],
                        [a * invphi, b * phi, 0.0],
                        [a * phi, 0.0, b * invphi],
                    ]
                )
        X = np.vstack([X, np.asarray(extra, dtype=float)])
        return _normalize(X)

    if n_points == 12:  # icosahedron
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        X = []
        for a in (-1.0, 1.0):
            for b in (-1.0, 1.0):
                X.append([0.0, a, b * phi])
                X.append([a, b * phi, 0.0])
                X.append([a * phi, 0.0, b])
        return _normalize(X)

    # --- General N: Fibonacci / golden-angle spiral (deterministic quasi-uniform) ---
    i = np.arange(n_points, dtype=float)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))

    z = 1.0 - 2.0 * (i + 0.5) / n_points
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = golden_angle * i

    X = np.column_stack((r * np.cos(theta), r * np.sin(theta), z))
    return _normalize(X)
