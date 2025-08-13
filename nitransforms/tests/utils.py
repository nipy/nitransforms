"""Utilities for testing."""

from pathlib import Path
import numpy as np
import nibabel as nb

from nitransforms import linear as nbl
from nitransforms.base import ImageGrid


def assert_affines_by_filename(affine1, affine2):
    """Check affines by filename."""
    affine1 = Path(affine1)
    affine2 = Path(affine2)
    assert affine1.suffix == affine2.suffix, "affines of different type"

    ext_to_fmt = {
        ".tfm": "itk",  # An ITK transform
        ".lta": "fs",  # FreeSurfer LTA
    }

    ext = affine1.suffix[-4:]
    if ext in ext_to_fmt:
        xfm1 = nbl.load(str(affine1), fmt=ext_to_fmt[ext])
        xfm2 = nbl.load(str(affine2), fmt=ext_to_fmt[ext])
        assert xfm1 == xfm2
    else:
        xfm1 = np.loadtxt(str(affine1))
        xfm2 = np.loadtxt(str(affine2))
        assert np.allclose(xfm1, xfm2, atol=1e-04)


def get_points(reference_nii, ongrid, npoints=5000, rng=None):
    """Get points in RAS space."""
    if rng is None:
        rng = np.random.default_rng()

    # Get sampling indices
    shape = reference_nii.shape[:3]
    ref_affine = reference_nii.affine.copy()
    reference = ImageGrid(nb.Nifti1Image(np.zeros(shape), ref_affine, None))
    grid_ijk = reference.ndindex
    grid_xyz = reference.ras(grid_ijk)

    subsample = rng.choice(grid_ijk.shape[0], npoints)
    points_ijk = grid_ijk.copy() if ongrid else grid_ijk[subsample]
    coords_xyz = (
        grid_xyz
        if ongrid
        else reference.ras(points_ijk) + rng.normal(size=points_ijk.shape)
    )

    return coords_xyz, points_ijk, grid_xyz, shape, ref_affine, reference, subsample
