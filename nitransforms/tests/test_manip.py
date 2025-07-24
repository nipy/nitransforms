# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests of nonlinear transforms."""

from subprocess import check_call
import shutil

import pytest

import numpy as np
import nibabel as nb
import h5py
from ..base import TransformError
from ..manip import TransformChain
from ..linear import Affine
from ..nonlinear import DenseFieldTransform
from ..io import x5, itk

FMT = {"lta": "fs", "tfm": "itk"}


@pytest.mark.parametrize("ext0", ["lta", "tfm"])
@pytest.mark.parametrize("ext1", ["lta", "tfm"])
@pytest.mark.parametrize("ext2", ["lta", "tfm"])
def test_collapse_affines(tmp_path, data_path, ext0, ext1, ext2):
    """Check whether affines are correctly collapsed."""
    chain = TransformChain(
        [
            Affine.from_filename(
                data_path
                / "regressions"
                / f"from-fsnative_to-scanner_mode-image.{ext0}",
                fmt=f"{FMT[ext0]}",
            ),
            Affine.from_filename(
                data_path / "regressions" / f"from-scanner_to-bold_mode-image.{ext1}",
                fmt=f"{FMT[ext1]}",
            ),
        ]
    )
    assert np.allclose(
        chain.asaffine().matrix,
        Affine.from_filename(
            data_path / "regressions" / f"from-fsnative_to-bold_mode-image.{ext2}",
            fmt=f"{FMT[ext2]}",
        ).matrix,
    )


def test_transformchain_x5_roundtrip(tmp_path):
    """Round-trip TransformChain with X5 storage."""

    # Test empty transform file
    x5.to_filename(tmp_path / "empty.x5", [])
    with pytest.raises(TransformError):
        TransformChain.from_filename(tmp_path / "empty.x5")

    mat = np.eye(4)
    mat[0, 3] = 1
    aff = Affine(mat)

    # Test loading X5 with no transforms chains
    x5.to_filename(tmp_path / "nochain.x5", [aff.to_x5()])
    with pytest.raises(TransformError):
        TransformChain.from_filename(tmp_path / "nochain.x5")

    field = nb.Nifti1Image(np.zeros((5, 5, 5, 3), dtype="float32"), np.eye(4))
    fdata = field.get_fdata()
    fdata[..., 1] = 1
    field = nb.Nifti1Image(fdata, np.eye(4))
    dfield = DenseFieldTransform(field, is_deltas=True)

    # Create a chain
    chain = TransformChain([aff, aff, aff, dfield])
    fname = tmp_path / "chain.x5"
    chain.to_filename(fname)

    with h5py.File(fname) as f:
        assert len(f["TransformGroup"]) == 2

    chain.to_filename(fname)  # append again, should not duplicate transforms

    with h5py.File(fname) as f:
        assert len(f["TransformGroup"]) == 2

    loaded0 = TransformChain.from_filename(fname, fmt="X5", x5_chain=0)
    loaded1 = TransformChain.from_filename(fname, fmt="X5", x5_chain=1)

    assert len(loaded0) == len(chain)
    assert len(loaded1) == len(chain)
    assert np.allclose(chain.map([[0, 0, 0]]), loaded1.map([[0, 0, 0]]))


@pytest.mark.xfail(
    reason="TransformChain.map() doesn't work properly (discovered with GH-167)",
    strict=False,
)
def test_composite_h5_map_against_ants(testdata_path, tmp_path):
    """Map points with NiTransforms and compare to ANTs."""
    h5file = testdata_path / "regressions" / "ants_t1_to_mniComposite.h5"
    if not h5file.exists():
        pytest.skip(f"Necessary file <{h5file}> missing.")

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [10.0, -10.0, 5.0],
            [-5.0, 7.0, -2.0],
            [12.0, 0.0, -11.0],
        ]
    )
    csvin = tmp_path / "points.csv"
    np.savetxt(csvin, points, delimiter=",", header="x,y,z", comments="")

    csvout = tmp_path / "out.csv"
    cmd = f"antsApplyTransformsToPoints -d 3 -i {csvin} -o {csvout} -t {h5file}"
    exe = cmd.split()[0]
    if not shutil.which(exe):
        pytest.skip(f"Command {exe} not found on host")
    check_call(cmd, shell=True)

    ants_res = np.genfromtxt(csvout, delimiter=",", names=True)
    ants_pts = np.vstack([ants_res[n] for n in ("x", "y", "z")]).T

    xforms = itk.ITKCompositeH5.from_filename(h5file)
    chain = Affine(xforms[0].to_ras()) + DenseFieldTransform(xforms[1])
    mapped = chain.map(points)

    assert np.allclose(mapped, ants_pts, atol=1e-6)
