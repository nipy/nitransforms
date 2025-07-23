# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""I/O test cases."""
import os
from subprocess import check_call
from io import StringIO
import filecmp
import shutil
import numpy as np
import pytest
from h5py import File as H5File

import nibabel as nb
from nibabel.eulerangles import euler2mat
from nibabel.affines import from_matvec
from scipy.io import loadmat
from nitransforms.linear import Affine
from nitransforms import nonlinear as nitnl, linear as nitl
from nitransforms.io import (
    afni,
    fsl,
    lta as fs,
    itk,
)
from nitransforms.io.lta import (
    VolumeGeometry as VG,
    FSLinearTransform as LT,
    FSLinearTransformArray as LTA,
)
from nitransforms.io.base import LinearParameters, TransformIOError, TransformFileError
from nitransforms.conftest import _datadir, _testdir
from nitransforms.resampling import apply


LPS = np.diag([-1, -1, 1, 1])
ITK_MAT = LPS.dot(np.ones((4, 4)).dot(LPS))


def test_VolumeGeometry(tmpdir, get_testdata):
    vg = VG()
    assert vg["valid"] == 0

    img = get_testdata["RAS"]
    vg = VG.from_image(img)
    assert vg["valid"] == 1
    assert np.all(vg["voxelsize"] == img.header.get_zooms()[:3])
    assert np.all(vg.as_affine() == img.affine)

    assert len(vg.to_string().split("\n")) == 8


def test_volume_group_voxel_ordering():
    """Check voxel scalings are correctly applied in non-canonical axis orderings."""
    vg = VG.from_string("""\
valid = 1  # volume info valid
filename = no_file
volume = 5 6 7
voxelsize = 2 3 4
xras   = -1 0 0
yras   = 0 0 1
zras   = 0 -1 0
cras   = 0 0 0""")
    aff = vg.as_affine()
    assert np.allclose(vg["voxelsize"], [2, 3, 4])
    assert np.allclose(nb.affines.voxel_sizes(aff), [2, 3, 4])
    assert nb.aff2axcodes(aff) == ("L", "S", "P")


def test_VG_from_LTA(data_path):
    """Check the affine interpolation from volume geometries."""
    # affine manually clipped after running mri_info on the image
    oracle = np.loadtxt(StringIO("""\
-3.0000   0.0000  -0.0000    91.3027
-0.0000   2.0575  -2.9111   -25.5251
 0.0000   2.1833   2.7433  -105.0820
 0.0000   0.0000   0.0000     1.0000"""))

    lta_text = "\n".join(
        (data_path / "bold-to-t1w.lta").read_text().splitlines()[13:21]
    )
    r2r = VG.from_string(lta_text)
    assert np.allclose(r2r.as_affine(), oracle, rtol=1e-4)


def test_LinearTransform(tmpdir):
    lt = LT()
    assert lt["m_L"].shape == (4, 4)
    assert np.all(lt["m_L"] == 0)
    for vol in ("src", "dst"):
        assert lt[vol]["valid"] == 0

    lta_text = """\
# LTA file created by NiTransforms
type      = 1
nxforms   = 1
mean      = 0.0000 0.0000 0.0000
sigma     = 1.0000
1 4 4
1.000000000000000e+00 0.000000000000000e+00 0.000000000000000e+00 0.000000000000000e+00
0.000000000000000e+00 1.000000000000000e+00 0.000000000000000e+00 0.000000000000000e+00
0.000000000000000e+00 0.000000000000000e+00 1.000000000000000e+00 0.000000000000000e+00
0.000000000000000e+00 0.000000000000000e+00 0.000000000000000e+00 1.000000000000000e+00
src volume info
valid = 1  # volume info valid
filename = file.nii.gz
volume = 57 67 56
voxelsize = 2.750000000000000e+00 2.750000000000000e+00 2.750000000000000e+00
xras   = -1.000000000000000e+00 0.000000000000000e+00 0.000000000000000e+00
yras   = 0.000000000000000e+00 1.000000000000000e+00 0.000000000000000e+00
zras   = 0.000000000000000e+00 0.000000000000000e+00 1.000000000000000e+00
cras   = -2.375000000000000e+00 1.125000000000000e+00 -1.400000000000000e+01
dst volume info
valid = 1  # volume info valid
filename = file.nii.gz
volume = 57 67 56
voxelsize = 2.750000000000000e+00 2.750000000000000e+00 2.750000000000000e+00
xras   = -1.000000000000000e+00 0.000000000000000e+00 0.000000000000000e+00
yras   = 0.000000000000000e+00 1.000000000000000e+00 0.000000000000000e+00
zras   = 0.000000000000000e+00 0.000000000000000e+00 1.000000000000000e+00
cras   = -2.375000000000000e+00 1.125000000000000e+00 -1.400000000000000e+01
"""
    xfm = LT.from_string(lta_text)
    assert xfm.to_string() == lta_text


def test_LinearTransformArray(tmpdir, data_path):
    lta = LTA()
    assert lta["nxforms"] == 0
    assert len(lta["xforms"]) == 0

    # read invalid LTA file
    test_lta = str(data_path / "affine-RAS.fsl")
    with pytest.raises(TransformFileError):
        with open(test_lta) as fp:
            LTA.from_fileobj(fp)

    test_lta = str(data_path / "affine-RAS.fs.lta")
    with open(test_lta) as fp:
        lta = LTA.from_fileobj(fp)

    assert lta.get("type") == 1
    assert len(lta["xforms"]) == lta["nxforms"] == 1
    xform = lta["xforms"][0]

    assert np.allclose(
        xform["m_L"], np.genfromtxt(test_lta, skip_header=6, skip_footer=20)
    )

    outlta = (tmpdir / "out.lta").strpath
    with open(outlta, "w") as fp:
        fp.write(lta.to_string())

    with open(outlta) as fp:
        lta2 = LTA.from_fileobj(fp)
    assert np.allclose(lta["xforms"][0]["m_L"], lta2["xforms"][0]["m_L"])


@pytest.mark.parametrize("fname", ["affine-RAS.fs", "bold-to-t1w"])
def test_LT_conversions(data_path, fname):
    r = str(data_path / f"{fname}.lta")
    v = str(data_path / f"{fname}.v2v.lta")
    with open(r) as fa, open(v) as fb:
        r2r = LTA.from_fileobj(fa)
        v2v = LTA.from_fileobj(fb)
    assert r2r["type"] == 1
    assert v2v["type"] == 0

    r2r_m = r2r["xforms"][0]["m_L"]
    v2v_m = v2v["xforms"][0]["m_L"]
    assert np.any(r2r_m != v2v_m)
    # convert vox2vox LTA to ras2ras
    v2v["xforms"][0].set_type("LINEAR_RAS_TO_RAS")
    assert v2v["xforms"][0]["type"] == 1
    assert np.allclose(r2r_m, v2v_m, rtol=1e-04)


@pytest.mark.parametrize(
    "image_orientation",
    [
        "RAS",
        "LAS",
        "LPS",
        "oblique",
    ],
)
@pytest.mark.parametrize("sw", ["afni", "fsl", "fs", "itk", "afni-array"])
def test_Linear_common(tmpdir, data_path, sw, image_orientation, get_testdata):
    tmpdir.chdir()

    moving = get_testdata[image_orientation]
    reference = get_testdata[image_orientation]

    ext = ""
    if sw == "afni":
        factory = afni.AFNILinearTransform
    elif sw == "afni-array":
        factory = afni.AFNILinearTransformArray
    elif sw == "fsl":
        factory = fsl.FSLLinearTransform
    elif sw == "itk":
        reference = None
        moving = None
        ext = ".tfm"
        factory = itk.ITKLinearTransform
    elif sw == "fs":
        ext = ".lta"
        factory = fs.FSLinearTransformArray

    with pytest.raises(TransformFileError):
        factory.from_string("")

    fname = f"affine-{image_orientation}.{sw}{ext}"

    # Test the transform loaders are implemented
    xfm = factory.from_filename(data_path / fname)

    with open(str(data_path / fname)) as f:
        text = f.read()
        f.seek(0)
        xfm = factory.from_fileobj(f)

    # Test to_string
    assert fs._drop_comments(text) == fs._drop_comments(xfm.to_string())

    xfm.to_filename(fname)
    assert filecmp.cmp(fname, str((data_path / fname).resolve()))

    # Test from_ras
    RAS = from_matvec(euler2mat(x=0.9, y=0.001, z=0.001), [4.0, 2.0, -1.0])
    if sw == "afni-array":
        RAS = np.array([RAS, RAS])

    xfm = factory.from_ras(RAS, reference=reference, moving=moving)
    assert np.allclose(xfm.to_ras(reference=reference, moving=moving), RAS)

    # Test without images
    if sw == "fsl":
        with pytest.raises(TransformIOError):
            factory.from_ras(RAS)
    else:
        xfm = factory.from_ras(RAS)
        assert np.allclose(xfm.to_ras(), RAS)


@pytest.mark.parametrize(
    "image_orientation",
    [
        "RAS",
        "LAS",
        "LPS",
        "oblique",
    ],
)
@pytest.mark.parametrize("sw", ["afni", "fsl", "itk"])
def test_LinearList_common(tmpdir, data_path, sw, image_orientation, get_testdata):
    tmpdir.chdir()

    angles = np.random.uniform(low=-3.14, high=3.14, size=(5, 3))
    translation = np.random.uniform(low=-5.0, high=5.0, size=(5, 3))
    mats = [from_matvec(euler2mat(*a), t) for a, t in zip(angles, translation)]

    ext = ""
    if sw == "afni":
        factory = afni.AFNILinearTransformArray
    elif sw == "fsl":
        factory = fsl.FSLLinearTransformArray
    elif sw == "itk":
        ext = ".tfm"
        factory = itk.ITKLinearTransformArray

    tflist1 = factory(mats)

    fname = f"affine-{image_orientation}.{sw}{ext}"

    with pytest.raises(FileNotFoundError):
        factory.from_filename(fname)

    tmpdir.join("singlemat.%s" % ext).write("")
    with pytest.raises(TransformFileError):
        factory.from_filename("singlemat.%s" % ext)

    tflist1.to_filename(fname)
    tflist2 = factory.from_filename(fname)

    assert tflist1["nxforms"] == tflist2["nxforms"]
    assert all(
        [
            np.allclose(x1["parameters"], x2["parameters"])
            for x1, x2 in zip(tflist1.xforms, tflist2.xforms)
        ]
    )


def test_ITKLinearTransform(tmpdir, testdata_path):
    tmpdir.chdir()

    matlabfile = testdata_path / "ds-005_sub-01_from-T1_to-OASIS_affine.mat"
    mat = loadmat(str(matlabfile))
    with open(str(matlabfile), "rb") as f:
        itkxfm = itk.ITKLinearTransform.from_fileobj(f)
    assert np.allclose(
        itkxfm["parameters"][:3, :3].flatten(),
        mat["AffineTransform_float_3_3"][:-3].flatten(),
    )
    assert np.allclose(itkxfm["offset"], mat["fixed"].reshape((3,)))

    itkxfm = itk.ITKLinearTransform.from_filename(matlabfile)
    assert np.allclose(
        itkxfm["parameters"][:3, :3].flatten(),
        mat["AffineTransform_float_3_3"][:-3].flatten(),
    )
    assert np.allclose(itkxfm["offset"], mat["fixed"].reshape((3,)))

    # Test to_filename(textfiles)
    itkxfm.to_filename("textfile.tfm")
    with open("textfile.tfm") as f:
        itkxfm2 = itk.ITKLinearTransform.from_fileobj(f)
    assert np.allclose(itkxfm["parameters"], itkxfm2["parameters"])

    # Test to_filename(matlab)
    itkxfm.to_filename("copy.mat")
    with open("copy.mat", "rb") as f:
        itkxfm3 = itk.ITKLinearTransform.from_fileobj(f)
    assert np.all(itkxfm["parameters"] == itkxfm3["parameters"])

    rasmat = from_matvec(euler2mat(x=0.9, y=0.001, z=0.001), [4.0, 2.0, -1.0])
    itkxfm = itk.ITKLinearTransform.from_ras(rasmat)
    assert np.allclose(itkxfm["parameters"], ITK_MAT * rasmat)
    assert np.allclose(itkxfm.to_ras(), rasmat)


def test_ITKLinearTransformArray(tmpdir, data_path):
    tmpdir.chdir()

    with open(str(data_path / "itktflist.tfm")) as f:
        text = f.read()
        f.seek(0)
        itklist = itk.ITKLinearTransformArray.from_fileobj(f)

    itklistb = itk.ITKLinearTransformArray.from_filename(data_path / "itktflist.tfm")
    assert itklist["nxforms"] == itklistb["nxforms"]
    assert all(
        [
            np.allclose(x1["parameters"], x2["parameters"])
            for x1, x2 in zip(itklist.xforms, itklistb.xforms)
        ]
    )

    tmpdir.join("empty.mat").write("")
    with pytest.raises(TransformFileError):
        itklistb.from_filename("empty.mat")

    assert itklist["nxforms"] == 9
    assert text == itklist.to_string()
    with pytest.raises(TransformFileError):
        itk.ITKLinearTransformArray.from_string("\n".join(text.splitlines()[1:]))

    itklist.to_filename("copy.tfm")
    with open("copy.tfm") as f:
        copytext = f.read()
    assert text == copytext

    itklist = itk.ITKLinearTransformArray(
        xforms=[np.random.normal(size=(4, 4)) for _ in range(4)]
    )

    assert itklist["nxforms"] == 4
    assert itklist["xforms"][1].structarr["index"] == 1

    with pytest.raises(KeyError):
        itklist["invalid_key"]

    xfm = itklist["xforms"][1]
    xfm["index"] = 1
    with open("extracted.tfm", "w") as f:
        f.write(xfm.to_string())

    with open("extracted.tfm") as f:
        xfm2 = itk.ITKLinearTransform.from_fileobj(f)
    assert np.allclose(
        xfm.structarr["parameters"][:3, ...], xfm2.structarr["parameters"][:3, ...]
    )

    # ITK does not handle transforms lists in Matlab format
    with pytest.raises(TransformFileError):
        itklist.to_filename("matlablist.mat")

    with pytest.raises(TransformFileError):
        xfm2 = itk.ITKLinearTransformArray.from_binary({})

    with open("filename.mat", "ab") as f:
        with pytest.raises(TransformFileError):
            xfm2 = itk.ITKLinearTransformArray.from_fileobj(f)


def test_LinearParameters(tmpdir):
    """Just pushes coverage up."""
    tmpdir.join("file.txt").write("")

    with pytest.raises(NotImplementedError):
        LinearParameters.from_string("")

    with pytest.raises(NotImplementedError):
        LinearParameters.from_fileobj(tmpdir.join("file.txt").open())


def test_afni_Displacements():
    """Test displacements fields."""
    field = nb.Nifti1Image(np.zeros((10, 10, 10)), None, None)
    with pytest.raises(TransformFileError):
        afni.AFNIDisplacementsField.from_image(field)

    field = nb.Nifti1Image(np.zeros((10, 10, 10, 2, 3)), None, None)
    with pytest.raises(TransformFileError):
        afni.AFNIDisplacementsField.from_image(field)

    field = nb.Nifti1Image(np.zeros((10, 10, 10, 1, 4)), None, None)
    with pytest.raises(TransformFileError):
        afni.AFNIDisplacementsField.from_image(field)


@pytest.mark.parametrize("only_linear", [True, False])
@pytest.mark.parametrize("h5_path,nxforms", [
    (_datadir / "affine-antsComposite.h5", 1),
    (_testdir / "ds-005_sub-01_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5", 2),
])
def test_itk_h5(tmpdir, only_linear, h5_path, nxforms):
    """Test displacements fields."""
    assert (
        len(
            list(
                itk.ITKCompositeH5.from_filename(
                    h5_path,
                    only_linear=only_linear,
                )
            )
        )
        == nxforms if not only_linear else 1
    )

    with pytest.raises(TransformFileError):
        list(
            itk.ITKCompositeH5.from_filename(
                h5_path.absolute().name.replace(".h5", ".x5"),
                only_linear=only_linear,
            )
        )

    tmpdir.chdir()
    shutil.copy(h5_path, "test.h5")
    os.chmod("test.h5", 0o666)

    with H5File("test.h5", "r+") as h5file:
        h5group = h5file["TransformGroup"]
        xfm = h5group[list(h5group.keys())[1]]
        xfm["TransformType"][0] = b"InventTransform"

    with pytest.raises(TransformIOError):
        itk.ITKCompositeH5.from_filename("test.h5")


@pytest.mark.parametrize(
    "file_type, test_file", [(LTA, "from-fsnative_to-scanner_mode-image.lta")]
)
def test_regressions(file_type, test_file, data_path):
    file_type.from_filename(data_path / "regressions" / test_file)


@pytest.mark.parametrize("parameters", [
    {"x": 0.1, "y": 0.03, "z": 0.002},
    {"x": 0.001, "y": 0.3, "z": 0.002},
    {"x": 0.01, "y": 0.03, "z": 0.2},
])
@pytest.mark.parametrize("dir_x", (-1, 1))
@pytest.mark.parametrize("dir_y", (-1, 1))
@pytest.mark.parametrize("dir_z", (1, -1))
@pytest.mark.parametrize("swapaxes", [
    None, (0, 1), (1, 2), (0, 2),
])
def test_afni_oblique(tmpdir, parameters, swapaxes, testdata_path, dir_x, dir_y, dir_z):
    tmpdir.chdir()
    img, R = _generate_reoriented(
        testdata_path / "someones_anatomy.nii.gz",
        (dir_x, dir_y, dir_z),
        swapaxes,
        parameters
    )
    img.to_filename("orig.nii.gz")

    # Run AFNI's 3drefit -deoblique
    if not shutil.which("3drefit"):
        pytest.skip("Command 3drefit not found on host")

    shutil.copy(f"{tmpdir}/orig.nii.gz", f"{tmpdir}/deob_3drefit.nii.gz")
    cmd = f"3drefit -deoblique {tmpdir}/deob_3drefit.nii.gz"
    assert check_call([cmd], shell=True) == 0

    # Check that nitransforms can make out the deoblique affine:
    card_aff = afni._dicom_real_to_card(img.affine)
    assert np.allclose(card_aff, nb.load("deob_3drefit.nii.gz").affine)

    # Check that nitransforms can emulate 3drefit -deoblique
    nt3drefit = apply(
        Affine(
            afni._cardinal_rotation(img.affine, False),
            reference="deob_3drefit.nii.gz",
        ),
        "orig.nii.gz",
    )

    diff = (
        np.asanyarray(img.dataobj, dtype="uint8")
        - np.asanyarray(nt3drefit.dataobj, dtype="uint8")
    )
    assert np.sqrt((diff[10:-10, 10:-10, 10:-10] ** 2).mean()) < 0.1

    # Check that nitransforms can revert 3drefit -deoblique
    nt_undo3drefit = apply(
        Affine(
            afni._cardinal_rotation(img.affine, True),
            reference="orig.nii.gz",
        ),
        "deob_3drefit.nii.gz",
    )

    diff = (
        np.asanyarray(img.dataobj, dtype="uint8")
        - np.asanyarray(nt_undo3drefit.dataobj, dtype="uint8")
    )
    assert np.sqrt((diff[10:-10, 10:-10, 10:-10] ** 2).mean()) < 0.1

    # Check the target grid by 3dWarp and the affine & size interpolated by NiTransforms
    cmd = f"3dWarp -verb -deoblique -NN -prefix {tmpdir}/deob.nii.gz {tmpdir}/orig.nii.gz"
    assert check_call([cmd], shell=True) == 0

    deobnii = nb.load("deob.nii.gz")
    deobaff, deobshape = afni._afni_deobliqued_grid(img.affine, img.shape)

    assert np.all(deobshape == deobnii.shape[:3])
    assert np.allclose(deobaff, deobnii.affine)

    # Check resampling in deobliqued grid
    ntdeobnii = apply(
        Affine(np.eye(4), reference=deobnii.__class__(
            np.zeros(deobshape, dtype="uint8"),
            deobaff,
            deobnii.header
        )),
        img,
        order=0,
    )

    # Generate an internal box to exclude border effects
    box = np.zeros(img.shape, dtype="uint8")
    box[10:-10, 10:-10, 10:-10] = 1
    ntdeobmask = apply(
        Affine(np.eye(4), reference=ntdeobnii),
        nb.Nifti1Image(box, img.affine, img.header),
        order=0,
    )
    mask = np.asanyarray(ntdeobmask.dataobj, dtype=bool)

    diff = (
        np.asanyarray(deobnii.dataobj, dtype="uint8")
        - np.asanyarray(ntdeobnii.dataobj, dtype="uint8")
    )
    assert np.sqrt((diff[mask] ** 2).mean()) < 0.1

    # Confirm AFNI's rotation of axis is consistent with the one we introduced
    afni_warpdrive_inv = afni._afni_header(
        nb.load("deob.nii.gz"),
        field="WARPDRIVE_MATVEC_INV_000000",
        to_ras=True,
    )
    assert np.allclose(afni_warpdrive_inv[:3, :3], R[:3, :3])

    # Check nitransforms' estimation of warpdrive with header
    nt_warpdrive_inv = afni._afni_warpdrive(img.affine, forward=False)
    assert not np.allclose(afni_warpdrive_inv, nt_warpdrive_inv)


def _generate_reoriented(path, directions, swapaxes, parameters):
    img = nb.load(path)
    shape = np.array(img.shape[:3])
    hdr = img.header.copy()
    aff = img.affine.copy()
    data = np.asanyarray(img.dataobj, dtype="uint8")

    if directions != (1, 1, 1):
        last_ijk = np.hstack((shape - 1, 1.0))
        last_xyz = aff @ last_ijk
        aff = np.diag((*directions, 1)) @ aff

        for ax in range(3):
            if (directions[ax] == -1):
                aff[ax, 3] = last_xyz[ax]
                data = np.flip(data, ax)

    if swapaxes is not None:
        data = np.swapaxes(data, swapaxes[0], swapaxes[1])
        aff[list(reversed(swapaxes)), :] = aff[(swapaxes), :]

    R = from_matvec(euler2mat(**parameters), [0.0, 0.0, 0.0])

    newaff = R @ aff
    hdr.set_qform(newaff, code=1)
    hdr.set_sform(newaff, code=1)
    return img.__class__(data, newaff, hdr), R


def test_itk_linear_h5(tmpdir, data_path, testdata_path):
    """Check different lower-level loading options."""

    # File loadable with transform array
    h5xfm = itk.ITKLinearTransformArray.from_filename(
        data_path / "affine-antsComposite.h5"
    )
    assert len(h5xfm.xforms) == 1

    with open(data_path / "affine-antsComposite.h5", "rb") as f:
        h5xfm = itk.ITKLinearTransformArray.from_fileobj(f)
    assert len(h5xfm.xforms) == 1

    # File loadable with single affine object
    itk.ITKLinearTransform.from_filename(
        data_path / "affine-antsComposite.h5"
    )

    with open(data_path / "affine-antsComposite.h5", "rb") as f:
        itk.ITKLinearTransform.from_fileobj(f)

    # Exercise only_linear
    itk.ITKCompositeH5.from_filename(
        testdata_path / "ds-005_sub-01_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5",
        only_linear=True,
    )

    tmpdir.chdir()
    shutil.copy(data_path / "affine-antsComposite.h5", "test.h5")
    os.chmod("test.h5", 0o666)

    with H5File("test.h5", "r+") as h5file:
        h5group = h5file["TransformGroup"]
        xfm = h5group.create_group("2")
        xfm["TransformType"] = (b"AffineTransform", b"")
        xfm["TransformParameters"] = np.zeros(12, dtype=float)
        xfm["TransformFixedParameters"] = np.zeros(3, dtype=float)

    # File loadable with transform array
    h5xfm = itk.ITKLinearTransformArray.from_filename("test.h5")
    assert len(h5xfm.xforms) == 2

    # File loadable with generalistic object (NOTE we directly access the list)
    h5xfm = itk.ITKCompositeH5.from_filename("test.h5")
    assert len(h5xfm) == 2

    # Error raised if the we try to use the single affine loader
    with pytest.raises(TransformIOError):
        itk.ITKLinearTransform.from_filename("test.h5")

    shutil.copy(data_path / "affine-antsComposite.h5", "test.h5")
    os.chmod("test.h5", 0o666)

    # Generate an empty h5 file
    with H5File("test.h5", "r+") as h5file:
        h5group = h5file["TransformGroup"]
        del h5group["1"]

    # File loadable with generalistic object
    h5xfm = itk.ITKCompositeH5.from_filename("test.h5")
    assert len(h5xfm) == 0

    # Error raised if the we try to use the single affine loader
    with pytest.raises(TransformIOError):
        itk.ITKLinearTransform.from_filename("test.h5")

# Added tests for h5 orientation bug


@pytest.mark.xfail(
    reason="GH-137/GH-171: displacement field dimension order is wrong",
    strict=False,
)
def test_itk_h5_field_order(tmp_path):
    """Displacement fields stored in row-major order should fail to round-trip."""
    shape = (3, 4, 5)
    vals = np.arange(np.prod(shape), dtype=float).reshape(shape)
    field = np.stack([vals, vals + 100, vals + 200], axis=0)

    params = field.reshape(-1, order="C")
    fixed = np.array(list(shape) + [0, 0, 0] + [1, 1, 1] + list(np.eye(3).ravel()), dtype=float)
    fname = tmp_path / "field.h5"
    with H5File(fname, "w") as f:
        grp = f.create_group("TransformGroup")
        grp.create_group("0")["TransformType"] = np.array([b"CompositeTransform_double_3_3"])
        g1 = grp.create_group("1")
        g1["TransformType"] = np.array([b"DisplacementFieldTransform_float_3_3"])
        g1["TransformFixedParameters"] = fixed
        g1["TransformParameters"] = params

    img = itk.ITKCompositeH5.from_filename(fname)[0]
    expected = np.moveaxis(field, 0, -1)
    expected[..., (0, 1)] *= -1
    assert np.allclose(img.get_fdata(), expected)


def _load_composite_testdata(data_path):
    """Return the composite HDF5 and displacement field from regressions."""
    h5file = data_path / "regressions" / "ants_t1_to_mniComposite.h5"
    # Generated using
    # CompositeTransformUtil --disassemble ants_t1_to_mniComposite.h5 \
    #     ants_t1_to_mniComposite
    warpfile = data_path / "regressions" / (
        "01_ants_t1_to_mniComposite_DisplacementFieldTransform.nii.gz"
    )
    if not (h5file.exists() and warpfile.exists()):
        pytest.skip("Composite transform test data not available")
    return h5file, warpfile


@pytest.mark.xfail(
    reason="GH-137/GH-171: displacement field dimension order is wrong",
    strict=False,
)
def test_itk_h5_displacement_mismatch(testdata_path):
    """Composite displacements should match the standalone field"""
    h5file, warpfile = _load_composite_testdata(testdata_path)
    xforms = itk.ITKCompositeH5.from_filename(h5file)
    field_h5 = xforms[1]
    field_img = itk.ITKDisplacementsField.from_filename(warpfile)

    np.testing.assert_array_equal(
        np.asanyarray(field_h5.dataobj), np.asanyarray(field_img.dataobj)
    )


def test_itk_h5_transpose_fix(testdata_path):
    """Check the displacement field orientation explicitly.

    ITK stores displacement fields with the vector dimension leading in
    Fortran (column-major) order [1]_. Transposing the parameters from the HDF5
    composite file accordingly should match the standalone displacement image.

    References
    ----------
    .. [1] ITK Software Guide. https://itk.org/ItkSoftwareGuide.pdf
    """
    h5file, warpfile = _load_composite_testdata(testdata_path)

    with H5File(h5file, "r") as f:
        group = f["TransformGroup"]["2"]
        size = group["TransformFixedParameters"][:3].astype(int)
        params = group["TransformParameters"][:].reshape(*size, 3)

    img = nb.load(warpfile)
    ref = np.squeeze(np.asanyarray(img.dataobj))

    np.testing.assert_array_equal(params.transpose(2, 1, 0, 3), ref)


def test_itk_h5_field_order_fortran(tmp_path):
    """Verify Fortran-order displacement fields load correctly"""
    shape = (3, 4, 5)
    vals = np.arange(np.prod(shape), dtype=float).reshape(shape)
    field = np.stack([vals, vals + 100, vals + 200], axis=0)

    params = field.reshape(-1, order="F")
    fixed = np.array(list(shape) + [0, 0, 0] + [1, 1, 1] + list(np.eye(3).ravel()), dtype=float)
    fname = tmp_path / "field_f.h5"
    with H5File(fname, "w") as f:
        grp = f.create_group("TransformGroup")
        grp.create_group("0")["TransformType"] = np.array([b"CompositeTransform_double_3_3"])
        g1 = grp.create_group("1")
        g1["TransformType"] = np.array([b"DisplacementFieldTransform_float_3_3"])
        g1["TransformFixedParameters"] = fixed
        g1["TransformParameters"] = params

    img = itk.ITKCompositeH5.from_filename(fname)[0]
    expected = np.moveaxis(field, 0, -1)
    expected[..., (0, 1)] *= -1
    assert np.allclose(img.get_fdata(), expected)


def test_composite_h5_map_against_ants(testdata_path, tmp_path):
    """Map points with NiTransforms and compare to ANTs."""
    h5file, _ = _load_composite_testdata(testdata_path)
    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    csvin = tmp_path / "points.csv"
    with open(csvin, "w") as f:
        f.write("x,y,z\n")
        for row in points:
            f.write(",".join(map(str, row)) + "\n")

    csvout = tmp_path / "out.csv"
    cmd = f"antsApplyTransformsToPoints -d 3 -i {csvin} -o {csvout} -t {h5file}"
    exe = cmd.split()[0]
    if not shutil.which(exe):
        pytest.skip(f"Command {exe} not found on host")
    check_call(cmd, shell=True)

    ants_res = np.genfromtxt(csvout, delimiter=",", names=True)
    ants_pts = np.vstack([ants_res[n] for n in ("x", "y", "z")]).T

    xforms = itk.ITKCompositeH5.from_filename(h5file)
    chain = (
        nitl.Affine(xforms[0].to_ras())
        + nitnl.DenseFieldTransform(xforms[1])
    )
    mapped = chain.map(points)

    assert np.allclose(mapped, ants_pts, atol=1e-6)
