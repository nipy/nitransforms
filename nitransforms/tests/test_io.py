# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""I/O test cases."""
import numpy as np
import pytest

import filecmp
import nibabel as nb
from nibabel.eulerangles import euler2mat
from nibabel.affines import from_matvec
from scipy.io import loadmat, savemat
from ..io import (
    afni,
    fsl,
    lta as fs,
    itk,
    VolumeGeometry as VG,
    LinearTransform as LT,
    LinearTransformArray as LTA,
)
from ..io.base import _read_mat, LinearParameters, TransformFileError

LPS = np.diag([-1, -1, 1, 1])
ITK_MAT = LPS.dot(np.ones((4, 4)).dot(LPS))


def test_VolumeGeometry(tmpdir, get_testdata):
    vg = VG()
    assert vg['valid'] == 0

    img = get_testdata['RAS']
    vg = VG.from_image(img)
    assert vg['valid'] == 1
    assert np.all(vg['voxelsize'] == img.header.get_zooms()[:3])
    assert np.all(vg.as_affine() == img.affine)

    assert len(vg.to_string().split('\n')) == 8


def test_LinearTransform(tmpdir):
    lt = LT()
    assert lt['m_L'].shape == (4, 4)
    assert np.all(lt['m_L'] == 0)
    for vol in ('src', 'dst'):
        assert lt[vol]['valid'] == 0


def test_LinearTransformArray(tmpdir, data_path):
    lta = LTA()
    assert lta['nxforms'] == 0
    assert len(lta['xforms']) == 0

    # read invalid LTA file
    test_lta = str(data_path / 'affine-RAS.fsl')
    with pytest.raises(TransformFileError):
        with open(test_lta) as fp:
            LTA.from_fileobj(fp)

    test_lta = str(data_path / 'affine-RAS.fs.lta')
    with open(test_lta) as fp:
        lta = LTA.from_fileobj(fp)

    assert lta.get('type') == 1
    assert len(lta['xforms']) == lta['nxforms'] == 1
    xform = lta['xforms'][0]

    assert np.allclose(
        xform['m_L'], np.genfromtxt(test_lta, skip_header=5, skip_footer=20)
    )

    outlta = (tmpdir / 'out.lta').strpath
    with open(outlta, 'w') as fp:
        fp.write(lta.to_string())

    with open(outlta) as fp:
        lta2 = LTA.from_fileobj(fp)
    assert np.allclose(lta['xforms'][0]['m_L'], lta2['xforms'][0]['m_L'])


def test_LT_conversions(data_path):
    r = str(data_path / 'affine-RAS.fs.lta')
    v = str(data_path / 'affine-RAS.fs.v2v.lta')
    with open(r) as fa, open(v) as fb:
        r2r = LTA.from_fileobj(fa)
        v2v = LTA.from_fileobj(fb)
    assert r2r['type'] == 1
    assert v2v['type'] == 0

    r2r_m = r2r['xforms'][0]['m_L']
    v2v_m = v2v['xforms'][0]['m_L']
    assert np.any(r2r_m != v2v_m)
    # convert vox2vox LTA to ras2ras
    v2v['xforms'][0].set_type('LINEAR_RAS_TO_RAS')
    assert v2v['xforms'][0]['type'] == 1
    assert np.allclose(r2r_m, v2v_m, atol=1e-05)


@pytest.mark.xfail(raises=(FileNotFoundError, NotImplementedError))
@pytest.mark.parametrize('image_orientation', [
    'RAS', 'LAS', 'LPS', 'oblique',
])
@pytest.mark.parametrize('sw', ['afni', 'fsl', 'fs', 'itk'])
def test_Linear_common(tmpdir, data_path, sw, image_orientation,
                       get_testdata):
    tmpdir.chdir()

    moving = get_testdata[image_orientation]
    reference = get_testdata[image_orientation]

    ext = ''
    if sw == 'afni':
        factory = afni.AFNILinearTransform
    elif sw == 'fsl':
        factory = fsl.FSLLinearTransform
    elif sw == 'itk':
        reference = None
        moving = None
        ext = '.tfm'
        factory = itk.ITKLinearTransform
    elif sw == 'fs':
        ext = '.lta'
        factory = fs.LinearTransformArray

    with pytest.raises(TransformFileError):
        factory.from_string('')

    fname = 'affine-%s.%s%s' % (image_orientation, sw, ext)

    # Test the transform loaders are implemented
    xfm = factory.from_filename(data_path / fname)

    with open(str(data_path / fname)) as f:
        text = f.read()
        f.seek(0)
        xfm = factory.from_fileobj(f)

    # Test to_string
    assert text == xfm.to_string()

    xfm.to_filename(fname)
    assert filecmp.cmp(fname, str((data_path / fname).resolve()))

    # Test from_ras
    RAS = from_matvec(euler2mat(x=0.9, y=0.001, z=0.001), [4.0, 2.0, -1.0])
    xfm = factory.from_ras(RAS, reference=reference, moving=moving)
    assert np.allclose(xfm.to_ras(reference=reference, moving=moving), RAS)


@pytest.mark.parametrize('image_orientation', [
    'RAS', 'LAS', 'LPS', 'oblique',
])
@pytest.mark.parametrize('sw', ['afni', 'fsl', 'itk'])
def test_LinearList_common(tmpdir, data_path, sw, image_orientation,
                           get_testdata):
    tmpdir.chdir()

    angles = np.random.uniform(low=-3.14, high=3.14, size=(5, 3))
    translation = np.random.uniform(low=-5., high=5., size=(5, 3))
    mats = [from_matvec(euler2mat(*a), t)
            for a, t in zip(angles, translation)]

    ext = ''
    if sw == 'afni':
        factory = afni.AFNILinearTransformArray
    elif sw == 'fsl':
        factory = fsl.FSLLinearTransformArray
    elif sw == 'itk':
        ext = '.tfm'
        factory = itk.ITKLinearTransformArray

    tflist1 = factory(mats)

    fname = 'affine-%s.%s%s' % (image_orientation, sw, ext)

    with pytest.raises(FileNotFoundError):
        factory.from_filename(fname)

    tmpdir.join('singlemat.%s' % ext).write('')
    with pytest.raises(TransformFileError):
        factory.from_filename('singlemat.%s' % ext)

    tflist1.to_filename(fname)
    tflist2 = factory.from_filename(fname)

    assert tflist1['nxforms'] == tflist2['nxforms']
    assert all([np.allclose(x1['parameters'], x2['parameters'])
                for x1, x2 in zip(tflist1.xforms, tflist2.xforms)])


def test_ITKLinearTransform(tmpdir, testdata_path):
    tmpdir.chdir()

    matlabfile = testdata_path / 'ds-005_sub-01_from-T1_to-OASIS_affine.mat'
    mat = loadmat(str(matlabfile))
    with open(str(matlabfile), 'rb') as f:
        itkxfm = itk.ITKLinearTransform.from_fileobj(f)
    assert np.allclose(itkxfm['parameters'][:3, :3].flatten(),
                       mat['AffineTransform_float_3_3'][:-3].flatten())
    assert np.allclose(itkxfm['offset'], mat['fixed'].reshape((3, )))

    itkxfm = itk.ITKLinearTransform.from_filename(matlabfile)
    assert np.allclose(itkxfm['parameters'][:3, :3].flatten(),
                       mat['AffineTransform_float_3_3'][:-3].flatten())
    assert np.allclose(itkxfm['offset'], mat['fixed'].reshape((3, )))

    # Test to_filename(textfiles)
    itkxfm.to_filename('textfile.tfm')
    with open('textfile.tfm', 'r') as f:
        itkxfm2 = itk.ITKLinearTransform.from_fileobj(f)
    assert np.allclose(itkxfm['parameters'], itkxfm2['parameters'])

    # Test to_filename(matlab)
    itkxfm.to_filename('copy.mat')
    with open('copy.mat', 'rb') as f:
        itkxfm3 = itk.ITKLinearTransform.from_fileobj(f)
    assert np.all(itkxfm['parameters'] == itkxfm3['parameters'])

    rasmat = from_matvec(euler2mat(x=0.9, y=0.001, z=0.001), [4.0, 2.0, -1.0])
    itkxfm = itk.ITKLinearTransform.from_ras(rasmat)
    assert np.allclose(itkxfm['parameters'], ITK_MAT * rasmat)
    assert np.allclose(itkxfm.to_ras(), rasmat)


def test_ITKLinearTransformArray(tmpdir, data_path):
    tmpdir.chdir()

    with open(str(data_path / 'itktflist.tfm')) as f:
        text = f.read()
        f.seek(0)
        itklist = itk.ITKLinearTransformArray.from_fileobj(f)

    itklistb = itk.ITKLinearTransformArray.from_filename(
        data_path / 'itktflist.tfm')
    assert itklist['nxforms'] == itklistb['nxforms']
    assert all([np.allclose(x1['parameters'], x2['parameters'])
                for x1, x2 in zip(itklist.xforms, itklistb.xforms)])

    tmpdir.join('empty.mat').write('')
    with pytest.raises(TransformFileError):
        itklistb.from_filename('empty.mat')

    assert itklist['nxforms'] == 9
    assert text == itklist.to_string()
    with pytest.raises(TransformFileError):
        itk.ITKLinearTransformArray.from_string(
            '\n'.join(text.splitlines()[1:]))

    itklist.to_filename('copy.tfm')
    with open('copy.tfm') as f:
        copytext = f.read()
    assert text == copytext

    itklist = itk.ITKLinearTransformArray(
        xforms=[np.random.normal(size=(4, 4))
                for _ in range(4)])

    assert itklist['nxforms'] == 4
    assert itklist['xforms'][1].structarr['index'] == 1

    with pytest.raises(KeyError):
        itklist['invalid_key']

    xfm = itklist['xforms'][1]
    xfm['index'] = 1
    with open('extracted.tfm', 'w') as f:
        f.write(xfm.to_string())

    with open('extracted.tfm') as f:
        xfm2 = itk.ITKLinearTransform.from_fileobj(f)
    assert np.allclose(xfm.structarr['parameters'][:3, ...],
                       xfm2.structarr['parameters'][:3, ...])

    # ITK does not handle transforms lists in Matlab format
    with pytest.raises(TransformFileError):
        itklist.to_filename('matlablist.mat')

    with pytest.raises(TransformFileError):
        xfm2 = itk.ITKLinearTransformArray.from_binary({})

    with open('filename.mat', 'ab') as f:
        with pytest.raises(TransformFileError):
            xfm2 = itk.ITKLinearTransformArray.from_fileobj(f)


def test_LinearParameters(tmpdir):
    """Just pushes coverage up."""
    tmpdir.join('file.txt').write('')

    with pytest.raises(NotImplementedError):
        LinearParameters.from_string('')

    with pytest.raises(NotImplementedError):
        LinearParameters.from_fileobj(tmpdir.join('file.txt').open())


@pytest.mark.parametrize('matlab_ver', ['4', '5'])
def test_read_mat1(tmpdir, matlab_ver):
    """Test read from matlab."""
    tmpdir.chdir()

    savemat('val.mat', {'val': np.ones((3,))},
            format=matlab_ver)
    with open('val.mat', 'rb') as f:
        mdict = _read_mat(f)

    assert np.all(mdict['val'] == np.ones((3,)))


@pytest.mark.parametrize('matlab_ver', [-1] + list(range(2, 7)))
def test_read_mat2(tmpdir, monkeypatch, matlab_ver):
    """Check read matlab raises adequate errors."""
    from ..io import base

    tmpdir.chdir()
    savemat('val.mat', {'val': np.ones((3,))})

    def _mockreturn(arg):
        return (matlab_ver, 0)

    with monkeypatch.context() as m:
        m.setattr(base, 'get_matfile_version', _mockreturn)
        with pytest.raises(TransformFileError):
            with open('val.mat', 'rb') as f:
                _read_mat(f)


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


def test_itk_h5(testdata_path):
    """Test displacements fields."""
    assert len(list(itk.ITKCompositeH5.from_filename(
        testdata_path / 'ds-005_sub-01_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5'
    ))) == 2

    with pytest.raises(RuntimeError):
        list(itk.ITKCompositeH5.from_filename(
            testdata_path / 'ds-005_sub-01_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.x5'
        ))


@pytest.mark.parametrize('file_type, test_file', [
    (LTA, 'from-fsnative_to-scanner_mode-image.lta')
])
def test_regressions(file_type, test_file, data_path):
    file_type.from_filename(data_path / 'regressions' / test_file)
