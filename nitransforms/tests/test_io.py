# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""I/O test cases."""
import numpy as np
import pytest

from nibabel.eulerangles import euler2mat
from nibabel.affines import from_matvec
from scipy.io import loadmat, savemat
from ..io import (
    itk,
    VolumeGeometry as VG,
    LinearTransform as LT,
    LinearTransformArray as LTA,
)
from ..io.base import _read_mat, TransformFileError

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

    test_lta = str(data_path / 'inv.lta')
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


def test_ITKLinearTransform(tmpdir, data_path):
    tmpdir.chdir()

    matlabfile = str(data_path / 'ds-005_sub-01_from-T1_to-OASIS_affine.mat')
    mat = loadmat(matlabfile)
    with open(matlabfile, 'rb') as f:
        itkxfm = itk.ITKLinearTransform.from_fileobj(f)
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


def test_ITKLinearTransformArray(tmpdir, data_path):
    tmpdir.chdir()

    with open(str(data_path / 'itktflist.tfm')) as f:
        text = f.read()
        f.seek(0)
        itklist = itk.ITKLinearTransformArray.from_fileobj(f)

    assert itklist['nxforms'] == 9
    assert text == itklist.to_string()
    with pytest.raises(ValueError):
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


@pytest.mark.parametrize('matlab_ver', ['4', '5'])
def test_read_mat1(tmpdir, matlab_ver):
    """Test read from matlab."""
    tmpdir.chdir()

    savemat('val.mat', {'val': np.ones((3,))},
            format=matlab_ver)
    with open('val.mat', 'rb') as f:
        mdict = _read_mat(f)

    assert np.all(mdict['val'] == np.ones((3,)))


def test_read_mat2(tmpdir, monkeypatch):
    """Check read matlab raises adequate errors."""
    from ..io import base

    def _mockreturn(arg):
        return (2, 0)

    tmpdir.chdir()
    savemat('val.mat', {'val': np.ones((3,))})

    with monkeypatch.context() as m:
        m.setattr(base, 'get_matfile_version', _mockreturn)
        with pytest.raises(TransformFileError):
            with open('val.mat', 'rb') as f:
                _read_mat(f)
