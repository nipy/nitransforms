"""py.test configuration."""
import os
from pathlib import Path
import numpy as np
import nibabel as nb
import pytest
import tempfile

_data = None
_testdir = Path(os.getenv('TEST_DATA_HOME', '~/.nitransforms/testdata')).expanduser()


@pytest.fixture(autouse=True)
def doctest_autoimport(doctest_namespace):
    """Make available some fundamental modules to doctest modules."""
    doctest_namespace['np'] = np
    doctest_namespace['nb'] = nb
    doctest_namespace['os'] = os
    doctest_namespace['Path'] = Path
    doctest_namespace['regress_dir'] = Path(__file__).parent / 'tests' / 'data'
    doctest_namespace['test_dir'] = _testdir

    tmpdir = tempfile.TemporaryDirectory()
    doctest_namespace['tmpdir'] = tmpdir.name

    testdata = np.zeros((11, 11, 11), dtype='uint8')
    nifti_fname = str(Path(tmpdir.name) / 'test.nii.gz')
    nb.Nifti1Image(testdata, np.eye(4)).to_filename(nifti_fname)
    doctest_namespace['testfile'] = nifti_fname
    yield
    tmpdir.cleanup()


@pytest.fixture
def data_path():
    """Return the test data folder."""
    return Path(__file__).parent / 'tests' / 'data'


@pytest.fixture
def testdata_path():
    """Return the heavy test-data folder."""
    return _testdir


@pytest.fixture
def get_testdata():
    """Generate data in the requested orientation."""
    global _data

    if _data is not None:
        return _data

    img = nb.load(_testdir / 'someones_anatomy.nii.gz')
    imgaff = img.affine

    _data = {'RAS': img}
    newaff = imgaff.copy()
    newaff[0, 0] *= -1.0
    newaff[0, 3] = imgaff.dot(np.hstack((np.array(img.shape[:3]) - 1, 1.0)))[0]
    _data['LAS'] = nb.Nifti1Image(np.flip(img.get_fdata(), 0), newaff, img.header)
    newaff = imgaff.copy()
    newaff[0, 0] *= -1.0
    newaff[1, 1] *= -1.0
    newaff[:2, 3] = imgaff.dot(np.hstack((np.array(img.shape[:3]) - 1, 1.0)))[:2]
    _data['LPS'] = nb.Nifti1Image(
        np.flip(np.flip(img.get_fdata(), 0), 1), newaff, img.header
    )
    A = nb.volumeutils.shape_zoom_affine(
        img.shape, img.header.get_zooms(), x_flip=False
    )
    R = nb.affines.from_matvec(nb.eulerangles.euler2mat(x=0.09, y=0.001, z=0.001))
    newaff = R.dot(A)
    oblique_img = nb.Nifti1Image(img.get_fdata(), newaff, img.header)
    oblique_img.header.set_qform(newaff, 1)
    oblique_img.header.set_sform(newaff, 1)
    _data['oblique'] = oblique_img

    return _data
