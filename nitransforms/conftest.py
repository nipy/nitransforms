"""py.test configuration."""
import os
from pathlib import Path
import numpy as np
import nibabel as nb
import pytest
import tempfile


@pytest.fixture(autouse=True)
def doctest_autoimport(doctest_namespace):
    """Make available some fundamental modules to doctest modules."""
    doctest_namespace['np'] = np
    doctest_namespace['nb'] = nb
    doctest_namespace['os'] = os
    doctest_namespace['Path'] = Path
    doctest_namespace['datadir'] = os.path.join(os.path.dirname(__file__), 'tests/data')

    tmpdir = tempfile.TemporaryDirectory()
    doctest_namespace['tmpdir'] = tmpdir.name

    testdata = np.zeros((11, 11, 11), dtype='uint8')
    nifti_fname = str(Path(tmpdir.name) / 'test.nii.gz')
    nb.Nifti1Image(testdata, np.eye(4)).to_filename(nifti_fname)
    doctest_namespace['testfile'] = nifti_fname
    yield
    tmpdir.cleanup()
