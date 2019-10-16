import numpy as np
from nibabel.affines import from_matvec
from nibabel.eulerangles import euler2mat
from ..patched import obliquity


def test_obliquity():
    """Check the calculation of inclination of an affine axes."""
    from math import pi
    aligned = np.diag([2.0, 2.0, 2.3, 1.0])
    aligned[:-1, -1] = [-10, -10, -7]
    R = from_matvec(euler2mat(x=0.09, y=0.001, z=0.001), [0.0, 0.0, 0.0])
    oblique = R.dot(aligned)
    np.testing.assert_almost_equal(obliquity(aligned), [0.0, 0.0, 0.0])
    np.testing.assert_almost_equal(obliquity(oblique) * 180 / pi,
                                   [0.0810285, 5.1569949, 5.1569376])
