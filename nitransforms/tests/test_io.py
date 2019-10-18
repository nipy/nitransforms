import os

import numpy as np

from ..io import LinearTransformArray as LTA


def test_LinearTransformArray_input(tmpdir, data_path):
    lta = LTA()
    assert lta['nxforms'] == 0
    assert len(lta['xforms']) == 0

    test_lta = os.path.join(data_path, 'inv.lta')
    with open(test_lta) as fp:
        lta = LTA.from_fileobj(fp)

    assert lta.get('type') == 1
    assert len(lta['xforms']) == lta['nxforms'] == 1
    xform = lta['xforms'][0]

    assert np.allclose(
        xform['m_L'], np.genfromtxt(test_lta, skip_header=5, skip_footer=20)
    )
