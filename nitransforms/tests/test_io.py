import os

import numpy as np

from ..io import (
    VolumeGeometry as VG,
    LinearTransform as LT,
    LinearTransformArray as LTA,
)


def test_VolumeGeometry(tmpdir, get_testdata):
    vg = VG()
    assert vg['valid'] == 0

    img = get_testdata['RAS']
    vg = VG.from_image(img)
    assert vg['valid'] == 1
    assert np.all(vg['voxelsize'] == img.header.get_zooms()[:3])
    assert np.all(vg.as_affine() == img.affine)

    assert len(vg.to_string().split('\n')) == 8


def test_LinearTransform(tmpdir, get_testdata):
    lt = LT()
    assert lt['m_L'].shape == (4, 4)
    assert np.all(lt['m_L'] == 0)
    for vol in ('src', 'dst'):
        assert lt[vol]['valid'] == 0


def test_LinearTransformArray(tmpdir, data_path):
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

    outlta = (tmpdir / 'out.lta').strpath
    with open(outlta, 'w') as fp:
        fp.write(lta.to_string())

    with open(outlta) as fp:
        lta2 = LTA.from_fileobj(fp)
    assert np.allclose(lta['xforms'][0]['m_L'], lta2['xforms'][0]['m_L'])
