# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""I/O test cases."""
import numpy as np

from ..io import (
    itk,
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
    v2v.set_type('LINEAR_RAS_TO_RAS')
    assert v2v['type'] == 1
    assert np.allclose(r2r_m, v2v_m, atol=1e-05)


def test_ITKLinearTransformArray(tmpdir, data_path):
    tmpdir.chdir()

    with open(str(data_path / 'itktflist.tfm')) as f:
        text = f.read()
        f.seek(0)
        itklist = itk.ITKLinearTransformArray.from_fileobj(f)

    assert itklist['nxforms'] == 9
    assert text == itklist.to_string()

    itklist = itk.ITKLinearTransformArray(
        xforms=[np.around(np.random.normal(size=(4, 4)), decimals=5)
                for _ in range(4)])

    assert itklist['nxforms'] == 4
    assert itklist['xforms'][1].structarr['index'] == 2

    xfm = itklist['xforms'][1]
    xfm['index'] = 1
    with open('extracted.tfm', 'w') as f:
        f.write(xfm.to_string())

    with open('extracted.tfm') as f:
        xfm2 = itk.ITKLinearTransform.from_fileobj(f)
    assert np.allclose(xfm.structarr['parameters'][:3, ...],
                       xfm2.structarr['parameters'][:3, ...])
