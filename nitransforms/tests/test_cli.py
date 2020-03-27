from textwrap import dedent

import pytest

from ..cli import cli_apply, main as ntcli


def test_cli(capsys):
    # empty command
    with pytest.raises(SystemExit):
        ntcli()
    # invalid command
    with pytest.raises(SystemExit):
        ntcli(['idk'])

    with pytest.raises(SystemExit) as sysexit:
        ntcli(['-h'])
    console = capsys.readouterr()
    assert sysexit.value.code == 0
    # possible commands
    assert r"{apply}" in console.out

    with pytest.raises(SystemExit):
        ntcli(['apply', '-h'])
    console = capsys.readouterr()
    assert dedent(cli_apply.__doc__) in console.out
    assert sysexit.value.code == 0


def test_apply_linear(tmpdir, data_path, get_testdata):
    tmpdir.chdir()
    img = 'img.nii.gz'
    get_testdata['RAS'].to_filename(img)
    lin_xform = str(data_path / 'affine-RAS.itk.tfm')
    lin_xform2 = str(data_path / 'affine-RAS.fs.lta')

    # unknown transformation format
    with pytest.raises(ValueError):
        ntcli(['apply', 'unsupported.xform', 'img.nii.gz'])

    # linear transform arguments
    output = tmpdir / 'nt_img.nii.gz'
    ntcli(['apply', lin_xform, img, '--ref', img])
    assert output.check()
    output.remove()
    ntcli(['apply', lin_xform2, img, '--ref', img])
    assert output.check()


def test_apply_nl(tmpdir, testdata_path):
    tmpdir.chdir()
    img = str(testdata_path / 'tpl-OASIS30ANTs_T1w.nii.gz')
    nl_xform = str(testdata_path / 'ds-005_sub-01_from-OASIS_to-T1_warp_afni.nii.gz')

    nlargs = ['apply', nl_xform, img]
    # format not specified
    with pytest.raises(ValueError):
        ntcli(nlargs)

    nlargs.extend(['--fmt', 'afni'])
    # no linear afni support
    with pytest.raises(NotImplementedError):
        ntcli(nlargs)

    output = 'moved_from_warp.nii.gz'
    nlargs.extend(['--nonlinear', '--out', output])
    ntcli(nlargs)
    assert (tmpdir / output).check()
