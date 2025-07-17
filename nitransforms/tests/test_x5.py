import numpy as np
from h5py import File as H5File

from ..io.x5 import X5Transform, X5Domain, to_filename


def test_x5_transform_defaults():
    xf = X5Transform(
        type="linear",
        transform=np.eye(4),
        dimension_kinds=("space", "space", "space", "vector"),
    )
    assert xf.domain is None
    assert xf.subtype is None
    assert xf.representation is None
    assert xf.metadata is None
    assert xf.inverse is None
    assert xf.jacobian is None
    assert xf.array_length == 1
    # Disabled for now
    # assert xf.additional_parameters is None


def test_to_filename(tmp_path):
    domain = X5Domain(grid=True, size=(10, 10, 10), mapping=np.eye(4))
    node = X5Transform(
        type="linear",
        transform=np.eye(4),
        dimension_kinds=("space", "space", "space", "vector"),
        domain=domain,
    )
    fname = tmp_path / "test.x5"
    to_filename(fname, [node])

    with H5File(fname, "r") as f:
        assert f.attrs["Format"] == "X5"
        assert f.attrs["Version"] == 1
        grp = f["TransformGroup"]
        assert "0" in grp
        assert grp["0"].attrs["Type"] == "linear"
        assert grp["0"].attrs["ArrayLength"] == 1
