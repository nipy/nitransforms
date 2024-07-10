import matplotlib.pyplot as plt
import pytest, unittest
from pathlib import Path

from nitransforms.vis import PlotDenseField

test_dir = Path("tests/data/")
test_file = Path("ds-005_sub-01_from-OASIS_to-T1_warp_fsl.nii.gz")

xfm = Path(test_dir/test_file)

def test_slice_values(xfm, xslice, yslice, zslice, is_deltas=True):
    PlotDenseField(
        path_to_file=Path(xfm),
        is_deltas=is_deltas, 
    ).test_slices(
        xslice=xslice,
        yslice=yslice,
        zslice=zslice,
    )

def test_show_transform(xfm, xslice=50, yslice=50, zslice=50, is_deltas=True):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    PlotDenseField(
        path_to_file=Path(xfm),
        is_deltas=is_deltas, 
    ).plot_quiverdsm(
        axes=axes,
        xslice=xslice,
        yslice=yslice,
        zslice=zslice,
    )
    plt.show()

test_slice_values(xfm, 50, -50, 50) #should raise ValueError
test_slice_values(xfm, 500, 50, 50) #should raise IndexError
test_show_transform(Path("tests/data/ds-005_sub-01_from-OASIS_to-T1_warp_fsl.nii.gz"))
