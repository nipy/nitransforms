import numpy as np
import matplotlib.pyplot as plt
import pytest

import nibabel as nb
from nitransforms.nonlinear import DenseFieldTransform
from nitransforms.vis import PlotDenseField


def test_read_path(data_path):
    """Check that filepaths are a supported method for loading and reading transforms with PlotDenseField"""
    PlotDenseField(transform = data_path / "ds-005_sub-01_from-OASIS_to-T1_warp_fsl.nii.gz")


def test_slice_values():
    """Check that ValueError is issued if negative slices are provided"""
    with pytest.raises(ValueError):
        PlotDenseField(
            transform = np.zeros((10, 10, 10, 3)),
            reference=nb.Nifti1Image(np.zeros((10, 10, 10, 3)), np.eye(4), None),
        ).test_slices(
            xslice=-1,
            yslice=-1,
            zslice=-1,
        )

    """Check that IndexError is issued if provided slices are beyond range of transform dimensions"""
    with pytest.raises(IndexError):
        xfm = DenseFieldTransform(
            field=np.zeros((10, 10, 10, 3)),
            reference=nb.Nifti1Image(np.zeros((10, 10, 10, 3)), np.eye(4), None),
        )
        PlotDenseField(
            transform=xfm._field,
            reference=xfm._reference,
        ).test_slices(
            xslice=xfm._field.shape[0]+1,
            yslice=xfm._field.shape[1]+1,
            zslice=xfm._field.shape[2]+1,
        )


def test_show_transform(data_path, output_path):
    PlotDenseField(
        transform = data_path / "ds-005_sub-01_from-OASIS_to-T1_warp_fsl.nii.gz"
    ).show_transform(
        xslice=50,
        yslice=50,
        zslice=50,
    )
    if output_path is not None:
        plt.savefig(output_path / "show_transform.svg", bbox_inches="tight")
    else:
        plt.show()


def test_plot_distortion(data_path, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    PlotDenseField(
        transform = data_path / "ds-005_sub-01_from-OASIS_to-T1_warp_fsl.nii.gz"
    ).plot_distortion(
        axes=axes,
        xslice=50,
        yslice=50,
        zslice=50,
    )
    if output_path is not None:
        plt.savefig(output_path / "show_transform.svg", bbox_inches="tight")
    else:
        plt.show()


def test_plot_quiverdsm(data_path, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    PlotDenseField(
        transform = data_path / "ds-005_sub-01_from-OASIS_to-T1_warp_fsl.nii.gz"
    ).plot_quiverdsm(
        axes=axes,
        xslice=50,
        yslice=50,
        zslice=50,
    )
    if output_path is not None:
        plt.savefig(output_path / "show_transform.svg", bbox_inches="tight")
    else:
        plt.show()


def test_3dquiver(data_path, output_path):
    with pytest.raises(NotImplementedError):
        fig = plt.figure()
        axes = plt.subplots(projection='3d')
        PlotDenseField(
            transform = data_path / "ds-005_sub-01_from-OASIS_to-T1_warp_fsl.nii.gz"
        ).plot_quiverdsm(axes=axes, three_D=True)

    if output_path is not None:
        plt.savefig(output_path / "show_transform.svg", bbox_inches="tight")
    else:
        plt.show()


def test_plot_jacobian(data_path, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    PlotDenseField(
        transform = data_path / "ds-005_sub-01_from-OASIS_to-T1_warp_fsl.nii.gz"
    ).plot_jacobian(
        axes=axes,
        xslice=50,
        yslice=50,
        zslice=50,
    )
    if output_path is not None:
        plt.savefig(output_path / "show_transform.svg", bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    pytest.main([__file__])
