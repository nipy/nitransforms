# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
import nibabel as nb
import pytest

import nitransforms as nt

from nitransforms.analysis.utils import (
    compute_fd_from_motion,
    compute_fd_from_transform,
    displacements_within_mask,
    extract_motion_parameters,
    identify_spikes,
)


@pytest.fixture
def identity_affine():
    return np.eye(4)


@pytest.fixture
def simple_mask_img(identity_affine):
    # 3x3x3 mask with center voxel as 1, rest 0
    data = np.zeros((3, 3, 3), dtype=np.uint8)
    data[1, 1, 1] = 1
    return nb.Nifti1Image(data, identity_affine)


@pytest.fixture
def translation_transform():
    # Simple translation of (1, 2, 3) mm
    return nt.linear.Affine(map=np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 2],
        [0, 0, 1, 3],
        [0, 0, 0, 1],
    ]))


@pytest.fixture
def rotation_transform():
    # 90 degree rotation around z axis
    angle = np.pi / 2
    rot = np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    return nt.linear.Affine(map=rot)


@pytest.mark.parametrize(
    "test_xfm, reference_xfm, expected",
    [
        (nt.linear.Affine(np.eye(4)), None, np.zeros(1)),
        (nt.linear.Affine(np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1],
        ])), None, [np.linalg.norm([1, 2, 3])]),
        (nt.linear.Affine(np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1],
        ])), nt.linear.Affine(np.eye(4)), [np.linalg.norm([1, 2, 3])]),
    ],
)
def test_displacements_within_mask(simple_mask_img, test_xfm, reference_xfm, expected):
    disp = displacements_within_mask(simple_mask_img, test_xfm, reference_xfm)
    np.testing.assert_allclose(disp, expected)


@pytest.mark.parametrize(
    "test_xfm, expected",
    [
        (nt.linear.Affine(np.eye(4)), 0),
        (nt.linear.Affine(np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1],
        ])), np.linalg.norm([1, 2, 3])),
    ],
)
def test_compute_fd_from_transform(simple_mask_img, test_xfm, expected):
    fd = compute_fd_from_transform(simple_mask_img, test_xfm)
    assert np.isclose(fd, expected)


@pytest.mark.parametrize(
    "motion_params, radius, expected",
    [
        (np.zeros((5, 6)), 50, np.zeros(5)),  # 5 frames, 3 trans, 3 rot
        (
            np.array([
                [0,0,0,0,0,0],
                [2,0,0,0,0,0],  # 2mm translation in x at frame 1
                [2,0,0,90,0,0],
            ]),  # 90deg rotation in x at frame 2
            50,
            [0, 2, abs(np.deg2rad(90)) * 50]
        ),  # First frame: 0, Second: translation 2mm, Third: rotation (pi/2)*50
    ],
)
def test_compute_fd_from_motion(motion_params, radius, expected):
    fd = compute_fd_from_motion(motion_params, radius=radius)
    np.testing.assert_allclose(fd, expected, atol=1e-4)


@pytest.mark.parametrize(
    "affine, expected_trans, expected_rot",
    [
        (np.eye(4) + np.array([[0,0,0,10],[0,0,0,15],[0,0,0,20],[0,0,0,0]]),  # translation only
         [10, 15, 20], [0, 0, 0]),
        (np.array([
            [1, 0, 0, 0],
            [0, np.cos(np.deg2rad(30)), -np.sin(np.deg2rad(30)), 0],
            [0, np.sin(np.deg2rad(30)), np.cos(np.deg2rad(30)), 0],
            [0, 0, 0, 1],  # rotation only
        ]), [0, 0, 0], [30, 0, 0]),  # Only one rot will be close to 30
    ],
)
def test_extract_motion_parameters(affine, expected_trans, expected_rot):
    params = extract_motion_parameters(affine)
    assert np.allclose(params[:3], expected_trans)
    # For rotation case, at least one value close to 30
    if np.any(np.abs(expected_rot)):
        assert np.any(np.isclose(np.abs(params[3:]), 30, atol=1e-4))
    else:
        assert np.allclose(params[3:], expected_rot)


def test_identify_spikes(request):
    rng = request.node.rng

    n_samples = 450

    fd = rng.normal(0, 5, n_samples)
    threshold = 2.0

    expected_indices = np.asarray(
        [5, 57, 85, 100, 127, 180, 191, 202, 335, 393, 409]
    )
    expected_mask = np.zeros(n_samples, dtype=bool)
    expected_mask[expected_indices] = True

    obtained_indices, obtained_mask = identify_spikes(fd, threshold=threshold)

    assert np.array_equal(obtained_indices, expected_indices)
    assert np.array_equal(obtained_mask, expected_mask)
