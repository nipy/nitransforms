# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import re

import numpy as np
import nibabel as nb
import pytest
from scipy.spatial.transform import Rotation as R

import nitransforms as nt

from nitransforms.analysis.utils import (
    AFFINE_SHAPE_ERROR,
    AFFINE_SEQ_SHAPE_ERROR,
    AFFINE_TYPE_ERROR_MSG,
    MOTION_FORMAT_AFNI,
    MOTION_FORMAT_FSL,
    MOTION_PARAMS_SHAPE_ERROR_MSG,
    MOTION_PARAMS_FMT_ERROR_MSG,
    MOTION_PARAMS_INST_ERROR_MSG,
    ROTATIONS_SHAPE_ERROR_MSG,
    TRANSLATIONS_ERROR_MSG,
    MotionParameters,
    affine_to_motion_params,
    compute_fd_from_motion,
    compute_fd_from_transform,
    displacements_within_mask,
    extract_motion_parameters,
    motion_params_to_affine,
)
from nitransforms.linear import Affine, LinearTransformsMapping


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
    "translations, rotations, match_msg",
    [
        (np.zeros((10, 6)), np.zeros((10, 3)), TRANSLATIONS_ERROR_MSG),
        (np.zeros((10, 3)), np.zeros((10, 6)), ROTATIONS_SHAPE_ERROR_MSG),
        (np.zeros((10,)), np.zeros((10, 3)), TRANSLATIONS_ERROR_MSG),
        (np.zeros((10, 3)), np.zeros((10,)), ROTATIONS_SHAPE_ERROR_MSG),
    ],
)
def test_motion_parameters_validation(translations, rotations, match_msg):
    with pytest.raises(ValueError, match=re.escape(match_msg)):
        MotionParameters(translations, rotations)


def test_motion_parameters_creation():
    t = np.zeros((10, 3))
    r = np.ones((10, 3))
    params = MotionParameters(t, r)

    assert params.translations.shape == (10, 3)
    assert params.rotations.shape == (10, 3)

    # Test that it supports tuple unpacking
    translations, rotations = params
    assert np.array_equal(translations, t)
    assert np.array_equal(rotations, r)


def test_extract_motion_parameters_invalid_format():
    arr = np.random.rand(5, 6)
    fmt = "unknown_format"
    with pytest.raises(
        ValueError, match=re.escape(MOTION_PARAMS_FMT_ERROR_MSG.format(fmt=fmt))
    ):
        extract_motion_parameters(arr, fmt=fmt)


@pytest.mark.parametrize(
    "invalid_arr",
    [
        np.zeros((5, 5)),
        np.zeros((5,)),
        np.zeros((2, 3, 3)),
    ],
)
def test_extract_motion_parameters_shape_validation(invalid_arr):
    with pytest.raises(ValueError, match=re.escape(MOTION_PARAMS_SHAPE_ERROR_MSG)):
        extract_motion_parameters(invalid_arr)


@pytest.mark.parametrize(
    "fmt, expected_conversion",
    [
        (None, lambda r: r),
        (MOTION_FORMAT_FSL, lambda r: r),
        (MOTION_FORMAT_AFNI, lambda r: np.deg2rad(r)),
    ],
)
def test_extract_motion_parameters_fmt(fmt, expected_conversion):
    arr = np.random.rand(10, 6)
    translations_expected = arr[:, :3]
    rotations_raw = arr[:, 3:]

    params = extract_motion_parameters(arr, fmt=fmt)

    assert isinstance(params, MotionParameters)
    assert np.allclose(params.translations, translations_expected)
    assert np.allclose(params.rotations, expected_conversion(rotations_raw))


@pytest.mark.parametrize(
    "invalid_input, expected_exception, expected_match",
    [
        (np.zeros((4,)), ValueError, AFFINE_SEQ_SHAPE_ERROR),
        (np.zeros((3, 3)), ValueError, AFFINE_SHAPE_ERROR),
        (np.zeros((2, 3, 3)), ValueError, AFFINE_SEQ_SHAPE_ERROR),
        ("not_an_affine", TypeError, AFFINE_TYPE_ERROR_MSG),
    ],
)
def test_affine_to_motion_params_validation(invalid_input, expected_exception, expected_match):
    with pytest.raises(expected_exception, match=re.escape(expected_match)):
        affine_to_motion_params(invalid_input)


def test_motion_params_to_affine_type_error():
    with pytest.raises(TypeError, match=MOTION_PARAMS_INST_ERROR_MSG):
        motion_params_to_affine("not_motion_params")


@pytest.mark.parametrize("num_frames", [1, 5])
def test_affine_motion_params_roundtrip(num_frames):
    # Create valid synthetic translations and pure rotations via euler angles
    translations = np.random.uniform(-5.0, 5.0, size=(num_frames, 3))
    rotations = np.random.uniform(-0.1, 0.1, size=(num_frames, 3))  # small angles

    original_params = MotionParameters(translations=translations, rotations=rotations)

    # Convert to mapping
    mapping = motion_params_to_affine(original_params)
    assert isinstance(mapping, LinearTransformsMapping)
    assert len(mapping) == num_frames

    # Convert back to motion parameters
    # mapping.matrix returns the batch array of shape (T, 4, 4)
    recovered_params = affine_to_motion_params(mapping.matrix)

    assert np.allclose(recovered_params.translations, original_params.translations, atol=1e-5)
    assert np.allclose(recovered_params.rotations, original_params.rotations, atol=1e-5)


@pytest.mark.parametrize(
    "input_shape, input_type",
    [
        ((4, 4), "ndarray"),
        ((3, 4, 4), "ndarray"),
        ((1, 4, 4), "ndarray"),
        ((4, 4), "affine"),
        ((3, 4, 4), "mapping"),
    ],
)
def test_affine_to_motion_params_misc_data(input_shape, input_type):
    if len(input_shape) == 2:
        mat = np.eye(4)
        mat[:3, 3] = [1.0, 2.0, 3.0]
    else:
        T = input_shape[0]
        mat = np.tile(np.eye(4), (T, 1, 1))

    if input_type == "ndarray":
        affine_in = mat
    elif input_type == "affine":
        affine_in = Affine(mat)
    elif input_type == "mapping":
        affine_in = LinearTransformsMapping([Affine(m) for m in mat])

    params = affine_to_motion_params(affine_in)

    expected_T = input_shape[0] if len(input_shape) == 3 else 1
    assert isinstance(params, MotionParameters)
    assert params.translations.shape == (expected_T, 3)
    assert params.rotations.shape == (expected_T, 3)


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
        ]), [0, 0, 0], [np.deg2rad(30), 0, 0]),  # Only one rot will be close to 30
    ],
)
def test_affine_to_motion_params(affine, expected_trans, expected_rot):
    params = affine_to_motion_params(affine)

    # Since single affine results in T=1, check the first row [0]
    assert np.allclose(params.translations[0], expected_trans)

    # For rotation case, verify the values
    if np.any(np.abs(expected_rot)):
        assert np.any(np.isclose(np.abs(params.rotations[0]), np.deg2rad(30), atol=1e-4))
    else:
        assert np.allclose(params.rotations[0], expected_rot)


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


def test_compute_fd_from_transform_exceptions():
    img = nb.Nifti1Image(np.zeros((5, 5, 5), dtype=float), np.eye(4))
    xfm = nt.linear.Affine(np.eye(4))
    with pytest.raises(ValueError, match=r"n_vertices must be >= 1"):
        compute_fd_from_transform(img=img, xfm=xfm, n_vertices=0)


@pytest.mark.parametrize(
    "test_xfm, expected",
    [
        (nt.linear.Affine(np.eye(4)), 0),
        (nt.linear.Affine(np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1],
        ])), np.linalg.norm([1, 2, 3], ord=1)),  # L1 norm of translation displacement
    ],
)
def test_compute_fd_from_transform(simple_mask_img, test_xfm, expected):
    fd = compute_fd_from_transform(simple_mask_img, test_xfm)
    assert np.isclose(fd, expected, atol=1e-4, rtol=1e-6)


def test_compute_fd_from_motion_exceptions():
    arr = np.zeros((4, 4), dtype=float)
    with pytest.raises(TypeError, match=MOTION_PARAMS_INST_ERROR_MSG):
        compute_fd_from_motion(arr)


def test_compute_fd_from_motion_single_vertex_variants():
    """For n_vertices=1, FD equals the L1 displacement of the single sampled point."""
    radius = 50.0

    # One-step motion parameters: [tx, ty, tz, rx, ry, rz] (deg)
    motion_arr = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, -2.0, 0.5, 1.5, -0.5, 0.25],
    ])

    # Convert raw array (with rotations in degrees) into MotionParameters
    motion_params = extract_motion_parameters(motion_arr, fmt=MOTION_FORMAT_AFNI)

    # Expected from canonical function
    fd_motion = compute_fd_from_motion(motion_params, radius=radius)[1]

    # Build matching transforms (prev identity, current from same params)
    t = motion_arr[1, :3]
    r_deg = motion_arr[1, 3:]
    rot = R.from_euler("xyz", r_deg, degrees=True).as_matrix()

    M_prev = np.eye(4)
    M_curr = np.eye(4)
    M_curr[:3, :3] = rot
    M_curr[:3, 3] = t

    xfm_prev = nt.linear.Affine(M_prev)
    xfm = nt.linear.Affine(M_curr)

    img = nb.Nifti1Image(np.zeros((5, 5, 5), dtype=np.float32), np.eye(4))
    fd_xfm = compute_fd_from_transform(
        img, xfm, xfm_prev=xfm_prev, radius=radius, n_vertices=1
    )

    assert np.isclose(fd_xfm, fd_motion, atol=1e-6)


@pytest.mark.parametrize(
    "motion_arr, radius, expected",
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
def test_compute_fd_from_motion(motion_arr, radius, expected):
    # Wrap with extract_parameters using AFNI format to handle
    # degree-to-radian conversion
    motion_params = extract_motion_parameters(motion_arr, fmt=MOTION_FORMAT_AFNI)
    fd = compute_fd_from_motion(motion_params, radius=radius)

    # Verify output shape matches the expected number of frames
    assert fd.shape == (len(expected),)
    # Verify the initial frame has zero displacement
    assert fd[0] == 0.0

    # Comprehensive value check
    np.testing.assert_allclose(fd, expected, atol=1e-4)
