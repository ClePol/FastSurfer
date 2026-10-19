"""Integration coverage for CC geometry, probability padding and template metadata."""

import json

import numpy as np
import pytest
import torch

from CorpusCallosum.data.read_write import convert_numpy_to_json_serializable
from CorpusCallosum.segmentation.inference import run_inference
from CorpusCallosum.shape.contour import CCContour, calculate_volume, contours_for_analysis_width
from CorpusCallosum.shape.mesh import CCMesh


@pytest.mark.parametrize("reverse", [False, True])
def test_mesh_upsampling_keeps_outward_normals_and_analysis_width(reverse):
    """Unequal contour sampling yields outward faces for either slab direction."""
    square = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    dense = np.insert(square, 1, [0., -1.], axis=0)
    contours = [
        CCContour(points.copy(), np.ones(len(points)), lr_position=position)
        for points, position in [(square, 1.), (dense, 0.), (square, -1.)]
    ]
    if reverse:
        contours.reverse()
    mesh = CCMesh.from_contours(contours)
    centers = mesh.v[mesh.t].mean(axis=1)
    normals = np.cross(mesh.v[mesh.t[:, 1]] - mesh.v[mesh.t[:, 0]],
                       mesh.v[mesh.t[:, 2]] - mesh.v[mesh.t[:, 0]])
    assert np.all(np.sum(centers[:, 1:] * normals[:, 1:], axis=1) > 0)
    np.testing.assert_allclose(mesh.mesh_vertex_colors, 1.)
    display_contours = contours_for_analysis_width(contours)
    np.testing.assert_allclose([c.lr_position for c in display_contours], [2.5, 0., -2.5])
    np.testing.assert_allclose([c.lr_position for c in contours], [-1., 0., 1.] if reverse else [1., 0., -1.])


@pytest.mark.parametrize("reverse", [False, True])
def test_contour_volume_uses_physical_slice_positions(reverse):
    """The five millimeter slab integrates the same area in either slice order."""
    points = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    positions = np.arange(-2., 3.)
    if reverse:
        positions = positions[::-1]
    contours = [CCContour(points.copy(), None, lr_position=position) for position in positions]
    assert calculate_volume(contours) == pytest.approx(20.)


def test_padded_softlabels_are_background_probabilities():
    """The cropped model output retains a normalized background outside its field of view."""
    class UniformModel(torch.nn.Module):
        def forward(self, inputs, scale_factor):
            return torch.zeros((len(inputs), 3, *inputs.shape[-2:]), device=inputs.device)

    labels, probabilities, crop_box = run_inference(
        UniformModel(), np.ones((9, 128, 128), dtype=np.float32),
        np.array([0., 64., 45.]), np.array([0., 64., 80.]), (1., 1.), device=torch.device("cpu"),
    )
    inside = np.zeros(probabilities.shape[:-1], dtype=bool)
    (i_start, i_stop), (a_start, a_stop) = crop_box
    inside[:, i_start:i_stop, a_start:a_stop] = True
    assert np.any(inside) and np.any(~inside)
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.)
    np.testing.assert_allclose(probabilities[~inside], np.broadcast_to([1., 0., 0.], probabilities[~inside].shape))
    np.testing.assert_array_equal(labels[~inside], probabilities[~inside])


def test_partial_metrics_produce_strict_json():
    """Failed or undefined measurements serialize as null throughout nested metrics."""
    metrics = {"slices": [None, {"curvature": np.array([np.nan, np.inf, -np.inf, 2.])}],
               "selected": np.int64(2), "complete": np.bool_(False)}
    encoded = json.dumps(convert_numpy_to_json_serializable(metrics), allow_nan=False)
    assert json.loads(encoded) == {"slices": [None, {"curvature": [None, None, None, 2.]}],
                                   "selected": 2, "complete": False}
