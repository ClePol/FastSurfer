# Copyright 2025 DeepMI Lab, German Center for Neurodegenerative Diseases(DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path
from typing import overload

import numpy as np
import torch
from monai import transforms

from CorpusCallosum.data import constants
from CorpusCallosum.transforms.segmentation import CropAroundACPC
from FastSurferCNN.models.networks import FastSurferVINN
from FastSurferCNN.utils import Image3d, Image4d, Shape2d, Shape3d, Shape4d, Vector2d
from FastSurferCNN.utils.checkpoint import (
    get_checkpoints,
    get_config_file,
    load_checkpoint_config_defaults,
)

CropBox = tuple[tuple[int, int], tuple[int, int]]


def load_model(device: torch.device | None = None) -> FastSurferVINN:
    """Load trained model from checkpoint.

    Parameters
    ----------
    device : torch.device or None, optional
        Device to load model to, by default None.
        If None, uses CUDA if available, else CPU.

    Returns
    -------
    FastSurferVINN
        Loaded and initialized model in evaluation mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = {
        "num_classes": 3,
        "num_filters": 71,
        "num_filters_interpol": 32,
        "num_channels": 9,
        "kernel_h": 3,
        "kernel_w": 3,
        "kernel_c": 1,
        "stride_conv": 1,
        "stride_pool": 2,
        "pool": 2,
        "height": 128,
        "width": 128,
        "base_res": 1.0,
        "interpolation_mode": "bilinear",
        "crop_position": "top_left",
        "out_tensor_width": 320,
        "out_tensor_height": 320,
    }
    model = FastSurferVINN(params)
    
    config_file = get_config_file("CorpusCallosum")
    cc_config: dict[str, Path] = load_checkpoint_config_defaults("checkpoint", filename=config_file)
    urls = load_checkpoint_config_defaults("url", filename=config_file)
    checkpoint_path = constants.FASTSURFER_ROOT / cc_config['segmentation']
    get_checkpoints(checkpoint_path, urls=urls)
    
    weights = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model.load_state_dict(weights)
    model.eval()
    model.to(device)
    return model


def run_inference(
    model: "torch.nn.Module",
    image_slice: Image3d,
    ac_center: Vector2d,
    pc_center: Vector2d,
    voxel_size: tuple[float, float],
    device: torch.device | None = None,
) -> tuple[np.ndarray[Shape4d, np.dtype[np.int_]], Image4d, CropBox]:
    """Run inference on a single image slice.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    image_slice : np.ndarray
        LIA-oriented input image as numpy array of shape (L, I, A).
    ac_center : np.ndarray
        Anterior commissure coordinates.
    pc_center : np.ndarray
        Posterior commissure coordinates.
    voxel_size : a pair of floats
        Voxel size of inferior/superior and anterior/posterior direction in mm.
    device : torch.device, optional
        Device to run inference on. If None, uses the device of the model.

    Returns
    -------
    seg_labels : np.ndarray of int
        The segmentation result.
    soft_labels : np.ndarray of float
        The softlabel output.
    crop_box : pair of pairs of ints
        The region of the image that was passed to the model as ((start_I, stop_I), (start_A, stop_A)), outputs outside
        of this crop box are padded as background (one-hot/probability of 1 for the background channel 0).
    """
    if device is None:
        device = next(model.parameters()).device
        
    crop_around_acpc = CropAroundACPC(keys=['image'], padding_mm=35, random_translate=0)
    to_discrete = transforms.AsDiscrete(argmax=True, to_onehot=3)

    # Preprocess slice
    _inputs = torch.from_numpy(image_slice[:,None])
    sample = {'image': _inputs, 'AC_center': ac_center, 'PC_center': pc_center, 'res': np.asarray(voxel_size)}
    sample_cropped = crop_around_acpc(sample)
    _inputs, to_pad = sample_cropped['image'], sample_cropped['to_pad']
    _inputs = transforms.utils.rescale_array(_inputs, 0, 1, dtype=np.float32).to(device)

    # split into slices with 9 channels each
    # Generate views with sliding window of 9 slices
    batch_size, channels, height, width = _inputs.shape
    _inputs = _inputs.unfold(0, 9, 1).swapdims(-1, 1).reshape(-1, 9*channels, height, width)

    # Post-process outputs
    with torch.no_grad():
        scale_factors = torch.ones((_inputs.shape[0], 2), device=device) / torch.asarray([voxel_size], device=device)
        
        _logits = model(_inputs, scale_factor=scale_factors)
        _softlabels = transforms.Activations(softmax=True, dim=1)(_logits)
        
        softlabels = _softlabels.cpu().numpy()
        _labels = torch.stack([to_discrete(i) for i in _softlabels])

        # Pad back to original size, to_pad is a tuple[int, int, int, int]
        pad_tuples = ((0, 0),) * 2 + (to_pad[:2], to_pad[2:])
        labels = np.pad(_labels.cpu().numpy(), pad_tuples, mode='constant', constant_values=0)
        softlabels = np.pad(softlabels, pad_tuples, mode='constant', constant_values=0)
        # outside the crop box, voxels are background with certainty, i.e. probability 1 for channel 0 (background)
        outside_crop_box = np.pad(np.zeros(_softlabels.shape[2:], dtype=bool), pad_tuples[2:], constant_values=True)
        labels[:, 0, outside_crop_box] = 1
        softlabels[:, 0, outside_crop_box] = 1.

    # the region of the image (in I and A) that was actually segmented, outside the outputs are padded as background
    crop_box: CropBox = (
        (int(to_pad[0]), int(image_slice.shape[1] - to_pad[1])),
        (int(to_pad[2]), int(image_slice.shape[2] - to_pad[3])),
    )
    labels, softlabels = (x.transpose(0, 2, 3, 1) for x in (labels, softlabels))
    return labels, softlabels, crop_box


@overload
def one_hot_to_label(one_hot: Image4d, label_ids: list[int] | None = None) \
        -> np.ndarray[Shape3d, np.dtype[np.int_]]: ...

@overload
def one_hot_to_label(one_hot: Image3d, label_ids: list[int] | None = None) \
        -> np.ndarray[Shape2d, np.dtype[np.int_]]: ...

def one_hot_to_label(
    one_hot: np.ndarray[tuple[int, ...], np.dtype[np.bool_]],
    label_ids: list[int] | None = None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.int_]]:
    """Convert one-hot encoded segmentation to label map.

    Converts a one-hot encoded segmentation array to discrete labels by taking
    the argmax along the last axis and optionally mapping to specific label values.

    Parameters
    ----------
    one_hot : np.ndarray of floats
        One-hot encoded segmentation array of shape (..., num_classes).
    label_ids : array_like of ints, optional
        List of label IDs to map classes to. If None, defaults to [0, CC_LABEL, FORNIX_LABEL].
        The index in this list corresponds to the class index from argmax.

    Returns
    -------
    np.ndarray of int
        Label map with discrete integer labels.
    """
    if label_ids is None:
        from CorpusCallosum.data.constants import CC_LABEL, FORNIX_LABEL
        label_ids = [0, CC_LABEL, FORNIX_LABEL]

    label = np.argmax(one_hot, axis=-1)
    return np.asarray(label_ids)[label]


def run_inference_on_slice(
        model: "torch.nn.Module",
        test_slab: Image3d,
        ac_center: Vector2d,
        pc_center: Vector2d,
        voxel_size: tuple[float, float],
) -> tuple[np.ndarray[Shape3d, np.dtype[np.int_]], Image4d, CropBox]:
    """Run inference on a single slice.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model for inference.
    test_slab : np.ndarray
        Input image slice.
    ac_center : np.ndarray of float
        Anterior commissure coordinates (Inferior and Anterior values).
    pc_center : np.ndarray of float
        Posterior commissure coordinates (Inferior and Anterior values).
    voxel_size : a pair of floats
        Voxel sizes in superior/inferior and anterior/posterior direction in mm.

    Returns
    -------
    results: np.ndarray
        Label map after one-hot conversion.
    outputs_soft: np.ndarray of float
        Softlabel outputs (non-discrete).
    crop_box : pair of pairs of ints
        The region of the image that was segmented as ((start_I, stop_I), (start_A, stop_A)), outputs outside of this
        crop box are padded as background (results are 0, softlabels are 1 for background and 0 otherwise).
    """
    # add zero in front of AC_center and PC_center
    ac_center = np.concatenate([np.zeros(1), ac_center])
    pc_center = np.concatenate([np.zeros(1), pc_center])

    _results, outputs_soft, crop_box = run_inference(model, test_slab, ac_center, pc_center, voxel_size)
    results = one_hot_to_label(_results)

    return results, outputs_soft, crop_box
