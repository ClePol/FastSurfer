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

from typing import Literal, cast

import numpy as np
import torch
from monai import transforms
from monai.networks.nets import DenseNet

from CorpusCallosum.transforms.localization import CropAroundACPCFixedSize
from CorpusCallosum.utils.types import Points2dType
from FastSurferCNN.utils import Image3d, Vector2d, Vector3d
from FastSurferCNN.utils.checkpoint import (
    get_checkpoints,
    get_config_file,
    load_checkpoint_config_defaults,
)
from FastSurferCNN.utils.parser_defaults import FASTSURFER_ROOT

PATCH_SIZE = (64, 64)


def load_model(device: torch.device) -> DenseNet:
    """Load trained numerical localization model from checkpoint.

    Parameters
    ----------
    device : torch.device
        Device to load model to.

    Returns
    -------
    DenseNet
        Loaded and initialized model in evaluation mode.
    """

    # Initialize model architecture (must match training)
    model = DenseNet( # densenet201
        spatial_dims=2,
        in_channels=3,
        out_channels=4,
        init_features=64,
        growth_rate=32,
        block_config=(6, 12, 48, 32),
        bn_size=4,
        act=("relu", {"inplace": True}),
        norm=("batch", {"affine": True}),
        dropout_prob=0.2
    )

    config_file = get_config_file("CorpusCallosum")
    cc_config = load_checkpoint_config_defaults("checkpoint", filename=config_file)
    urls = load_checkpoint_config_defaults("url", filename=config_file)
    checkpoint_path = FASTSURFER_ROOT / cc_config['localization']
    get_checkpoints(checkpoint_path, urls=urls)

    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def get_transforms() -> transforms.Compose:
    """Get preprocessing transforms for inference.

    Returns
    -------
    transforms.Compose
        Composed transform pipeline including:
        - Intensity scaling to [0,1]
        - Fixed size cropping around AC-PC points
    """
    tr = [
        transforms.ScaleIntensityd(keys=['image'], minv=0, maxv=1),
        CropAroundACPCFixedSize(keys=['image'], fixed_size=PATCH_SIZE, random_translate=0),
    ]
    return transforms.Compose(tr)


def preprocess_volume(
    image_volume: np.ndarray,
    center_pt: Vector3d,
    transform: transforms.Transform | None = None
) -> dict[str, torch.Tensor | tuple[int, ...]]:
    """Preprocess a volume for inference.

    Parameters
    ----------
    image_volume : np.ndarray
        Input image slab in LIA orientation of shape (L, I, A).
    center_pt : np.ndarray
        Center point coordinates for cropping on the slice with shape (3,).
    transform : transforms.Transform or None, optional
        Custom transform pipeline, by default None.
        If None, uses default transforms from get_transforms().

    Returns
    -------
    dict[str, torch.Tensor | tuple[int, ...]]
        Dictionary containing preprocessed image tensor.
    """
    if transform is None:
        transform = get_transforms()

    # During training we used AC/PC coordinates, but during inference we approximate this by the center of the third
    # ventricle. Therefore we put in the third ventricle center as dummy AC/PC coordinates for cropping the image.
    sample = {"image": image_volume[None], "AC_center": center_pt[1:][None], "PC_center": center_pt[1:][None]}

    # Apply transforms
    transformed = transform(sample)
    
    # Add batch dimension if needed
    if torch.is_tensor(transformed["image"]):
        if transformed["image"].ndim == 3:
            transformed["image"] = transformed["image"].unsqueeze(0)
            
    return transformed

def predict(
        model: torch.nn.Module,
        image_volume: Image3d,
        patch_center: np.ndarray,
        device: torch.device | None = None,
        transform: transforms.Transform | None = None
    ) -> tuple[Points2dType, Points2dType]:
    """
    Run inference on an image volume
    
    Parameters
    ----------
    model : DenseNet
        Trained model for inference.
    image_volume : np.ndarray
        Input volume as numpy array.
    patch_center : np.ndarray
        Initial center point estimate for cropping.
    device : torch.device, optional
        Device to run inference on, by default None.
    transform : transforms.Transform, optional
        Custom transform pipeline, defaults to preconfigured transforms of `get_transforms`.

    Returns
    -------
    pc_coord : np.ndarray
        Predicted PC coordinates.
    ac_coord : np.ndarray
        Predicted AC coordinates.
    """
    if device is None:
        device = next(model.parameters()).device

    # prepend zero to third_ventricle_center
    patch_center_3d = np.concatenate([np.zeros(1), patch_center])
    
    # Preprocess
    t_dict = preprocess_volume(image_volume, patch_center_3d, transform)

    transformed_original = cast(torch.Tensor, t_dict["image"])
    inputs = transformed_original.to(device)

    inputs = inputs.transpose(0, 1)
    inputs = inputs.unfold(0, 3, 1).transpose(1, -1)[..., 0]

    # Run inference
    with torch.no_grad():
        outputs = model(inputs) * torch.as_tensor([PATCH_SIZE + PATCH_SIZE], device=device)

    # "crop_left"/"crop_top" are the crop offsets along the inferior/superior (i) and anterior/posterior (a) axes
    crop_start_i = cast(tuple[int, int], t_dict["crop_left"])
    crop_start_a = cast(tuple[int, int], t_dict["crop_top"])
    t_crops = [(crop_start_i + crop_start_a) * 2]
    outs: np.ndarray[tuple[int, Literal[4]], np.dtype[np.float64]]
    outs = outputs.cpu().numpy() + np.asarray(t_crops, dtype=float)
    return outs[:, :2], outs[:, 2:]


def run_inference_on_slice(
        model: DenseNet,
        image_slab: Image3d,
        center_pt: Vector2d,
        num_iterations: int = 2,
) -> tuple[Vector2d, Vector2d]:
    """Run inference on a single slice to detect AC and PC points.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model for AC-PC detection.
    image_slab : np.ndarray
        3D image mid-slices to run inference on in LIA orientation of shape (L, I, A).
    center_pt : np.ndarray
        Initial center point estimate for cropping.
    num_iterations : int, default=2
        Number of refinement iterations to run.

    Returns
    -------
    ac_coords : np.ndarray
        Detected AC voxel coordinates with shape (2,) containing its [I, A] (LIA axes 1 and 2) positions.
    pc_coords : np.ndarray
        Detected PC voxel coordinates with shape (2,) containing its [I, A] (LIA axes 1 and 2) positions.
    """

    if num_iterations < 1:
        raise ValueError("localization inference with less than 1 iteration is invalid!")

    pc_coords, ac_coords = center_pt[None], center_pt[None]
    # Run inference
    for _ in range(num_iterations):
        pc_coords, ac_coords = predict(model, image_slab, center_pt)
        center_pt = np.mean(np.stack([ac_coords, pc_coords], axis=0), axis=(0, 1))
    # average ac and pc coords across sagittal slices
    _pc_coords = np.mean(pc_coords, axis=0)
    _ac_coords = np.mean(ac_coords, axis=0)

    return _ac_coords, _pc_coords
