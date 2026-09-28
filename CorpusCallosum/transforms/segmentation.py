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
import logging
from typing import Literal

import numpy as np
from monai.transforms import MapTransform, RandomizableTransform


class CropAroundACPC(RandomizableTransform, MapTransform):
    """Crop image around anterior and posterior commissure points.

    A transform that crops the input image around the AC and PC points with
    optional padding and random translation.

    Parameters
    ----------
    keys : list[str]
        Keys of the data dictionary to apply the transform to.
    allow_missing_keys : bool, default=False
        Whether to allow missing keys in the data dictionary.
    padding_mm : float, default=10.0
        Padding around AC-PC region in millimeters.
    random_translate : float, default=0
        Maximum random translation in voxels, off by default.

    Notes
    -----
    The transform expects the following keys in the data dictionary:

    - AC_center : np.ndarray
        Coordinates of anterior commissure
    - PC_center : np.ndarray
        Coordinates of posterior commissure
    - res : float
        Voxel resolution in mm

    """
    
    def __init__(self, keys: list[str], allow_missing_keys: bool = False, 
                 padding_mm: float = 10, random_translate: float = 0) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=1, do_transform=True)
        self.padding_mm = padding_mm
        self.random_translate = random_translate

    def __call__(self, data: dict) -> dict:
        """Apply the transform to the data.

        Parameters
        ----------
        data : dict
            Dictionary containing the data to transform.

        Returns
        -------
        dict
            Transformed data dictionary.
        """
        d = dict(data)

        if self.random_translate > 0:
            random_translate = np.random.randint(-self.random_translate, self.random_translate, size=2)
        else:
            random_translate = (0,0,0)

        pc_center = d["PC_center"]
        ac_center = d["AC_center"]

        ac_pc = np.stack([ac_center, pc_center], axis=0)

        ac_pc_bottomleft = np.min(ac_pc, axis=0).astype(int)
        ac_pc_topright = np.max(ac_pc, axis=0).astype(int)

        voxel_padding: np.ndarray[tuple[Literal[2]], np.dtype[np.int_]] = np.round(
            self.padding_mm / d["res"]).astype(int)

        crop_left = ac_pc_bottomleft[1] - int(voxel_padding[0] * 1.5) + random_translate[0]
        crop_right = ac_pc_topright[1] + voxel_padding[0] // 2 + random_translate[0]
        crop_top = ac_pc_bottomleft[2] - voxel_padding[1] + random_translate[1]
        crop_bottom = ac_pc_topright[2] + voxel_padding[1] + random_translate[1]

        keys_to_process = [key for key in self.keys if key in d.keys()]

        if not self.allow_missing_keys and set(keys_to_process) != set(self.keys):
            raise ValueError("Some keys are missing in the data dictionary.")

        if len(keys_to_process) == 0:
            logging.getLogger(__name__).warning("No keys to process.")
            return d

        first_key = keys_to_process[0]
        d["to_pad"] = crop_left, d[first_key].shape[2] - crop_right, crop_top, d[first_key].shape[3] - crop_bottom

        for key in keys_to_process:
            d[key] = d[key][:, :, crop_left:crop_right, crop_top:crop_bottom]

        return d
