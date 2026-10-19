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

from typing import TypedDict

import numpy as np
from nibabel.freesurfer.mghformat import MGHHeader

from FastSurferCNN.utils import AffineMatrix3x3, Vector3d, nibabelHeader


class MGHHeaderDict(TypedDict):
    """A dictionary with the four required fields of a MGH Header."""

    dims: Vector3d
    delta: Vector3d
    Mdc: AffineMatrix3x3
    Pxyz_c: Vector3d


def convert_numpy_to_json_serializable(obj: object) -> object:
    """Convert numpy types to JSON serializable types.

    Parameters
    ----------
    obj : dict, list, array, number, serializable
        Object to convert to JSON serializable type.

    Returns
    -------
    object
        JSON serializable version of the input object.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_to_json_serializable(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return obj


def init_mgh_header(header: nibabelHeader, header_dict: MGHHeaderDict) -> MGHHeader:
    """
    Generates a MGHHeader object from a header and a header dictionary.

    Parameters
    ----------
    header : nibabelHeader
        The header object used to initialize the generated header.
    header_dict : MGHHeaderDict
        A dictionary of values to overwrite in the generated header.

    Returns
    -------
    MGHHeader
        The header updated with values in header_dict.
    """
    new_header: MGHHeader = MGHHeader.from_header(header)
    if "dims" in header_dict:
        new_header["dims"] = np.append(header_dict["dims"], [1])
    for key in ("delta", "Pxyz_c", "Mdc"):
        if key in header_dict:
            new_header[key] = header_dict[key]
    return new_header
