#!/usr/bin/env python3
from typing import Dict

# Copyright 2021 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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

import numpy.typing as npt

# Collection of functions related to FreeSurfer's LTA (linear transform array) files:


def writeLTA(
        filename: str,
        T: npt.ArrayLike,
        src_fname: str,
        src_header: Dict,
        dst_fname: str,
        dst_header: Dict
) -> None:
    """
    Write linear transform array info to an .lta file.

    Parameters
    ----------
    filename : str
        File to write on.
    T : npt.ArrayLike
        Linear transform array to be saved.
    src_fname : str
        Source filename.
    src_header : Dict
        Source header.
    dst_fname : str
        Destination filename.
    dst_header : Dict
        Destination header.

    Raises
    ------
    ValueError
        Header format missing field (Source or Destination).
    """
    from datetime import datetime
    import getpass

    fields = ("dims", "delta", "Mdc", "Pxyz_c")
    for field in fields:
        if field not in src_header:
            raise ValueError(
                "writeLTA Error: src_header format missing field: {}".format(field)
            )
        if field not in dst_header:
            raise ValueError(
                "writeLTA Error: dst_header format missing field: {}".format(field)
            )

    src_dims = str(src_header["dims"][0:3]).replace("[", "").replace("]", "")
    src_vsize = str(src_header["delta"][0:3]).replace("[", "").replace("]", "")
    src_v2r = src_header["Mdc"]
    src_c = src_header["Pxyz_c"]

    dst_dims = str(dst_header["dims"][0:3]).replace("[", "").replace("]", "")
    dst_vsize = str(dst_header["delta"][0:3]).replace("[", "").replace("]", "")
    dst_v2r = dst_header["Mdc"]
    dst_c = dst_header["Pxyz_c"]

    f = open(filename, "w")
    f.write("# transform file {}\n".format(filename))
    f.write(
        "# created by {} on {}\n\n".format(getpass.getuser(), datetime.now().ctime())
    )
    f.write("type      = 1 # LINEAR_RAS_TO_RAS\n")
    f.write("nxforms   = 1\n")
    f.write("mean      = 0.0 0.0 0.0\n")
    f.write("sigma     = 1.0\n")
    f.write("1 4 4\n")
    f.write(str(T).replace(" [", "").replace("[", "").replace("]", ""))
    f.write("\n")
    f.write("src volume info\n")
    f.write("valid = 1  # volume info valid\n")
    f.write("filename = {}\n".format(src_fname))
    f.write("volume = {}\n".format(src_dims))
    f.write("voxelsize = {}\n".format(src_vsize))
    f.write("xras   = {}\n".format(src_v2r[0, :]).replace("[", "").replace("]", ""))
    f.write("yras   = {}\n".format(src_v2r[1, :]).replace("[", "").replace("]", ""))
    f.write("zras   = {}\n".format(src_v2r[2, :]).replace("[", "").replace("]", ""))
    f.write("cras   = {}\n".format(src_c).replace("[", "").replace("]", ""))
    f.write("dst volume info\n")
    f.write("valid = 1  # volume info valid\n")
    f.write("filename = {}\n".format(dst_fname))
    f.write("volume = {}\n".format(dst_dims))
    f.write("voxelsize = {}\n".format(dst_vsize))
    f.write("xras   = {}\n".format(dst_v2r[0, :]).replace("[", "").replace("]", ""))
    f.write("yras   = {}\n".format(dst_v2r[1, :]).replace("[", "").replace("]", ""))
    f.write("zras   = {}\n".format(dst_v2r[2, :]).replace("[", "").replace("]", ""))
    f.write("cras   = {}\n".format(dst_c).replace("[", "").replace("]", ""))
    f.close()
