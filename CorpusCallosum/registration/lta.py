#!/usr/bin/env python3

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

import re
import numpy as np
import numpy.typing as npt

# Collection of functions related to FreeSurfer's LTA (linear transform array) files:



def get_affine(xform_xyz_ras: np.ndarray, xform_cras: np.ndarray, resolution: np.ndarray, volume_shape: np.ndarray) -> np.ndarray:
    ''' Get the affine transform from the header/xform information.
    Modeled after https://neurostars.org/t/freesurfer-cras-offset/5587 https://github.com/nipy/nibabel/blob/d1518aa71a8a80f5e7049a3509dfb49cf6b78005/nibabel/freesurfer/mghformat.py#L175-L185

    MGH format doesn't store the transform directly. Instead it's gleaned
    from the zooms/resolution ( delta ), direction cosines ( Mdc / xyz_ras ), RAS centers (
    Pxyz_c / c_ras ) and the dimensions.

    This format is the same format used in mri_info "x_form" output and in .lta files

    paramters:

    xform_xyz_ras [3,3]:  xform arr rows = x_ras , y_ras, z_ras, c_ras   rotation matrix in mm resolution
    xform_cras    [3,1]:  offset to RAS center in mm resolution
    resolution    [3,1]:  mm per voxel in all three directions
    volume_shape  [3,1]:  volume voxel count in all three directions

    return        [3,3]:  affine vox2ras matrix
    '''

    rotation_mat = xform_xyz_ras.T * resolution
    vol_center = rotation_mat @ np.array(volume_shape) / 2
    out_affine = np.identity(4)

    out_affine[:3,:3] = rotation_mat
    out_affine[:3,3] = xform_cras - vol_center
    return out_affine


def readLTA(filename: str, RAS_RAS: bool = True, get_affines: bool = False, get_shape: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    read lta files

    filename    path to .lta file
    RAS_RAS     enforce that the file is a ras2ras transfrom
    get_affines calculate the affine matrices of source and target of the transform from the xform (adds two return arguments)

    return      arr, src_affine, trg_affine
    '''
    
    arr = np.zeros((4,4))

    if get_affines:
        src_vol_shape = np.zeros((3,1))
        src_resolution = np.zeros((3,1))
        src_xyz_ras = np.zeros((3,3))
        src_cras = np.zeros((3,1))

        trg_vol_shape = np.zeros((3,1))
        trg_resolution = np.zeros((3,1))
        trg_xyz_ras = np.zeros((3,3))
        trg_cras = np.zeros((3,1))

        SRC_AFFINE_READ = False

    
    read_lines = 0

    read_matrix_line = lambda x: [float(x) for x in x.split('=')[1].split('#')[0].strip().split(' ')]
    
    with open(filename, 'r') as f:
        #while (line := f.readline().rstrip()): # read lines on demand

        
        for i,line in enumerate(f.readlines()):
            line = line.rstrip()
            line = re.sub(' +', ' ', line)
            line = line.lstrip()

            if RAS_RAS:
                if line.startswith('type'):
                    t_type = int(line.split('=')[1].split('#')[0])
                    assert(t_type==1)
            
            ras2ras_line = line.split(' ')
            if len(ras2ras_line) == 4 and not ras2ras_line[0].startswith('#'):
                #print(ras2ras_line)
                arr[read_lines,:] = [float(x) for x in ras2ras_line]
                read_lines += 1
            if read_lines == 4 and not get_affines:
                return arr

            if get_affines:
                if not SRC_AFFINE_READ:
                    if line.startswith('volume'): #i == 12:
                        src_vol_shape = read_matrix_line(line)
                    elif line.startswith('voxelsize'):#i == 13:
                        src_resolution = read_matrix_line(line)
                    elif line.startswith('xras'):#i == 14:
                        src_xyz_ras[0] = read_matrix_line(line)
                    elif line.startswith('yras'):#i == 15:
                        src_xyz_ras[1] = read_matrix_line(line)
                    elif line.startswith('zras'):#i == 16:
                        src_xyz_ras[2] = read_matrix_line(line)
                    elif line.startswith('cras'):#i == 17:
                        src_cras = read_matrix_line(line)
                        SRC_AFFINE_READ = True
                else:
                    if line.startswith('volume'):#i == 21:
                        trg_vol_shape = read_matrix_line(line)
                    elif line.startswith('voxelsize'):#i == 22:
                        trg_resolution = read_matrix_line(line)
                    elif line.startswith('xras'):#i == 23:
                        trg_xyz_ras[0] = read_matrix_line(line)
                    elif line.startswith('yras'):#i == 24:
                        trg_xyz_ras[1] = read_matrix_line(line)
                    elif line.startswith('zras'):#i == 25:
                        trg_xyz_ras[2] = read_matrix_line(line)
                    elif line.startswith('cras'):#i == 26:
                        trg_cras = read_matrix_line(line)
                        src_affine = get_affine(src_xyz_ras, src_cras, src_resolution, src_vol_shape)
                        trg_affine = get_affine(trg_xyz_ras, trg_cras, trg_resolution, trg_vol_shape)

                        if get_shape:
                            return arr, src_affine, trg_affine, np.array(src_vol_shape).astype(int), np.array(trg_vol_shape).astype(int)
                        else:
                            return arr, src_affine, trg_affine



def writeLTA(
        filename: str,
        T: npt.ArrayLike,
        src_fname: str,
        src_header: dict,
        dst_fname: str,
        dst_header: dict
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
    import getpass
    from datetime import datetime

    fields = ("dims", "delta", "Mdc", "Pxyz_c")
    for field in fields:
        if field not in src_header:
            raise ValueError(
                f"writeLTA Error: src_header format missing field: {field}"
            )
        if field not in dst_header:
            raise ValueError(
                f"writeLTA Error: dst_header format missing field: {field}"
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
    f.write(f"# transform file {filename}\n")
    f.write(
        f"# created by {getpass.getuser()} on {datetime.now().ctime()}\n\n"
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
    f.write(f"filename = {src_fname}\n")
    f.write(f"volume = {src_dims}\n")
    f.write(f"voxelsize = {src_vsize}\n")
    f.write(f"xras   = {src_v2r[0, :]}\n".replace("[", "").replace("]", ""))
    f.write(f"yras   = {src_v2r[1, :]}\n".replace("[", "").replace("]", ""))
    f.write(f"zras   = {src_v2r[2, :]}\n".replace("[", "").replace("]", ""))
    f.write(f"cras   = {src_c}\n".replace("[", "").replace("]", ""))
    f.write("dst volume info\n")
    f.write("valid = 1  # volume info valid\n")
    f.write(f"filename = {dst_fname}\n")
    f.write(f"volume = {dst_dims}\n")
    f.write(f"voxelsize = {dst_vsize}\n")
    f.write(f"xras   = {dst_v2r[0, :]}\n".replace("[", "").replace("]", ""))
    f.write(f"yras   = {dst_v2r[1, :]}\n".replace("[", "").replace("]", ""))
    f.write(f"zras   = {dst_v2r[2, :]}\n".replace("[", "").replace("]", ""))
    f.write(f"cras   = {dst_c}\n".replace("[", "").replace("]", ""))
    f.close()
