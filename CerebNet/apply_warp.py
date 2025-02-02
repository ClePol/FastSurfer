
# Copyright 2022 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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

# IMPORTS
from os.path import join
import numpy as np
import nibabel as nib

from CerebNet.datasets import utils


def save_nii_image(img_data, save_path, header, affine):
    img_out = nib.Nifti1Image(img_data, header=header, affine=affine)
    print(f"Saving {save_path}")
    nib.save(img_out, save_path)


def store_warped_data(img_path, lbl_path, warp_path, result_path, patch_size):

    img, img_file = utils.load_reorient_rescale_image(img_path)

    lbl_file = nib.load(lbl_path)
    label = np.asarray(lbl_file.get_fdata(), dtype=np.int16)

    warp_field = np.asarray(nib.load(warp_path).get_fdata())
    img = utils.map_size(img, base_shape=warp_field.shape[:3])
    label = utils.map_size(label, base_shape=warp_field.shape[:3])
    warped_img = utils.apply_warp_field(warp_field, img, interpol_order=3)
    warped_lbl = utils.apply_warp_field(warp_field, label, interpol_order=0)
    utils.map_subseg2label(warped_lbl, label_type='cereb_subseg')
    roi = utils.bounding_volume(label, patch_size)

    img = utils.map_size(warped_img[roi], patch_size)
    label = utils.map_size(warped_lbl[roi], patch_size)

    img_file.header['dim'][1:4] = patch_size
    img_file.set_data_dtype(img.dtype)
    lbl_file.header['dim'][1:4] = patch_size
    save_nii_image(img,
                   join(result_path, "T1_warped_cropped.nii.gz"),
                   header=img_file.header,
                   affine=img_file.affine)
    save_nii_image(label,
                   join(result_path, "label_warped_cropped.nii.gz"),
                   header=lbl_file.header,
                   affine=lbl_file.affine)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path",
                        help="path to T1 image",
                        type=str)
    parser.add_argument("--lbl_path",
                        help="path to label image",
                        type=str)
    parser.add_argument("--result_path",
                        help="folder to store the results",
                        type=str)

    parser.add_argument("--warp_filename",
                        help="Warp field file",
                        default='1Warp.nii.gz',
                        type=str)

    args = parser.parse_args()
    warp_path = join(args.result_path, args.warp_filename)
    store_warped_data(args.img_path,
                      args.lbl_path,
                      warp_path=warp_path,
                      result_path=args.result_path,
                      patch_size=(128, 128, 128))
