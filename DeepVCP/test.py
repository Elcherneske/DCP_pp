import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
import nibabel.orientations as nio

img_nib = nib.load("data.nii")
# array info
img_data = img_nib.get_fdata()
print('img shape: ', img_nib.shape)
print('data shape: ', img_data.shape)
print('data type: ', type(img_data))
# volume info
zooms = img_nib.header.get_zooms()  # similar to the spacing settings in the 3D compounding
print('zooms of the voxel: ', zooms)
axs_code = nio.ornt2axcodes(nio.io_orientation(img_nib.affine))
print('img orientation code: {}'.format(axs_code))
# global info
print('affine matrix: ', img_nib.affine)


def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice, cmap='gray')

img_shape = img_data.shape

slice_0 = img_data[img_shape[0] // 2, :, :]
slice_1 = img_data[:, img_shape[1] // 2, :]
slice_2 = img_data[:, :, img_shape[2] // 2]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for body image")
plt.show()
