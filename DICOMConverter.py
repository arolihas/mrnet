import pydicom as dicom
import numpy as np
import os
import cv2
import PIL # optional


def get_dicom_nparray(folder_path, filename):
    ds = dicom.dcmread(os.path.join(folder_path, filename))
    return ds.pixel_array


"""
# make it True if you want in PNG format
PNG = False
# Specify the .dcm folder path
folder_path = "stage_1_test_images"
# Specify the output jpg/png folder path
jpg_folder_path = "JPG_test"
images_path = os.listdir(folder_path)

for n, image in enumerate(images_path):
    pixel_array_numpy = get_nparray(folder_path, image)

    if PNG == False:
        image = image.replace('.dcm', '.jpg')
    else:
        image = image.replace('.dcm', '.png')
    cv2.imwrite(os.path.join(jpg_folder_path, image), pixel_array_numpy)
    if n % 1 == 0:
        print('{} image converted'.format(n + 1))
"""
