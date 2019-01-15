#!/usr/bin/env python
#  Needed to set seed for random generators for making reproducible experiments
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

import numpy as np
import tifffile as tiff
import os
import random
import shutil
from PIL import Image
from ..utils import patch_image


def make_numpy_dataset(params):
    """
    Process a numpy dataset from the raw data, and do training/val data split
    """
    if params.satellite == 'Sentinel-2':
        __make_sentinel2_dataset__(params)
        print('Processing validation data set')
        __make_sentinel2_val_dataset__(params)
    elif params.satellite == 'Landsat8':
        if 'Biome' in params.train_dataset:
            __make_landsat8_biome_dataset__(params)
            print('Processing validation data set')
            __make_landsat8_val_dataset__(params)
        elif 'SPARCS' in params.train_dataset:
            __make_landsat8_sparcs_dataset__(params)
            print('Processing validation data set')
            __make_landsat8_val_dataset__(params)


def __make_sentinel2_dataset__(params):
    """
    Loads the training data into numpy arrays
    """
    # The training data always contain all 13 bands and all 12 classes from the sen2cor classification
    n_cls_sen2cor = 12
    n_cls_fmask = 4
    n_bands = 13

    imgsize = params.tile_size
    patch_size = params.patch_size
    data_path = params.project_path + 'data/processed/'

    # Initialize x
    x = np.zeros((imgsize, imgsize, n_bands), dtype=np.uint16)

    # Pixelwise labels for 1 image of size s, with n classes
    y_sen2cor = np.zeros((imgsize, imgsize, 1), dtype=np.uint8)  # Not in one-hot (classes are integers from 0-11)
    y_fmask = np.zeros((imgsize, imgsize, 1), dtype=np.uint8)  # Not in one-hot (classes are integers from ?-?)

    # Find all files in the raw dataset and filter for .tif files
    files = sorted(os.listdir("data/raw"))
    files = [f for f in files if '.tif' in f]

    # Iterate through the files and create the training data
    tile_old = []
    date_old = []
    for f in files:
        # Check if you "hit" a new tile or date
        if f[4:10] != tile_old or f[11:19] != date_old:
            print('Processing tile: ' + f[0:-13])

            # Load bands for the specific tile and date
            x[:, :, 0] = tiff.imread('data/raw/' + f[0:-12] + 'B01_60m.tiff')
            x[:, :, 1] = tiff.imread('data/raw/' + f[0:-12] + 'B02_10m.tiff')
            x[:, :, 2] = tiff.imread('data/raw/' + f[0:-12] + 'B03_10m.tiff')
            x[:, :, 3] = tiff.imread('data/raw/' + f[0:-12] + 'B04_10m.tiff')
            x[:, :, 4] = tiff.imread('data/raw/' + f[0:-12] + 'B05_20m.tiff')
            x[:, :, 5] = tiff.imread('data/raw/' + f[0:-12] + 'B06_20m.tiff')
            x[:, :, 6] = tiff.imread('data/raw/' + f[0:-12] + 'B07_20m.tiff')
            x[:, :, 7] = tiff.imread('data/raw/' + f[0:-12] + 'B08_10m.tiff')
            x[:, :, 8] = tiff.imread('data/raw/' + f[0:-12] + 'B09_60m.tiff')
            x[:, :, 9] = tiff.imread('data/raw/' + f[0:-12] + 'B10_60m.tiff')
            x[:, :, 10] = tiff.imread('data/raw/' + f[0:-12] + 'B11_20m.tiff')
            x[:, :, 11] = tiff.imread('data/raw/' + f[0:-12] + 'B12_20m.tiff')
            x[:, :, 12] = tiff.imread('data/raw/' + f[0:-12] + 'B8A_20m.tiff')

            # For sen2cor classification mask
            y_sen2cor[:, :, 0] = tiff.imread('data/raw/' + f[0:-12] + 'SCL_20m.tiff')  # The 20 m is the native resolution

            # For fmask classification mask
            im = Image.open('data/raw/' + f[0:-12] + 'Fma_20m.tiff')  # The 20 m is the native resolution
            y_fmask[:, :, 0] = np.array(im)

            # Patch the image and the mask
            x_patched, _, _ = patch_image(x, patch_size, overlap=0)
            y_sen2cor_patched, _, _ = patch_image(y_sen2cor, patch_size, overlap=0)
            y_fmask_patched, _, _ = patch_image(y_fmask, patch_size, overlap=0)

            # Save all the patches individually
            for patch in range(np.size(x_patched, axis=0)):
                if np.mean(x_patched[patch, :, :, :]) != 0:  # Ignore blank patches
                    np.save(data_path + 'train/img/' + f[0:-13] + '_x_patch-%d'
                            % patch, x_patched[patch, :, :, :])

                    np.save(data_path + 'train/mask/' + f[0:-13] + '_y_sen2cor_%d-cls_patch-%d'
                            % (n_cls_sen2cor, patch), y_sen2cor_patched[patch, :, :, :])

                    np.save(data_path + 'train/mask/' + f[0:-13] + '_y_fmask_%d-cls_patch-%d'
                            % (n_cls_fmask, patch), y_fmask_patched[patch, :, :, :])

            tile_old = f[4:10]
            date_old = f[11:19]


def __make_landsat8_biome_dataset__(params):
    """
    Loads the training data into numpy arrays
    """
    # Start by deleting the old processed dataset and create new folders
    shutil.rmtree(params.project_path + 'data/processed/train', ignore_errors=True)
    shutil.rmtree(params.project_path + 'data/processed/val', ignore_errors=True)
    os.makedirs(params.project_path + 'data/processed/train/img')
    os.makedirs(params.project_path + 'data/processed/train/mask')
    os.makedirs(params.project_path + 'data/processed/val/img')
    os.makedirs(params.project_path + 'data/processed/val/mask')

    # Create the new dataset
    n_bands = 10  # Omitting the panchromatic band for now

    patch_size = params.patch_size
    data_path = params.project_path + 'data/processed/'

    folders = sorted(os.listdir("./data/raw/Biome_dataset/"))
    folders = [f for f in folders if '.' not in f]  # Filter out .gitignore

    for folder in folders:
        products = sorted(os.listdir("./data/raw/Biome_dataset/" + folder + "/BC/"))

        for product in products:
            print('Processing product: ' + folder + ' - ' + product)

            product_path = "./data/raw/Biome_dataset/" + folder + "/BC/" + product + "/"
            toa_path = "./data/processed/Biome_TOA/" + folder + "/BC/" + product + "/"
            fmask_path = "./data/output/Biome/"

            # Initialize x and mask
            temp = tiff.imread(product_path + product + "_B1.TIF")
            x = np.zeros((np.shape(temp)[0], np.shape(temp)[1], n_bands), dtype=np.uint16)
            y = np.zeros((np.shape(temp)[0], np.shape(temp)[1], 1), dtype=np.uint8)

            # Load bands for the specific tile and date
            # x[:, :, 0] = temp
            # x[:, :, 1] = tiff.imread(product_path + product + "_B2.TIF")
            # x[:, :, 2] = tiff.imread(product_path + product + "_B3.TIF")
            # x[:, :, 3] = tiff.imread(product_path + product + "_B4.TIF")
            # x[:, :, 4] = tiff.imread(product_path + product + "_B5.TIF")
            # x[:, :, 5] = tiff.imread(product_path + product + "_B6.TIF")
            # x[:, :, 6] = tiff.imread(product_path + product + "_B7.TIF")
            # x[:, :, 7] = tiff.imread(product_path + product + "_B9.TIF")  # Omitting panchromatic band (8)
            x[:, :, 0:8] = tiff.imread(toa_path + product + "_toa.TIF")

            # Set all NaN values to 0
            x[x == 32767] = 0

            # Load thermal bands
            x[:, :, 8] = tiff.imread(product_path + product + "_B10.TIF")
            x[:, :, 9] = tiff.imread(product_path + product + "_B11.TIF")

            if params.train_dataset == 'Biome_gt':
                y[:, :, 0] = tiff.imread(product_path + product + "_fixedmask.TIF")
            elif params.train_dataset == 'Biome_fmask':
                y[:, :, 0] = Image.open(fmask_path + product + "_fmask.png")
            else:
                raise ValueError('Invalid dataset. Choose Biome_gt, Biome_fmask, SPARCS_gt, or SPARCS_fmask.')

            # Patch the image and the mask
            x_patched, _, _ = patch_image(x, patch_size, overlap=params.overlap_train_set)
            y_patched, _, _ = patch_image(y, patch_size, overlap=params.overlap_train_set)

            # Save all the patches individually
            for patch in range(np.size(x_patched, axis=0)):
                # if np.mean(x_patched[patch, :, :, :]) != 0:  # Ignore blank patches
                if np.all(x_patched[patch, :, :, :]) != 0:  # Ignore patches with any black pixels
                    category = folder[0:4].lower()
                    np.save(data_path + 'train/img/' + category + '_' + product + '_x_patch-%d'
                            % patch, x_patched[patch, :, :, :])

                    np.save(data_path + 'train/mask/' + category + '_' + product + '_y_patch-%d'
                            % patch, y_patched[patch, :, :, :])


def __make_landsat8_sparcs_dataset__(params):
    """
    Loads the training data into numpy arrays
    """
    # Start by deleting the old processed dataset and create new folders
    shutil.rmtree(params.project_path + 'data/processed/train', ignore_errors=True)
    shutil.rmtree(params.project_path + 'data/processed/val', ignore_errors=True)
    os.makedirs(params.project_path + 'data/processed/train/img')
    os.makedirs(params.project_path + 'data/processed/train/mask')
    os.makedirs(params.project_path + 'data/processed/val/img')
    os.makedirs(params.project_path + 'data/processed/val/mask')

    # Define paths
    raw_data_path = "./data/raw/SPARCS_dataset/"
    toa_data_path = "./data/processed/SPARCS_TOA/"
    fmask_path = "./data/output/SPARCS/"
    processed_data_path = params.project_path + 'data/processed/'

    # Load product names
    products = sorted(os.listdir(raw_data_path))
    products = [f for f in products if 'data.tif' in f]
    products = [f for f in products if 'aux' not in f]

    # Init variable for image
    x = np.zeros((1000, 1000, 10), dtype=np.uint16) # Use to uint16 to save space (org. data is in uint16)

    for product in products:
        print('Processing product: ' + product)

        # Load img and mask
        x[:, :, 0:8] = tiff.imread(toa_data_path + product[:-8] + 'toa.TIF')
        x[:, :, 8:10] = tiff.imread(raw_data_path + product)[:, :, 9:11]

        x.astype(np.uint16)
        y = np.zeros((np.shape(x)[0], np.shape(x)[1], 1), dtype=np.uint8)
        if params.train_dataset == 'SPARCS_gt':
            y[:, :, 0] = Image.open(raw_data_path + product[:-8] + "mask.png")
        elif params.train_dataset == 'SPARCS_fmask':
            y[:, :, 0] = Image.open(fmask_path + product[:-8] + "fmask.png")
        else:
            raise ValueError('Invalid dataset. Choose Biome_gt, Biome_fmask, SPARCS_gt, or SPARCS_fmask.')

        # Mirror-pad the image and mask such that it matches the required patches
        padding_size = int(params.patch_size/2)
        npad = ((padding_size, padding_size), (padding_size, padding_size), (0, 0))
        x_padded = np.pad(x, pad_width=npad, mode='symmetric')
        y_padded = np.pad(y, pad_width=npad, mode='symmetric')

        # Patch the image and the mask
        x_patched, _, _ = patch_image(x_padded, params.patch_size, overlap=params.overlap_train_set)
        y_patched, _, _ = patch_image(y_padded, params.patch_size, overlap=params.overlap_train_set)

        # Save all the patches individually
        for patch in range(np.size(x_patched, axis=0)):
            if np.all(x_patched[patch, :, :, :]) != 0:  # Ignore patches with any black pixels
                np.save(processed_data_path + 'train/img/' + product[:-9] + '_x_patch-%d'
                        % patch, x_patched[patch, :, :, :])

                np.save(processed_data_path + 'train/mask/' + product[:-9] + '_y_patch-%d'
                        % patch, y_patched[patch, :, :, :])


def __make_sentinel2_val_dataset__(params):
    """
    Creates validation data set of 20% of the training data set (uses random patches)
    """

    data_path = params.project_path + 'data/processed/'

    # Create sorted lists of all the training data
    trn_files = sorted(os.listdir(data_path + 'train/img/'))
    mask_files = os.listdir(data_path + 'train/mask/')  # List all mask files
    mask_sen2cor_files = [f for f in mask_files if 'sen2cor' in f]  # Filter out sen2cor masks
    mask_sen2cor_files = sorted(mask_sen2cor_files)
    mask_fmask_files = [f for f in mask_files if 'fmask' in f]  # Filter out fmask files
    mask_fmask_files = sorted(mask_fmask_files)

    # Shuffle the list (use the same seed such that images and masks match)
    seed = 1
    random.seed(seed)
    random.shuffle(trn_files)

    random.seed(seed)
    random.shuffle(mask_sen2cor_files)

    random.seed(seed)
    random.shuffle(mask_fmask_files)

    # Remove the last 80% of the files (ie. keep 20% for validation data set)
    trn_files = trn_files[0: int(len(trn_files) * 0.20)]
    mask_sen2cor_files = mask_sen2cor_files[0: int(len(mask_sen2cor_files) * 0.20)]
    mask_fmask_files = mask_fmask_files[0: int(len(mask_fmask_files) * 0.20)]

    # Move the files
    for f in trn_files:
        shutil.move(data_path + 'train/img/' + f, data_path + 'val/img/' + f)

    for f in mask_sen2cor_files:
        shutil.move(data_path + 'train/mask/' + f, data_path + 'val/mask/' + f)

    for f in mask_fmask_files:
        shutil.move(data_path + 'train/mask/' + f, data_path + 'val/mask/' + f)


def __make_landsat8_val_dataset__(params):
    """
    Creates validation data set of 10% of the training data set (uses random patches)
    """
    data_path = params.project_path + 'data/processed/'

    # Create sorted lists of all the training data
    trn_files = sorted(os.listdir(data_path + 'train/img/'))
    mask_files = sorted(os.listdir(data_path + 'train/mask/'))  # List all mask files

    # Shuffle the list (use the same seed such that images and masks match)
    seed = 1
    random.seed(seed)
    random.shuffle(trn_files)

    random.seed(seed)
    random.shuffle(mask_files)

    # Remove the last 90% of the files (ie. keep 10% for validation data set)
    trn_files = trn_files[0: int(len(trn_files) * 0.10)]
    mask_files = mask_files[0: int(len(mask_files) * 0.10)]

    # Move the files
    for f in trn_files:
        shutil.move(data_path + 'train/img/' + f, data_path + 'val/img/' + f)

    for f in mask_files:
        shutil.move(data_path + 'train/mask/' + f, data_path + 'val/mask/' + f)
