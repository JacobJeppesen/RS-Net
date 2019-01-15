#!/usr/bin/env python
# Needed to set seed for random generators for making reproducible experiments
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

import datetime
import os
import os.path
import random
import threading
import numpy as np
from keras import backend as K
from keras.backend import binary_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.utils import Sequence

from src.utils import extract_collapsed_cls, extract_cls_mask, image_normalizer, get_cls


def swish(x):
    return (K.sigmoid(x) * x)


def jaccard_coef(y_true, y_pred):
    """
    Calculates the Jaccard index
    """
    # From https://github.com/ternaus/kaggle_dstl_submission/blob/master/src/unet_crops.py
    smooth = 1e-12
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])  # Sum the product in all axes

    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])  # Sum the sum in all axes

    jac = (intersection + smooth) / (sum_ - intersection + smooth)  # Calc jaccard

    return K.mean(jac)


def jaccard_coef_thresholded(y_true, y_pred):
    """
    Calculates the binarized Jaccard index
    """
    # From https://github.com/ternaus/kaggle_dstl_submission/blob/master/src/unet_crops.py
    smooth = 1e-12

    # Round to 0 or 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    # Calculate Jaccard index
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_loss(y_true, y_pred):
    """
    Calculates the loss as a function of the Jaccard index and binary crossentropy
    """
    # From https://github.com/ternaus/kaggle_dstl_submission/blob/master/src/unet_crops.py
    return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)


def get_callbacks(params):
    # Must use save_weights_only=True in model_checkpoint (BUG: https://github.com/fchollet/keras/issues/8123)
    model_checkpoint = ModelCheckpoint(params.project_path + 'models/Unet/unet_tmp.hdf5',
                                       monitor='val_acc',
                                       save_weights_only=True,
                                       save_best_only=params.save_best_only)

    tensorboard = TensorBoard(log_dir=params.project_path + "reports/Unet/tensorboard/{}".
                              format(params.modelID),
                              write_graph=True,
                              write_images=True)

    csv_logger = CSVLogger(params.project_path + 'reports/Unet/csvlogger/' + params.modelID + '.log')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=2,
                                  patience=16, min_lr=1e-10)

    early_stopping = EarlyStopping(monitor='val_acc', patience=100, verbose=2)

    return csv_logger, model_checkpoint, reduce_lr, tensorboard, early_stopping


class ImageSequence(Sequence):
    def __init__(self, params, shuffle, seed, augment_data, validation_generator=False):
        # Load the names of the numpy files, each containing one patch
        if validation_generator:
            self.path = params.project_path + "data/processed/val/"
        else:
            self.path = params.project_path + "data/processed/train/"
        self.x_files = sorted(os.listdir(self.path + "img/"))  # os.listdir loads in arbitrary order, hence use sorted()
        self.x_files = [f for f in self.x_files if '.npy' in f]  # Only use .npy files (e.g. avoid .gitkeep)
        self.y_files = sorted(os.listdir(self.path + "mask/"))  # os.listdir loads in arbitrary order, hence use sorted()
        self.y_files = [f for f in self.y_files if '.npy' in f]  # Only use .npy files (e.g. avoid .gitkeep)

        # Only train on the products as specified in the data split if using k-fold CV
        if params.split_dataset:
            for product in params.test_tiles[1]:
                self.x_files = [f for f in self.x_files if product[:-9] not in f]
                self.y_files = [f for f in self.y_files if product[:-9] not in f]

        # Create random generator used for shuffling files
        self.random = random.Random()

        # Shuffle the patches
        if shuffle:
            self.random.seed(seed)
            self.random.shuffle(self.x_files)
            self.random.seed(seed)
            self.random.shuffle(self.y_files)

        self.batch_size = params.batch_size

        # Create placeholders
        self.x_all_bands = np.zeros((params.batch_size, params.patch_size, params.patch_size, 10), dtype=np.float32)
        self.x = np.zeros((params.batch_size, params.patch_size, params.patch_size, np.size(params.bands)), dtype=np.float32)

        self.clip_pixels = np.int32(params.overlap / 2)
        self.y = np.zeros((params.batch_size, params.patch_size - 2*self.clip_pixels, params.patch_size - 2*self.clip_pixels, 1), dtype=np.float32)

        # Load the params object for the normalizer function (not nice!)
        self.params = params

        # Convert class names to the actual integers in the masks (convert e.g. 'cloud' to 255 for Landsat8)
        self.cls = get_cls(self.params.satellite, self.params.train_dataset, self.params.cls)

        # Augment the data
        self.augment_data = augment_data

    def __len__(self):
        return int(np.ceil(len(self.x_files) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x_filenames = self.x_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_filenames = self.y_files[idx * self.batch_size:(idx + 1) * self.batch_size]

        for i, filename in enumerate(batch_x_filenames):
            # Load all bands
            self.x_all_bands[i, :, :, :] = np.load(self.path + "img/" + filename)

            # Extract the wanted bands
            for j, b in enumerate(self.params.bands):
                if b == 1:
                    self.x[i, :, :, j] = self.x_all_bands[i, :, :, 0]
                elif b == 2:
                    self.x[i, :, :, j] = self.x_all_bands[i, :, :, 1]
                elif b == 3:
                    self.x[i, :, :, j] = self.x_all_bands[i, :, :, 2]
                elif b == 4:
                    self.x[i, :, :, j] = self.x_all_bands[i, :, :, 3]
                elif b == 5:
                    self.x[i, :, :, j] = self.x_all_bands[i, :, :, 4]
                elif b == 6:
                    self.x[i, :, :, j] = self.x_all_bands[i, :, :, 5]
                elif b == 7:
                    self.x[i, :, :, j] = self.x_all_bands[i, :, :, 6]
                elif b == 8:
                    raise ValueError('Band 8 (pan-chromatic band) cannot be included')
                elif b == 9:
                    self.x[i, :, :, j] = self.x_all_bands[i, :, :, 7]
                elif b == 10:
                    self.x[i, :, :, j] = self.x_all_bands[i, :, :, 8]
                elif b == 11:
                    self.x[i, :, :, j] = self.x_all_bands[i, :, :, 9]

            # Normalize
            self.x[i, :, :, :] = image_normalizer(self.x[i, :, :, :], self.params, self.params.norm_method)

        for i, filename in enumerate(batch_y_filenames):
            # Load the masks
            mask = np.load(self.path + "mask/" + filename)

            # Create the binary masks
            if self.params.collapse_cls:
                mask = extract_collapsed_cls(mask, self.cls)

                # Save the binary mask (cropped)
                self.y[i, :, :, :] = mask[self.clip_pixels:self.params.patch_size - self.clip_pixels,
                                          self.clip_pixels:self.params.patch_size - self.clip_pixels,
                                          :]

        if self.augment_data:
            if self.random.randint(0, 1):
                np.flip(self.x, axis=1)
                np.flip(self.y, axis=1)

            if self.random.randint(0, 1):
                np.flip(self.x, axis=2)
                np.flip(self.y, axis=2)

        return self.x, self.y
