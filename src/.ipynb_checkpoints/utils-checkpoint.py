#!/usr/bin/env python
import numpy as np
import os
import random
from keras import backend as K


def calc_jacc(model, n_cls, patch_size, overlap, threshold):
    img = np.load('data/processed/x_tmp_%d.npy' % n_cls)
    true_binary_mask = np.load('data/processed/y_tmp_%d.npy' % n_cls)

    # Divide the image into patches
    img, n_height, n_width = patch_image(img, 320, overlap=overlap)

    # Do the prediction
    predicted_mask = model.predict(img, batch_size=4)

    # Stitch the mask back together (important, as we would otherwise include the overlap in the evaluation)
    predicted_mask = stitch_image(predicted_mask, n_height, n_width, patch_size, overlap)

    # Threshold the prediction
    predicted_binary_mask = predicted_mask >= threshold

    # From https://www.kaggle.com/lopuhin/full-pipeline-demo-poly-pixels-ml-poly
    max_size = patch_size * n_height - overlap * n_height
    tp, fp, fn = ((predicted_binary_mask[:, :, 0] & true_binary_mask[0:max_size, 0:max_size, 0]).sum(),
                  (predicted_binary_mask[:, :, 0] & ~true_binary_mask[0:max_size, 0:max_size, 0]).sum(),
                  (~predicted_binary_mask[:, :, 0] & true_binary_mask[0:max_size, 0:max_size, 0]).sum())

    # See https://en.wikipedia.org/wiki/Jaccard_index#Similarity_of_asymmetric_binary_attributes
    pixel_jaccard = tp / (tp + fp + fn)

    return pixel_jaccard


def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    # Keras implementations of jaccard index (ie. using K.sum)
    # Remember this function includes edge effects (ie. it uses each isz * isz patch).
    smooth = 1e-12
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    # Keras implementations of rounded jaccard index (ie. using K.sum)
    # Remember this function includes edge effects (ie. it uses each isz * isz patch).
    smooth = 1e-12

    # Round to 0 or 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    # Calculate Jaccard index
    intersection = K.sum(y_true * y_pred_pos, axis=[0, 1, 2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, 1, 2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def stretch_nbands(bands, lower_percent=0, higher_percent=99.99):
    """
    This functions stretches the image to a specified percentile
    IMPORTANT: higher_percent=100 results in vertical lines in the image (don't know why), hence use 99.99 instead.
               Using skimage.exposure.rescale_intensity(image) results in the same issue with vertical lines.
    """
    out = np.zeros_like(bands, dtype=np.float32)
    n = bands.shape[2]
    for i in range(n):
        a = 0  # np.min(band)
        b = 1  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t <= a] = a
        t[t >= b] = b
        out[:, :, i] = t

    return out


def sentinel2_normalizer(img, threshold=12000):
    # The Sentinel-2 data has 15 significant bits, but normally maxes out between 10000-20000.
    # Here we clip and normalize to value between 0 and 1
    img = np.clip(img, 0, threshold)
    img = img / threshold

    return img


def patch_image(img, patch_size, overlap):
    # Find number of patches
    n_width = int(np.size(img, axis=1) / (patch_size - overlap))
    n_height = n_width

    # Now cut into patches
    n_bands = np.size(img, axis=2)
    img_patched = np.zeros((n_height * n_width, patch_size, patch_size, n_bands), dtype=np.float32)
    for i in range(0, n_height):
        for j in range(0, n_width):
            id = n_width * i + j

            # Define "pixel coordinates" of the patches in the whole image
            xmin = patch_size * i - i * overlap
            xmax = patch_size * i + patch_size - i * overlap
            ymin = patch_size * j - j * overlap
            ymax = patch_size * j + patch_size - j * overlap

            # Cut out the patches.
            # img_patched[id, width , height, depth]
            img_patched[id, :, :, :] = img[xmin:xmax, ymin:ymax, :]

    return img_patched, n_height, n_width  # n_height and n_width are necessary for stitching image back together


def stitch_image(img_patched, n_height, n_width, patch_size, overlap):
    isz_overlap = patch_size - overlap  # i.e. remove the overlap

    n_bands = np.size(img_patched, axis=3)

    img = np.zeros((n_height * isz_overlap, n_width * isz_overlap, n_bands))

    # Define bbox of the interior of the patch to be stitched
    xmin_overlap = int(overlap / 2)
    xmax_overlap = int(patch_size - overlap / 2)
    ymin_overlap = int(overlap / 2)
    ymax_overlap = int(patch_size - overlap / 2)

    # Stitch the patches together
    for i in range(0, n_height):
        for j in range(0, n_width):
            id = n_width * i + j

            # Cut out the interior of the patch
            interior_path = img_patched[id, xmin_overlap:xmax_overlap, ymin_overlap:ymax_overlap, :]

            # Define "pixel coordinates" of the patches in the whole image
            xmin = isz_overlap * i
            xmax = isz_overlap * i + isz_overlap
            ymin = isz_overlap * j
            ymax = isz_overlap * j + isz_overlap

            # Insert the patch into the stitched image
            img[xmin:xmax, ymin:ymax, :] = interior_path

    return img


def img_generator(type, path, batch_size, shuffle, seed):
    # Load the names of the numpy files, each containing one patch
    files = sorted(os.listdir(path + type))  # os.listdir loads in arbitrary order, hence use sorted()
    files = [f for f in files if '.npy' in f]  # Only use .npy files (e.g. avoid .gitkeep)

    # Create a placeholder for x which can hold batch_size patches
    patch_shape = np.shape(np.load(path + type + "/" + files[0]))
    x = np.zeros((batch_size, patch_shape[0], patch_shape[1], patch_shape[2]))

    # Shuffle the patches
    if shuffle:
        random.seed(seed)
        random.shuffle(files)

    # Create the generator
    index = 0
    max_index = len(files)  # max_index is the number of patches available
    while 1:
        for i in range(batch_size):
            if index >= max_index:
                index = 0

            x[i, :, :, :] = np.load(path + type + "/" + files[index])
            index = index + i

        yield x


def mask_generator(type, path, batch_size, shuffle, seed, cls, collapse_cls):
    # Load the names of the numpy files, each containing one patch
    files = sorted(os.listdir(path + type))  # os.listdir loads in arbitrary order, hence use sorted()
    files = [f for f in files if '.npy' in f]  # Only use .npy files (e.g. avoid .gitkeep)

    # Create a placeholder for x which can hold batch_size patches
    patch_shape = np.shape(np.load(path + type + "/" + files[0]))
    if collapse_cls:   # Find the number of classes
        n_cls = 1
    else:
        n_cls = np.size(cls)
    x = np.zeros((batch_size, patch_shape[0], patch_shape[1], n_cls))

    # Shuffle the patches
    if shuffle:
        random.seed(seed)
        random.shuffle(files)

    # Create the generator
    index = 0
    max_index = len(files)  # max_index is the number of patches available
    while 1:
        for i in range(batch_size):
            if index >= max_index:
                index = 0

            # Load the masks
            mask = np.load(path + type + "/" + files[index])

            # Create the binary masks
            if collapse_cls:
                extract_collapsed_cls(mask, cls)

                # Save the binary mask
                x[i, :, :, :] = mask

            else:
                # Get both the class and the iteration index in the for loop using enumerate()
                # https://stackoverflow.com/questions/522563/accessing-the-index-in-python-for-loops
                for j, c in enumerate(cls):
                    y = extract_cls_mask(mask, c)

                    # Save the binary masks as one hot representations
                    x[i, :, :, j] = y[:, :, 0]

            index = index + i

        yield x


def extract_collapsed_cls(mask, cls):
    # Remember to zeroize class 1
    if 1 not in cls:
        mask[mask == 1] = 0

    # Make a binary mask including all classes in cls
    for c in cls:
        mask[mask == c] = 1
    mask[mask != 1] = 0

    return mask


def extract_cls_mask(mask, c):
    # Copy the mask for every iteration (if you set "y=mask", then mask will be overwritten!
    # https://stackoverflow.com/questions/19951816/python-changes-to-my-copy-variable-affect-the-original-variable
    y = np.copy(mask)
    # Remember to zeroize class 1
    if c != 1:
        y[y == 1] = 0

    # Make a binary mask for the specific class
    y[y == c] = 1
    y[y != 1] = 0
    return y


def predict_img(model, hparams, img, n_bands, n_cls, cls, collapse_cls, num_gpus):
    # Stretch.
    img_norm = sentinel2_normalizer(img, threshold=12000)

    # Add zeropadding around the image (has to match the overlap)
    img_shape = np.shape(img_norm)
    img_padded = np.zeros((img_shape[0] + hparams.patch_size,
                           img_shape[1] + hparams.patch_size,
                           img_shape[2]))
    img_padded[hparams.overlap:hparams.overlap + img_shape[0],
               hparams.overlap:hparams.overlap + img_shape[1],
               :] = img_norm

    # Patch the image in isz * isz pixel patches
    img_patched, n_height, n_width = patch_image(img_padded, patch_size=hparams.patch_size, overlap=hparams.overlap)

    # Now do the cloud masking on patches
    predicted_patches = model.predict(img_patched, n_bands, n_cls, cls, collapse_cls, num_gpus, hparams)

    # Stitch the patches back together
    predicted_stitched = stitch_image(predicted_patches, n_height, n_width, patch_size=hparams.patch_size, overlap=hparams.overlap)

    # Now throw away the padded sections
    padding = int(hparams.overlap/2)  # The overlap is over 2 patches, so you need to throw away overlap/2 on each
    predicted_mask = predicted_stitched[padding:padding+img_shape[0], padding:padding+img_shape[1], :]

    # Threshold the prediction
    predicted_binary_mask = predicted_mask >= hparams.threshold

    return predicted_mask, predicted_binary_mask