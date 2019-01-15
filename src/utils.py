#!/usr/bin/env python
import numpy as np
import tifffile as tiff
import cv2
import time


def image_normalizer(img, params, type='enhance_contrast'):
    """
    Clip an image at certain threshold value, and then normalize to values between 0 and 1.
    Threshold is used for contrast enhancement.
    """
    if type == 'enhance_contrast':  # Enhance contrast of entire image
        # The Sentinel-2 data has 15 significant bits, but normally maxes out between 10000-20000.
        # Here we clip and normalize to value between 0 and 1
        img_norm = np.clip(img, 0, params.norm_threshold)
        img_norm = img_norm / params.norm_threshold

    elif type == 'running_normalization':  # Normalize each band of each incoming image based on that image
        # Based on stretch_n function found at https://www.kaggle.com/drn01z3/end-to-end-baseline-with-u-net-keras
        min_value = 0
        max_value = 1

        lower_percent = 0.02  # Used to discard lower bound outliers
        higher_percent = 0.98  # Used to discard upper bound outliers

        bands = img.shape[2]
        img_norm = np.zeros_like(img)

        for i in range(bands):
            c = np.percentile(img[:, :, i], lower_percent)
            d = np.percentile(img[:, :, i], higher_percent)
            t = min_value + (img[:, :, i] - c) * (max_value - min_value) / (d - c)
            t[t < min_value] = min_value
            t[t > max_value] = max_value
            img_norm[:, :, i] = t

    elif type == 'landsat8_biome_normalization':  # Normalize each band of each incoming image based on Landsat8 Biome
        # Standard deviations used for standardization
        std_devs = 4

        # Normalizes to zero mean and half standard deviation (find values in 'jhj_InspectLandsat8Data' notebook)
        img_norm = np.zeros_like(img)
        for i, b in enumerate(params.bands):
            if b == 1:
                img_norm[:, :, i] = (img[:, :, i] - 4654) / (std_devs * 1370)
            elif b == 2:
                img_norm[:, :, i] = (img[:, :, i] - 4435) / (std_devs * 1414)
            elif b == 3:
                img_norm[:, :, i] = (img[:, :, i] - 4013) / (std_devs * 1385)
            elif b == 4:
                img_norm[:, :, i] = (img[:, :, i] - 4112) / (std_devs * 1488)
            elif b == 5:
                img_norm[:, :, i] = (img[:, :, i] - 4776) / (std_devs * 1522)
            elif b == 6:
                img_norm[:, :, i] = (img[:, :, i] - 2371) / (std_devs * 998)
            elif b == 7:
                img_norm[:, :, i] = (img[:, :, i] - 1906) / (std_devs * 821)
            elif b == 8:
                img_norm[:, :, i] = (img[:, :, i] - 18253) / (std_devs * 4975)
            elif b == 9:
                img_norm[:, :, i] = (img[:, :, i] - 380) / (std_devs * 292)
            elif b == 10:
                img_norm[:, :, i] = (img[:, :, i] - 19090) / (std_devs * 2561)
            elif b == 11:
                img_norm[:, :, i] = (img[:, :, i] - 17607) / (std_devs * 2119)

    return img_norm


def patch_image(img, patch_size, overlap):
    """
    Split up an image into smaller overlapping patches
    """
    # TODO: Get the size of the padding right.
    # Add zeropadding around the image (has to match the overlap)
    img_shape = np.shape(img)
    img_padded = np.zeros((img_shape[0] + 2*patch_size, img_shape[1] + 2*patch_size, img_shape[2]))
    img_padded[overlap:overlap + img_shape[0], overlap:overlap + img_shape[1], :] = img

    # Find number of patches
    n_width = int((np.size(img_padded, axis=0) - patch_size) / (patch_size - overlap))
    n_height = int((np.size(img_padded, axis=1) - patch_size) / (patch_size - overlap))

    # Now cut into patches
    n_bands = np.size(img_padded, axis=2)
    img_patched = np.zeros((n_height * n_width, patch_size, patch_size, n_bands), dtype=img.dtype)
    for i in range(0, n_width):
        for j in range(0, n_height):
            id = n_height * i + j

            # Define "pixel coordinates" of the patches in the whole image
            xmin = patch_size * i - i * overlap
            xmax = patch_size * i + patch_size - i * overlap
            ymin = patch_size * j - j * overlap
            ymax = patch_size * j + patch_size - j * overlap

            # Cut out the patches.
            # img_patched[id, width , height, depth]
            img_patched[id, :, :, :] = img_padded[xmin:xmax, ymin:ymax, :]

    return img_patched, n_height, n_width  # n_height and n_width are necessary for stitching image back together


def stitch_image(img_patched, n_height, n_width, patch_size, overlap):
    """
    Stitch the overlapping patches together to one large image (the original format)
    """
    isz_overlap = patch_size - overlap  # i.e. remove the overlap

    n_bands = np.size(img_patched, axis=3)

    img = np.zeros((n_width * isz_overlap, n_height * isz_overlap, n_bands))

    # Define bbox of the interior of the patch to be stitched (not required if using Cropping2D layer in model)
    #xmin_overlap = int(overlap / 2)
    #xmax_overlap = int(patch_size - overlap / 2)
    #ymin_overlap = int(overlap / 2)
    #ymax_overlap = int(patch_size - overlap / 2)

    # Stitch the patches together
    for i in range(0, n_width):
        for j in range(0, n_height):
            id = n_height * i + j

            # Cut out the interior of the patch
            #interior_path = img_patched[id, xmin_overlap:xmax_overlap, ymin_overlap:ymax_overlap, :]
            interior_patch = img_patched[id, :, :, :]

            # Define "pixel coordinates" of the patches in the whole image
            xmin = isz_overlap * i
            xmax = isz_overlap * i + isz_overlap
            ymin = isz_overlap * j
            ymax = isz_overlap * j + isz_overlap

            # Insert the patch into the stitched image
            img[xmin:xmax, ymin:ymax, :] = interior_patch

    return img


def extract_collapsed_cls(mask, cls):
    """
    Combine several classes to one binary mask
    """
    # Remember to zeroize class 1
    if 1 not in cls:
        mask[mask == 1] = 0

    # Make a binary mask including all classes in cls
    for c in cls:
        mask[mask == c] = 1
    mask[mask != 1] = 0

    return mask


def extract_cls_mask(mask, c):
    """
    Create a binary mask for a class with integer value c (i.e. if class 'cloud' = 255, then change it to 'cloud' = 1)
    """
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


def predict_img(model, params, img, n_bands, n_cls, num_gpus):
    """
    Run prediction on an full image
    """
    # Find dimensions
    img_shape = np.shape(img)

    # Normalize the product
    img = image_normalizer(img, params, type=params.norm_method)

    # Patch the image in patch_size * patch_size pixel patches
    img_patched, n_height, n_width = patch_image(img, patch_size=params.patch_size, overlap=params.overlap)

    # Now find all completely black patches and inpaint partly black patches
    indices = []  # Used to ignore completely black patches during prediction
    use_inpainting = False
    for i in range(0, np.shape(img_patched)[0]):  # For all patches
        if np.any(img_patched[i, :, :, :] == 0):  # If any black pixels
            if np.mean(img_patched[i, :, :, :] != 0):  # Ignore completely black patches
                indices.append(i)  # Use the patch for prediction
                # Fill in zero pixels using the non-zero pixels in the patch
                for j in range(0, np.shape(img_patched)[3]):  # Loop over each spectral band in the patch
                    # Use more advanced inpainting method
                    if use_inpainting:
                        zero_mask = np.zeros_like(img_patched[i, :, :, j])
                        zero_mask[img_patched[i, :, :, j] == 0] = 1
                        inpainted_patch = cv2.inpaint(np.uint8(img_patched[i, :, :, j] * 255),
                                                      np.uint8(zero_mask),
                                                      inpaintRadius=5,
                                                      flags=cv2.INPAINT_TELEA)

                        img_patched[i, :, :, j] = np.float32(inpainted_patch) / 255
                    # Use very simple inpainting method (fill in the mean value)
                    else:
                        # Bands do not always overlap. Use mean of all bands if zero-slice is found, otherwise use
                        # mean of the specific band
                        if np.mean(img_patched[i, :, :, j]) == 0:
                            mean_value = np.mean(img_patched[i, img_patched[i, :, :, :] != 0])
                        else:
                            mean_value = np.mean(img_patched[i, img_patched[i, :, :, j] != 0, j])

                        img_patched[i, img_patched[i, :, :, j] == 0, j] = mean_value
        else:
            indices.append(i)  # Use the patch for prediction

    # Now do the cloud masking (on non-zero patches according to indices)
    #start_time = time.time()
    predicted_patches = np.zeros((np.shape(img_patched)[0],
                                  params.patch_size-params.overlap, params.patch_size-params.overlap, n_cls))
    predicted_patches[indices, :, :, :] = model.predict(img_patched[indices, :, :, :], n_bands, n_cls, num_gpus, params)
    #exec_time = str(time.time() - start_time)
    #print("Prediction of patches (not including splitting and stitching) finished in: " + exec_time + "s")

    # Stitch the patches back together
    predicted_stitched = stitch_image(predicted_patches, n_height, n_width, patch_size=params.patch_size, overlap=params.overlap)

    # Now throw away the padded sections from the overlap
    padding = int(params.overlap / 2)  # The overlap is over 2 patches, so you need to throw away overlap/2 on each
    predicted_mask = predicted_stitched[padding-1:padding-1+img_shape[0],  # padding-1 because it is index in array
                                        padding-1:padding-1+img_shape[1],
                                        :]

    # Throw away the inpainting of the zero pixels in the individual patches
    # The summation is done to ensure that all pixels are included. The bands do not perfectly overlap (!)
    predicted_mask[np.sum(img, axis=2) == 0] = 0

    # Threshold the prediction
    predicted_binary_mask = predicted_mask >= np.float32(params.threshold)

    return predicted_mask, predicted_binary_mask


def get_cls(satellite, dataset, cls_string):
    """
    Get the classes from the integer values in the true masks (i.e. 'water' in sen2cor has integer value 3)
    """
    if satellite == 'Sentinel-2':
        if dataset == 'sen2cor':
            if cls_string == 'clear':
                cls_int = [2, 4, 5]
            elif cls_string == 'cloud':
                cls_int = [8, 9]
            elif cls_string == 'shadow':
                cls_int = [3]
            elif cls_string == 'snow':
                cls_int = [11]
            elif cls_string == 'water':
                cls_int = [6]

        elif dataset == 'fmask':
            if cls_string == 'clear':
                cls_int = [1]
            elif cls_string == 'cloud':
                cls_int = [2]
            elif cls_string == 'shadow':
                cls_int = [3]
            elif cls_string == 'snow':
                cls_int = [4]
            elif cls_string == 'water':
                cls_int = [5]

    elif satellite == 'Landsat8':
        if dataset == 'Biome_gt':
            cls_int = []
            for c in cls_string:
                if c == 'fill':
                    cls_int.append(0)
                elif c == 'shadow':
                    cls_int.append(64)
                elif c == 'clear':
                    cls_int.append(128)
                elif c == 'thin':
                    cls_int.append(192)
                elif c == 'cloud':
                    cls_int.append(255)

        elif dataset == 'Biome_fmask':
            cls_int = []
            for c in cls_string:
                if c == 'fill':
                    cls_int.append(0)
                elif c == 'clear':
                    cls_int.append(1)
                elif c == 'cloud':
                    cls_int.append(2)
                elif c == 'shadow':
                    cls_int.append(3)
                elif c == 'snow':
                    cls_int.append(4)
                elif c == 'water':
                    cls_int.append(5)

        elif dataset == 'SPARCS_gt':
            cls_int = []
            for c in cls_string:
                if c == 'shadow':
                    cls_int.append(0)
                    cls_int.append(1)
                elif c == 'water':
                    cls_int.append(2)
                    cls_int.append(6)
                elif c == 'snow':
                    cls_int.append(3)
                elif c == 'cloud':
                    cls_int.append(5)
                elif c == 'clear':
                    cls_int.append(4)

        elif dataset == 'SPARCS_fmask':
            cls_int = []
            for c in cls_string:
                if c == 'fill':
                    cls_int.append(0)
                elif c == 'clear':
                    cls_int.append(1)
                elif c == 'cloud':
                    cls_int.append(2)
                elif c == 'shadow':
                    cls_int.append(3)
                elif c == 'snow':
                    cls_int.append(4)
                elif c == 'water':
                    cls_int.append(5)
    return cls_int


def get_model_name(params):
    '''
    Combine the parameters for the model into a string (to name the model file)
    '''
    if params.modelID:
        model_name = 'Unet_' + params.satellite + '_' + params.modelID
    elif params.satellite == 'Sentinel-2':
        model_name = 'sentinel2_unet_cls-' + "".join(str(c) for c in params.cls) + \
                     '_initmodel-' + params.initial_model + \
                     '_collapse' + str(params.collapse_cls) + \
                     '_bands' + "".join(str(b) for b in params.bands) + \
                     "_lr" + str(params.learning_rate) + \
                     '_decay' + str(params.decay) + \
                     '_L2reg' + str(params.L2reg) + \
                     '_dropout' + str(params.dropout) + '.hdf5'
    elif params.satellite == 'Landsat8':
        model_name = 'landsat8_unet_cls-' + "".join(str(c) for c in params.cls) + \
                     '_collapse' + str(params.collapse_cls) + \
                     '_bands' + "".join(str(b) for b in params.bands) + \
                     "_lr" + str(params.learning_rate) + \
                     '_decay' + str(params.decay) + \
                     '_L2reg' + str(params.L2reg) + \
                     '_dropout' + str(params.dropout) + '.hdf5'
    return model_name


def load_product(name, params, product_path, toa_path):

    if params.satellite == 'Sentinel-2':
        img_rgb = np.zeros((10980, 10980, 3))
        img_rgb[:, :, 2] = tiff.imread(product_path + name + '_B02_10m.tiff')
        img_rgb[:, :, 1] = tiff.imread(product_path + name + '_B03_10m.tiff')
        img_rgb[:, :, 0] = tiff.imread(product_path + name + '_B04_10m.tiff')

        # Load the img to be used for prediction
        img = np.zeros((10980, 10980, np.size(params.bands)))
        for i, b in enumerate(params.bands):  # Not beautiful. Should not have included resolution in filename
            if b == 1:
                img[:, :, i] = tiff.imread(product_path + name + '_B01_60m.tiff')
            elif b == 2:
                img[:, :, i] = tiff.imread(product_path + name + '_B02_10m.tiff')
            elif b == 3:
                img[:, :, i] = tiff.imread(product_path + name + '_B03_10m.tiff')
            elif b == 4:
                img[:, :, i] = tiff.imread(product_path + name + '_B04_10m.tiff')
            elif b == 5:
                img[:, :, i] = tiff.imread(product_path + name + '_B05_20m.tiff')
            elif b == 6:
                img[:, :, i] = tiff.imread(product_path + name + '_B06_20m.tiff')
            elif b == 7:
                img[:, :, i] = tiff.imread(product_path + name + '_B07_20m.tiff')
            elif b == 8:
                img[:, :, i] = tiff.imread(product_path + name + '_B08_10m.tiff')
            elif b == 9:
                img[:, :, i] = tiff.imread(product_path + name + '_B09_60m.tiff')
            elif b == 10:
                img[:, :, i] = tiff.imread(product_path + name + '_B10_60m.tiff')
            elif b == 11:
                img[:, :, i] = tiff.imread(product_path + name + '_B11_20m.tiff')
            elif b == 12:
                img[:, :, i] = tiff.imread(product_path + name + '_B12_20m.tiff')
            elif b == 13:
                img[:, :, i] = tiff.imread(product_path + name + '_B8A_20m.tiff')

    elif params.satellite == 'Landsat8':

        # Load TOA and set all NaN values to 0
        toa = tiff.imread(toa_path + name + '_toa.TIF')
        toa[toa == 32767] = 0

        # Create RGB image
        width, height = np.size(toa, axis=0), np.size(toa, axis=1)
        img_rgb = np.zeros((width, height, 3))
        img_rgb[:, :, 0] = toa[:, :, 3]
        img_rgb[:, :, 1] = toa[:, :, 2]
        img_rgb[:, :, 2] = toa[:, :, 1]

        # Load the img to be used for prediction
        img = np.zeros((width, height, np.size(params.bands)), dtype=np.float32)
        for i, b in enumerate(params.bands):  # Not beautiful. Should not have included resolution in filename
            if b == 1:
                img[:, :, i] = toa[:, :, 0]
            elif b == 2:
                img[:, :, i] = toa[:, :, 1]
            elif b == 3:
                img[:, :, i] = toa[:, :, 2]
            elif b == 4:
                img[:, :, i] = toa[:, :, 3]
            elif b == 5:
                img[:, :, i] = toa[:, :, 4]
            elif b == 6:
                img[:, :, i] = toa[:, :, 5]
            elif b == 7:
                img[:, :, i] = toa[:, :, 6]
            elif b == 8:
                img[:, :, i] = tiff.imread(product_path + name + '_B8.TIF')
            elif b == 9:
                img[:, :, i] = toa[:, :, 7]

            if b == 10:
                img[:, :, i] = tiff.imread(product_path + name + '_B10.TIF')
            elif b == 11:
                img[:, :, i] = tiff.imread(product_path + name + '_B11.TIF')

        # Create the overlapping pixels mask (Band 1-8 are overlapping, but 9, 10, 11 vary at the border regions)

        # Find the valid pixels and cast to uint8 to reduce processing time. The
        #
        overlapping_pixels_mask = np.uint8(np.clip(toa[:, :, 0], 0, 1)) & \
                                  np.uint8(np.clip(toa[:, :, 7], 0, 1)) & \
                                  np.uint8(np.clip(tiff.imread(product_path + name + '_B10.TIF'), 0, 1)) & \
                                  np.uint8(np.clip(tiff.imread(product_path + name + '_B11.TIF'), 0, 1))

    return img, img_rgb, overlapping_pixels_mask


