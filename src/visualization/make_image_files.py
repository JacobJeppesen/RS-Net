import numpy as np
import tifffile as tiff
import os
import time
from ..utils import image_normalizer, predict_img, extract_cls_mask, extract_collapsed_cls, get_cls, get_model_name, load_product
from PIL import Image


def visualize_test_data(model, num_gpus, params):
    if params.satellite == 'Sentinel-2':
        test_data_path = params.project_path + 'data/processed/visualization/'
        files = sorted(os.listdir(test_data_path))  # os.listdir loads in arbitrary order, hence use sorted()
        files = [f for f in files if 'B02_10m.tiff' in f]  # Filter out one ID for each tile

        print('-----------------------------------------------------------------------------------------------------')
        for i, file in enumerate(files):
            print('Evaluating tile (', i+1, 'of', np.size(files), ') :', file[0:26])
            file = file[0:26]
            __visualize_sentinel2_tile__(model, file, num_gpus, params)
            print('---')

    elif params.satellite == 'Landsat8':
        folders = sorted(os.listdir(params.project_path + "data/raw/"))
        folders = [f for f in folders if '.' not in f]  # Filter out .gitignore

        i = 1
        for folder in folders:
            products = sorted(os.listdir(params.project_path + "data/raw/" + folder + "/BC/"))
            products = [f for f in products if f in params.test_tiles]  # Filter for visualization set tiles

            for product in products:
                print('Evaluating tile no.', i,':', folder, ' - ', product)
                data_path = params.project_path + "data/raw/" + folder + "/BC/" + product + "/"
                __visualize_landsat8_tile__(model, product, data_path, num_gpus, params)
                print('---')

                i += 1


def __visualize_sentinel2_tile__(model, file, num_gpus, params):
    # Measure the time it takes to load data
    start_time = time.time()

    # Find the number of classes
    if params.collapse_cls:
        n_cls = 1
    else:
        n_cls = np.size(params.cls)
    n_bands = np.size(params.bands)

    # Cut out smaller patch
    patch_min = 0
    patch_max = 10980

    # Load the RGB data for the scene
    product_path = params.project_path + 'data/processed/visualization/'
    img, img_rgb = load_product(file, params, product_path)

    # Load the original sen2cor classification mask
    mask_sen2cor = tiff.imread(product_path + file + '_SCL_20m.tiff')  # The 20 m is the native resolution
    mask_sen2cor_patch = np.zeros((patch_max - patch_min, patch_max - patch_min, n_cls))
    mask_sen2cor_patch[:, :, 0] = mask_sen2cor[patch_min:patch_max, patch_min:patch_max]

    # Load the original fmask classification mask
    mask_fmask = np.array(Image.open(product_path + file + '_Fma_20m.tiff'))  # The 20 m is the native resolution
    mask_fmask_patch = np.zeros((patch_max - patch_min, patch_max - patch_min, n_cls))
    mask_fmask_patch[:, :, 0] = mask_fmask[patch_min:patch_max, patch_min:patch_max]

    # Get the masks
    cls_sen2cor = get_cls(params.cls, 'sen2cor')
    cls_fmask = get_cls(params.cls, 'fmask')

    # Create the binary masks
    if params.collapse_cls:
        mask_sen2cor_patch = extract_collapsed_cls(mask_sen2cor_patch, cls_sen2cor)
        mask_fmask_patch = extract_cls_mask(mask_fmask_patch, cls_fmask)

    else:
        for i, c in enumerate(params.cls):
            y = extract_cls_mask(mask_sen2cor, c)

            # Save the binary masks as one hot representations
            mask_sen2cor_patch[:, :, i] = y[:, :, 0]

    # Take out small patch and enhance contrast on RGB image
    img_patch = img[patch_min:patch_max, patch_min:patch_max, :]
    img_rgb_patch = img_rgb[patch_min:patch_max, patch_min:patch_max, :]
    exec_time = str(time.time() - start_time)
    print("Data loaded in              : " + exec_time + "s")

    # Get the predicted mask
    start_time = time.time()
    predicted_mask, predicted_binary_mask = predict_img(model, params, img_patch, n_bands, n_cls, num_gpus)
    exec_time = str(time.time() - start_time)
    print("Prediction finished in      : " + exec_time + "s")

    # Have at least one pixel for each class to avoid issues with the colors in the figure
    mask_sen2cor_patch[0, 0] = 0
    mask_sen2cor_patch[0, 1] = 1
    mask_fmask_patch[0, 0] = 0
    mask_fmask_patch[0, 1] = 1
    predicted_binary_mask[0, 0] = 0
    predicted_binary_mask[0, 1] = 1

    # Save as images
    data_output_path = params.project_path + 'data/output/'
    start_time = time.time()
    img_enhanced_contrast = image_normalizer(img_rgb_patch, params, type='enhanced_contrast')
    if not os.path.isfile(data_output_path + '%s-image.tiff' % file):
        Image.fromarray(np.uint8(img_enhanced_contrast * 255)).save(data_output_path + '%s-image.tiff' % file)

    predicted_mask_rgb = np.zeros((np.shape(img_enhanced_contrast)))
    model_name = get_model_name(params)
    for i, c in enumerate(params.cls):
        # Convert predicted mask to RGB and save all masks (use PIL to save as it is much faster than matplotlib)

        # UNCOMMENT BELOW TO SAVE THRESHOLDED MASK
        #predicted_mask_rgb[:, :, 0] = predicted_mask[:, :, i] * 253  # Color coding for the mask
        #predicted_mask_rgb[:, :, 1] = predicted_mask[:, :, i] * 231
        #predicted_mask_rgb[:, :, 2] = predicted_mask[:, :, i] * 36
        #Image.fromarray(np.uint8(predicted_mask_rgb)).save(data_output_path + '%s_cls-%s_%s.tiff' % (file, params.cls, model_name))

        Image.fromarray(np.uint8(predicted_mask[:, :, i] * 255)).save(data_output_path + '%s_cls-%s_%s.tiff' % (file, params.cls, model_name))
        Image.fromarray(np.uint8(predicted_binary_mask[:, :, i])).save(data_output_path + '%s_cls-%s_thresholded_%s.tiff' % (file, params.cls, model_name))
        if not os.path.isfile(data_output_path + '%s_cls-%s_sen2cor.tiff' % (file, params.cls)):
            Image.fromarray(np.uint8(mask_sen2cor_patch[:, :, i])).save(data_output_path + '%s_cls-%s_sen2cor.tiff' % (file, params.cls))
            Image.fromarray(np.uint8(mask_fmask_patch[:, :, i])).save(data_output_path + '%s_cls-%s_fmask.tiff' % (file, params.cls))

        exec_time = str(time.time() - start_time)
        print("Images saved in             : " + exec_time + "s")

        if params.collapse_cls:
            break


def __visualize_landsat8_tile__(model, file, data_path, num_gpus, params):
    # Measure the time it takes to load data
    start_time = time.time()

    # Find the number of classes
    if params.collapse_cls:
        n_cls = 1
    else:
        n_cls = np.size(params.cls)
    n_bands = np.size(params.bands)

    # Load the RGB data for the scene
    img, img_rgb = load_product(file, params, data_path)

    # Load the true classification mask
    mask_true = tiff.imread(data_path + file + '_fixedmask.TIF')  # The 30 m is the native resolution

    # Get the masks
    cls = get_cls(params)

    # Create the binary masks
    if params.collapse_cls:
        mask_true = extract_collapsed_cls(mask_true, cls)

    else:
        for i, c in enumerate(params.cls):
            y = extract_cls_mask(mask_true, cls)

            # Save the binary masks as one hot representations
            mask_true[:, :, i] = y[:, :, 0]

    exec_time = str(time.time() - start_time)
    print("Data loaded in        : " + exec_time + "s")

    # Get the predicted mask
    start_time = time.time()
    predicted_mask, predicted_binary_mask = predict_img(model, params, img, n_bands, n_cls, num_gpus)
    exec_time = str(time.time() - start_time)
    print("Prediction finished in: " + exec_time + "s")

    # Have at least one pixel for each class to avoid issues with the colors in the figure
    mask_true[0, 0] = 0
    mask_true[0, 1] = 1
    predicted_binary_mask[0, 0] = 0
    predicted_binary_mask[0, 1] = 1

    # Save as images
    data_output_path = params.project_path + 'data/output/'
    start_time = time.time()
    img_enhanced_contrast = image_normalizer(img_rgb, params, type='enhanced_contrast')
    if not os.path.isfile(data_output_path + '%s-image.tiff' % file):
        Image.fromarray(np.uint8(img_enhanced_contrast * 255)).save(data_output_path + '%s-image.tiff' % file)

    predicted_mask_rgb = np.zeros((np.shape(img_enhanced_contrast)))
    model_name = get_model_name(params)
    #for i, c in enumerate(params.cls):
    for i, c in enumerate([0]):
        # Convert predicted mask to RGB and save all masks (use PIL to save as it is much faster than matplotlib)

        # UNCOMMENT BELOW TO SAVE THRESHOLDED MASK
        #predicted_mask_rgb[:, :, 0] = predicted_mask[:, :, i] * 253  # Color coding for the mask
        #predicted_mask_rgb[:, :, 1] = predicted_mask[:, :, i] * 231
        #predicted_mask_rgb[:, :, 2] = predicted_mask[:, :, i] * 36
        #Image.fromarray(np.uint8(predicted_mask_rgb)).save(data_output_path + '%s_cls-%s_%s.tiff' % (file, params.cls, model_name))

        Image.fromarray(np.uint8(predicted_mask[:, :, i] * 255)).save(data_output_path + '%s_%s.tiff' % (file, model_name))
        Image.fromarray(np.uint8(predicted_binary_mask[:, :, i])).save(data_output_path + '%s_thresholded_%s.tiff' % (file, model_name))
        if not os.path.isfile(data_output_path + '%s_cls-%s.tiff' % (file, params.cls)):
            Image.fromarray(np.uint8(mask_true)).save(data_output_path + '%s_true_cls-%s_collapse%s.tiff' % (file, "".join(str(c) for c in params.cls), params.collapse_cls))

        exec_time = str(time.time() - start_time)
        print("Images saved in       : " + exec_time + "s")

        if params.collapse_cls:
            break