import os
import time
import numpy as np
import tifffile as tiff
from PIL import Image
from src.utils import load_product, get_cls, extract_collapsed_cls, extract_cls_mask, predict_img, image_normalizer


def evaluate_test_set(model, dataset, num_gpus, params, save_output=False, write_csv=True):
    if dataset == 'Biome_gt':
        __evaluate_biome_dataset__(model, num_gpus, params, save_output=save_output, write_csv=write_csv)

    elif dataset == 'SPARCS_gt':
        __evaluate_sparcs_dataset__(model, num_gpus, params, save_output=save_output, write_csv=write_csv)


def __evaluate_sparcs_dataset__(model, num_gpus, params, save_output=False, write_csv=True):
    # Find the number of classes and bands
    if params.collapse_cls:
        n_cls = 1
    else:
        n_cls = np.size(params.cls)
    n_bands = np.size(params.bands)

    # Get the name of all the products (scenes)
    data_path = params.project_path + "data/raw/SPARCS_dataset/"
    toa_path = params.project_path + "data/processed/SPARCS_TOA/"
    products = sorted(os.listdir(data_path))
    products = [p for p in products if 'data.tif' in p]
    products = [p for p in products if 'xml' not in p]

    # If doing CV, only evaluate on test split
    if params.split_dataset:
        products = params.test_tiles[1]

    # Define thresholds and initialize evaluation metrics dict
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    evaluation_metrics = {}
    evaluating_product_no = 1  # Used in print statement later

    for product in products:
        # Time the prediction
        start_time = time.time()

        # Load data
        img_all_bands = tiff.imread(data_path + product)
        img_all_bands[:, :, 0:8] = tiff.imread(toa_path + product[:-8] + 'toa.TIF')

        # Load the relevant bands and the mask
        img = np.zeros((np.shape(img_all_bands)[0], np.shape(img_all_bands)[1], np.size(params.bands)))
        for i, b in enumerate(params.bands):
            if b < 8:
                img[:, :, i] = img_all_bands[:, :, b-1]
            else:  # Band 8 is not included in the tiff file
                img[:, :, i] = img_all_bands[:, :, b-2]

        # Load true mask
        mask_true = np.array(Image.open(data_path + product[0:25] + 'mask.png'))

        # Pad the image for improved borders
        padding_size = params.overlap
        npad = ((padding_size, padding_size), (padding_size, padding_size), (0, 0))
        img_padded = np.pad(img, pad_width=npad, mode='symmetric')

        # Get the masks
        #cls = get_cls(params)
        cls = [5]  # TODO: Currently hardcoded to look at clouds - fix it!

        # Create the binary masks
        if params.collapse_cls:
            mask_true = extract_collapsed_cls(mask_true, cls)

        else:
            for l, c in enumerate(params.cls):
                y = extract_cls_mask(mask_true, cls)

                # Save the binary masks as one hot representations
                mask_true[:, :, l] = y[:, :, 0]

        # Predict the images
        predicted_mask_padded, _ = predict_img(model, params, img_padded, n_bands, n_cls, num_gpus)

        # Remove padding
        predicted_mask = predicted_mask_padded[padding_size:-padding_size, padding_size:-padding_size, :]

        # Create a nested dict to save evaluation metrics for each product
        evaluation_metrics[product] = {}

        # Find the valid pixels and cast to uint8 to reduce processing time
        valid_pixels_mask = np.uint8(np.clip(img[:, :, 0], 0, 1))
        mask_true = np.uint8(mask_true)

        # Loop over different threshold values
        for j, threshold in enumerate(thresholds):
            predicted_binary_mask = np.uint8(predicted_mask >= threshold)

            accuracy, omission, comission, pixel_jaccard, precision, recall, f_one_score, tp, tn, fp, fn, npix = calculate_evaluation_criteria(
                valid_pixels_mask, predicted_binary_mask, mask_true)

            # Create an additional nesting in the dict for each threshold value
            evaluation_metrics[product]['threshold_' + str(threshold)] = {}

            # Save the values in the dict
            evaluation_metrics[product]['threshold_' + str(threshold)]['tp'] = tp
            evaluation_metrics[product]['threshold_' + str(threshold)]['fp'] = fp
            evaluation_metrics[product]['threshold_' + str(threshold)]['fn'] = fn
            evaluation_metrics[product]['threshold_' + str(threshold)]['tn'] = tn
            evaluation_metrics[product]['threshold_' + str(threshold)]['npix'] = npix
            evaluation_metrics[product]['threshold_' + str(threshold)]['accuracy'] = accuracy
            evaluation_metrics[product]['threshold_' + str(threshold)]['precision'] = precision
            evaluation_metrics[product]['threshold_' + str(threshold)]['recall'] = recall
            evaluation_metrics[product]['threshold_' + str(threshold)]['f_one_score'] = f_one_score
            evaluation_metrics[product]['threshold_' + str(threshold)]['omission'] = omission
            evaluation_metrics[product]['threshold_' + str(threshold)]['comission'] = comission
            evaluation_metrics[product]['threshold_' + str(threshold)]['pixel_jaccard'] = pixel_jaccard

        print('Testing product ', evaluating_product_no, ':', product)

        exec_time = str(time.time() - start_time)
        print("Prediction finished in      : " + exec_time + "s")
        for threshold in thresholds:
            print("threshold=" + str(threshold) +
                  ": tp=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['tp']) +
                  ": fp=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['fp']) +
                  ": fn=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['fn']) +
                  ": tn=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['tn']) +
                  ": Accuracy=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['accuracy']) +
                  ": precision=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['precision']) +
                  ": recall=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['recall']) +
                  ": omission=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['omission']) +
                  ": comission=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['comission']) +
                  ": pixel_jaccard=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['pixel_jaccard']))

        evaluating_product_no += 1

        # Save images and predictions
        data_output_path = params.project_path + "data/output/SPARCS/"
        if not os.path.isfile(data_output_path + '%s_photo.png' % product[0:24]):
            Image.open(data_path + product[0:25] + 'photo.png').save(data_output_path + '%s_photo.png' % product[0:24])
            Image.open(data_path + product[0:25] + 'mask.png').save(data_output_path + '%s_mask.png' % product[0:24])

        if save_output:
            # Save predicted mask as 16 bit png file (https://github.com/python-pillow/Pillow/issues/2970)
            arr = np.uint16(predicted_mask[:, :, 0] * 65535)
            array_buffer = arr.tobytes()
            img = Image.new("I", arr.T.shape)
            img.frombytes(array_buffer, 'raw', "I;16")
            if save_output:
                img.save(data_output_path + '%s-model%s-prediction.png' % (product[0:24], params.modelID))

            #Image.fromarray(np.uint8(predicted_mask[:, :, 0]*255)).save(data_output_path + '%s-model%s-prediction.png' % (product[0:24], params.modelID))

    exec_time = str(time.time() - start_time)
    print("Dataset evaluated in: " + exec_time + "s")
    print("Or " + str(float(exec_time)/np.size(products)) + "s per image")

    if write_csv:
        write_csv_files(evaluation_metrics, params)


def __evaluate_biome_dataset__(model, num_gpus, params, save_output=False, write_csv=True):
    """
    Evaluates all products in data/processed/visualization folder, and returns performance metrics
    """
    print('------------------------------------------')
    print("Evaluate model on visualization data set:")

    # Find the number of classes and bands
    if params.collapse_cls:
        n_cls = 1
    else:
        n_cls = np.size(params.cls)
    n_bands = np.size(params.bands)

    folders = sorted(os.listdir(params.project_path + "data/raw/Biome_dataset/"))
    folders = [f for f in folders if '.' not in f]  # Filter out .gitignore

    product_names = []

    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    evaluation_metrics = {}
    evaluating_product_no = 1  # Used in print statement later

    # Used for timing tests
    load_time = []
    prediction_time = []
    threshold_loop_time = []
    save_time = []
    total_time = []

    for folder in folders:
        print('#########################')
        print('TESTING BIOME: ' + folder)
        print('#########################')
        products = sorted(os.listdir(params.project_path + "data/raw/Biome_dataset/" + folder + "/BC/"))
        
        # If doing CV, only evaluate on test split
        if params.split_dataset:
            print('NOTE: THE BIOME DATASET HAS BEEN SPLIT INTO TRAIN AND TEST')
            products = [f for f in products if f in params.test_tiles[1]]
        else:
            print('NOTE: THE ENTIRE BIOME DATASET IS CURRENTLY BEING USED FOR TEST')

        for product in products:
            print('------------------------------------------')
            print('Testing product ', evaluating_product_no, ':', product)
            data_path = params.project_path + "data/raw/Biome_dataset/" + folder + "/BC/" + product + "/"
            toa_path = params.project_path + "data/processed/Biome_TOA/" + folder + "/BC/" + product + "/"

            product_names.append(product)

            start_time = time.time()
            img, img_rgb, valid_pixels_mask = load_product(product, params, data_path, toa_path)
            load_time.append(time.time() - start_time)

            # Load the true classification mask
            mask_true = tiff.imread(data_path + product + '_fixedmask.TIF')  # The 30 m is the native resolution

            # Get the masks
            cls = get_cls('Landsat8', 'Biome_gt', params.cls)

            # Create the binary masks
            if params.collapse_cls:
                mask_true = extract_collapsed_cls(mask_true, cls)

            else:
                for l, c in enumerate(params.cls):
                    y = extract_cls_mask(mask_true, cls)

                    # Save the binary masks as one hot representations
                    mask_true[:, :, l] = y[:, :, 0]

            prediction_time_start = time.time()
            predicted_mask, _ = predict_img(model, params, img, n_bands, n_cls, num_gpus)
            prediction_time.append(time.time() - prediction_time_start)

            # Create a nested dict to save evaluation metrics for each product
            evaluation_metrics[product] = {}

            threshold_loop_time_start = time.time()
            mask_true = np.uint8(mask_true)
            # Loop over different threshold values
            for j, threshold in enumerate(thresholds):
                predicted_binary_mask = np.uint8(predicted_mask >= threshold)

                accuracy, omission, comission, pixel_jaccard, precision, recall, f_one_score, tp, tn, fp, fn, npix = calculate_evaluation_criteria(
                    valid_pixels_mask, predicted_binary_mask, mask_true)

                # Create an additional nesting in the dict for each threshold value
                evaluation_metrics[product]['threshold_' + str(threshold)] = {}

                # Save the values in the dict
                evaluation_metrics[product]['threshold_' + str(threshold)]['biome'] = folder  # Save biome type too
                evaluation_metrics[product]['threshold_' + str(threshold)]['tp'] = tp
                evaluation_metrics[product]['threshold_' + str(threshold)]['fp'] = fp
                evaluation_metrics[product]['threshold_' + str(threshold)]['fn'] = fn
                evaluation_metrics[product]['threshold_' + str(threshold)]['tn'] = tn
                evaluation_metrics[product]['threshold_' + str(threshold)]['npix'] = npix
                evaluation_metrics[product]['threshold_' + str(threshold)]['accuracy'] = accuracy
                evaluation_metrics[product]['threshold_' + str(threshold)]['precision'] = precision
                evaluation_metrics[product]['threshold_' + str(threshold)]['recall'] = recall
                evaluation_metrics[product]['threshold_' + str(threshold)]['f_one_score'] = f_one_score
                evaluation_metrics[product]['threshold_' + str(threshold)]['omission'] = omission
                evaluation_metrics[product]['threshold_' + str(threshold)]['comission'] = comission
                evaluation_metrics[product]['threshold_' + str(threshold)]['pixel_jaccard'] = pixel_jaccard

            for threshold in thresholds:
                print("threshold=" + str(threshold) +
                      ": tp=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['tp']) +
                      ": fp=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['fp']) +
                      ": fn=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['fn']) +
                      ": tn=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['tn']) +
                      ": Accuracy=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['accuracy']) +
                      ": precision=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['precision'])+
                      ": recall=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['recall']) +
                      ": omission=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['omission']) +
                      ": comission=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['comission'])+
                      ": pixel_jaccard=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['pixel_jaccard']))

            threshold_loop_time.append(time.time() - threshold_loop_time_start)

            evaluating_product_no += 1

            # Save images and predictions
            save_time_start = time.time()
            data_output_path = params.project_path + "data/output/Biome/"
            if not os.path.isfile(data_output_path + '%s-photo.png' % product):
                img_enhanced_contrast = image_normalizer(img_rgb, params, type='enhance_contrast')
                Image.fromarray(np.uint8(img_enhanced_contrast * 255)).save(data_output_path + '%s-photo.png' % product)
                Image.open(data_path + product + '_fixedmask.TIF').save(data_output_path + '%s-mask.png' % product)

            # Save predicted mask as 16 bit png file (https://github.com/python-pillow/Pillow/issues/2970)
            arr = np.uint16(predicted_mask[:, :, 0] * 65535)
            array_buffer = arr.tobytes()
            img = Image.new("I", arr.T.shape)
            img.frombytes(array_buffer, 'raw', "I;16")
            if save_output:
                img.save(data_output_path + '%s-model%s-prediction.png' % (product, params.modelID))
            save_time.append(time.time() - save_time_start)

            #Image.fromarray(np.uint16(predicted_mask[:, :, 0] * 65535)).\
            #    save(data_output_path + '%s-model%s-prediction.png' % (product, params.modelID))

            total_time.append(time.time() - start_time)
            print("Data loaded in                       : " + str(load_time[-1]) + "s")
            print("Prediction finished in               : " + str(prediction_time[-1]) + "s")
            print("Threshold loop finished in           : " + str(threshold_loop_time[-1]) + "s")
            print("Results saved in                     : " + str(save_time[-1]) + "s")
            print("Total time for product finished in   : " + str(total_time[-1]) + "s")

    # Print timing results
    print("Timing analysis for Biome dataset:")
    print("Load time: mean val.=" + str(np.mean(load_time)) + ", with std.=" + str(np.std(load_time)))
    print("Prediction time: mean val.=" + str(np.mean(prediction_time)) + ", with std.=" + str(np.std(prediction_time)))
    print("Threshold loop time: mean val.=" + str(np.mean(threshold_loop_time)) + ", with std.=" + str(np.std(threshold_loop_time)))
    print("Save time: mean val.=" + str(np.mean(save_time)) + ", with std.=" + str(np.std(save_time)))
    print("Total time: mean val.=" + str(np.mean(total_time)) + ", with std.=" + str(np.std(total_time)))

    # The mean jaccard index is not a weighted average of the number of pixels, because the number of pixels in the
    # product is dependent on the angle of the product. I.e., if the visible pixels are tilted 45 degrees, there will
    # be a lot of NaN pixels. Hence, the number of visible pixels is constant for all products.
    # for i, threshold in enumerate(thresholds):
    #     params.threshold = threshold  # Used when writing the csv files
    #     write_csv_files(np.mean(pixel_jaccard[i, :]), pixel_jaccard[i, :], product_names, params)
    if write_csv:
        write_csv_files(evaluation_metrics, params)


def calculate_evaluation_criteria(valid_pixels_mask, predicted_binary_mask, true_binary_mask):
    # From https://www.kaggle.com/lopuhin/full-pipeline-demo-poly-pixels-ml-poly
    # with correction for toggling true/false from
    # https://stackoverflow.com/questions/39164786/invert-0-and-1-in-a-binary-array
    # Need to AND with the a mask showing where there are pixels to avoid including pixels with value=0

    # Count number of actual pixels
    npix = valid_pixels_mask.sum()

    if np.ndim(predicted_binary_mask) == 3:
        tp = ((predicted_binary_mask[:, :, 0] & true_binary_mask) & valid_pixels_mask).sum()
        fp = ((predicted_binary_mask[:, :, 0] & (1 - true_binary_mask)) & valid_pixels_mask).sum()
        fn = (((1 - predicted_binary_mask)[:, :, 0] & true_binary_mask) & valid_pixels_mask).sum()
        #tn = (((1 - predicted_binary_mask)[:, :, 0] & (1 - true_binary_mask)) & actual_pixels_mask).sum()
        tn = npix - tp - fp - fn
        
    else:
        tp = ((predicted_binary_mask & true_binary_mask) & valid_pixels_mask).sum()
        fp = ((predicted_binary_mask & (1 - true_binary_mask)) & valid_pixels_mask).sum()
        fn = (((1 - predicted_binary_mask) & true_binary_mask) & valid_pixels_mask).sum()
        #tn = (((1 - predicted_binary_mask) & (1 - true_binary_mask)) & actual_pixels_mask).sum()
        tn = npix - tp - fp - fn

    # Calculate metrics
    accuracy = (tp + tn) / npix
    if tp != 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_one_score = 2 * (precision * recall) / (precision + recall)
        # See https://en.wikipedia.org/wiki/Jaccard_index#Similarity_of_asymmetric_binary_attributes
        pixel_jaccard = tp / (tp + fp + fn)
    else:
        precision = recall = f_one_score = pixel_jaccard = 0

    # Metrics from Foga 2017 paper
    if fp != 0:
        omission = fp / (tp + fp)
        comission = fp / (tn + fn)

    else:
        omission = comission = 0


    return accuracy, omission, comission, pixel_jaccard, precision, recall, f_one_score, tp, tn, fp, fn, npix


def __evaluate_sentinel2_dataset__(model, num_gpus, params):
    test_data_path = params.project_path + 'data/processed/visualization/'
    files = sorted(os.listdir(test_data_path))  # os.listdir loads in arbitrary order, hence use sorted()
    files = [f for f in files if 'B02_10m.tiff' in f]  # Filter out one ID for each tile

    print('-----------------------------------------------------------------------------------------------------')
    for i, file in enumerate(files):
        print('Evaluating tile (', i + 1, 'of', np.size(files), ') :', file[0:26])
        file = file[0:26]
        img, _ = load_product(file, params, test_data_path)
        # NOTE: Needs the masks
        print('---')


def write_csv_files(evaluation_metrics, params):
    if 'Biome' in params.train_dataset and 'Biome' in params.test_dataset:
        file_name = 'param_optimization_BiomeTrain_BiomeEval.csv'
    elif 'SPARCS' in params.train_dataset and 'SPARCS' in params.test_dataset:
        file_name = 'param_optimization_SPARCSTrain_SPARCSEval.csv'
    elif 'Biome' in params.train_dataset and 'SPARCS' in params.test_dataset:
        file_name = 'param_optimization_BiomeTrain_SPARCSEval.csv'
    elif 'SPARCS' in params.train_dataset and 'Biome' in params.test_dataset:
        file_name = 'param_optimization_SPARCSTrain_BiomeEval.csv'

    if 'fmask' in params.train_dataset:
        file_name = file_name[:-4] + '_fmask.csv'

    # Create csv file
    if not os.path.isfile(params.project_path + 'reports/Unet/' + file_name):
        f = open(params.project_path + 'reports/Unet/' + file_name, 'a')

        # Create headers for parameters
        string = 'modelID,'
        for key in params.values().keys():
            if key == 'modelID':
                pass
            elif key == 'test_tiles':
                pass
            else:
                string += key + ','

        # Create headers for evaluation metrics
        for i, product in enumerate(list(evaluation_metrics)):  # Lists all product names
            # Add product name to string
            string += 'tile_' + str(i) + ','

            # Only need examples from one threshold key (each threshold is a new line in the final csv file)
            threshold_example_key = list(evaluation_metrics[product])[0]
            for key in list(evaluation_metrics[product][threshold_example_key]):
                string += key + '_' + str(i) + ','

        # Create headers for averaged metrics
        f.write(string + 'mean_accuracy,mean_precision,mean_recall,mean_f_one_score,mean_omission,mean_comission,mean_pixel_jaccard\n')
        f.close()

    # Write a new line for each threshold value
    for threshold in list(evaluation_metrics[list(evaluation_metrics)[0]]):  # Use first product to list thresholds
        # Update the params threshold value before writing
        params.threshold = str(threshold[-3:])

        # Write params values
        f = open(params.project_path + 'reports/Unet/' + file_name, 'a')
        string = str(params.modelID) + ','
        for key in params.values().keys():
            if key == 'modelID':
                pass
            elif key == 'test_tiles':
                pass
            elif key == 'cls':
                string += ("".join(str(c) for c in params.values()[key])) + ','
            elif key == 'bands':
                string += ("".join(str(b) for b in params.values()[key])) + ','
            else:
                string += str(params.values()[key]) + ','

        # Initialize variables for calculating mean visualization set values
        accuracy_sum = precision_sum = recall_sum = f_one_score_sum = omission_sum = comission_sum = pixel_jaccard_sum=0

        # Write visualization set values
        for product in list(evaluation_metrics):
            # Add product name to string
            string += product + ','
            for key in (evaluation_metrics[product][threshold]):
                # Add values to string
                string += str(evaluation_metrics[product][threshold][key]) + ','

                # Extract values for calculating mean values of entire visualization set
                if 'accuracy' in key:
                    accuracy_sum += evaluation_metrics[product][threshold][key]
                elif 'precision' in key:
                    precision_sum += evaluation_metrics[product][threshold][key]
                elif 'recall' in key:
                    recall_sum += evaluation_metrics[product][threshold][key]
                elif 'f_one_score' in key:
                    f_one_score_sum += evaluation_metrics[product][threshold][key]
                elif 'omission' in key:
                    omission_sum += evaluation_metrics[product][threshold][key]
                elif 'comission' in key:
                    comission_sum += evaluation_metrics[product][threshold][key]
                elif 'pixel_jaccard' in key:
                    pixel_jaccard_sum += evaluation_metrics[product][threshold][key]

        # Add mean values to string
        n_products = np.size(list(evaluation_metrics))
        string += str(accuracy_sum / n_products) + ',' + str(precision_sum / n_products) + ',' + \
                  str(recall_sum / n_products) + ',' + str(f_one_score_sum / n_products) + ',' + \
                  str(omission_sum / n_products) + ',' + str(comission_sum / n_products) + ',' + \
                  str(pixel_jaccard_sum / n_products)

        # Write string and close csv file
        f.write(string + '\n')
        f.close()