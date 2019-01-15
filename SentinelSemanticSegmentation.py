#!/usr/bin/env python
"""
This file is used to run the project. Set all to true to run full pipeline.

Notes:
- The structure of this file (and the entire project in general) is made with emphasis on flexibility for research
purposes, and the pipelining is done in a python file such that newcomers can easily use and understand the code.

- Remember that relative paths in Python are always relative to the current working directory.
Hence, if you look at the functions in make_dataset.py, the file paths are relative to the path of
this file (SentinelSemanticSegmentation.py)
"""

__author__ = "Jacob Høxbroe Jeppesen"
__email__ = "jhj@eng.au.dk"

import time
import argparse
import datetime
import os
import random
import numpy as np
import tensorflow as tf
from src.data.make_dataset import make_numpy_dataset
from src.models.params import get_params
from src.models.Unet import Unet
from src.models.evaluate_model import evaluate_test_set, write_csv_files

# Don't allow tensorflow to reserve all memory available
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
#sess = tf.Session(config=config)  # set this TensorFlow session as the default session for Keras
#set_session(sess)

# ----------------------------------------------------------------------------------------------------------------------
# Define default pipeline
# ----------------------------------------------------------------------------------------------------------------------
# Create the parser. The formatter_class argument makes sure that default values are shown when --help is called.
parser = argparse.ArgumentParser(description='Pipeline for running the project',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Define which steps should be run automatically when this file is run. When using action='store_true', the argument
# has to be provided to run the step. When using action='store_false', the step will be run when this file is executed.
parser.add_argument('--make_dataset',
                    action='store_true',
                    help='Run the pre-processing step')

parser.add_argument('--train',
                    action='store_true',
                    help='Run the training step')

parser.add_argument('--hparam_optimization',
                    action='store_true',
                    help='Do hyperparameter optimization')

parser.add_argument('--test',
                    action='store_true',
                    help='Run test step')


# ----------------------------------------------------------------------------------------------------------------------
# Define the arguments used in the entire pipeline
# ----------------------------------------------------------------------------------------------------------------------
parser.add_argument('--satellite',
                    type=str,
                    default='Landsat8',
                    help='The satellite used (Sentinel-2 or Landsat8)')

parser.add_argument('--initial_model',
                    type=str,
                    default='sen2cor',
                    help='Which initial is model is wanted for training (sen2cor or fmask)')


# ----------------------------------------------------------------------------------------------------------------------
# Define the arguments for the training
# ----------------------------------------------------------------------------------------------------------------------
parser.add_argument('--model',
                    type=str,
                    default='U-net',
                    help='Comma separated list of "name=value" pairs.')

parser.add_argument('--params',
                    type=str,
                    help='Comma separated list of "name=value" pairs.')

parser.add_argument('--dev_dataset',
                    action='store_true',
                    help='Very small dataset to be used while developing the project')


# ----------------------------------------------------------------------------------------------------------------------
# Define the arguments for the visualization
# ----------------------------------------------------------------------------------------------------------------------
parser.add_argument('--dataset',
                    type=str,
                    default='Biome',
                    help='Dataset for evaluating Landsat 8 data')


if __name__ == '__main__':
    # Load the arguments
    args = parser.parse_args()

    # Store current time to calculate execution time later
    start_time = time.time()

    print("\n---------------------------------------")
    print("Script started")
    print("---------------------------------------\n")

    # Load hyperparameters into the params object containing name-value pairs
    params = get_params(args.model, args.satellite)

    # If any hyperparameters were overwritten in the commandline, parse them into params
    if args.params:
        params.parse(args.params)

    # If you want to use local files (else it uses network drive)
    if args.dev_dataset:
        params.data_path = "/home/jhj/phd/GitProjects/SentinelSemanticSegmentation/data/processed/dev_dataset/"

    # Check to see if a new data set should be processed from the raw data
    if args.make_dataset:
        print("Processing numpy data set")
        make_numpy_dataset(params)

    # Check to see if a model should be trained
    if args.train:
        print("Training " + args.model + " model")
        if not params.split_dataset:  # No k-fold cross-validation
            # Load the model
            params.modelID = datetime.datetime.now().strftime("%y%m%d%H%M%S")
            if args.model == 'U-net':
                model = Unet(params)

            model.train(params)
            # Run model on test data set
            evaluate_test_set(model, params.test_dataset, params.num_gpus, params)
        else:  # With k-fold cross-validation
            # Define number of k-folds
            if 'Biome' in params.train_dataset:
                k_folds = 2  # Biome dataset is made for 2-fold CV
            else:
                k_folds = 5  # SPARCS contains 80 scenes, so split it nicely

                # Create a list of names for the splitting
                sparcs_products = sorted(os.listdir(params.project_path + "data/raw/SPARCS_dataset/"))
                sparcs_products = [f for f in sparcs_products if 'data.tif' in f]
                sparcs_products = [f for f in sparcs_products if 'aux' not in f]

                # Randomize the list of SPARCS products
                seed = 1
                random.seed(seed)
                random.shuffle(sparcs_products)

            # Do the training/testing with k-fold cross-validation
            params.modelID = datetime.datetime.now().strftime("%y%m%d%H%M%S")
            for k in range(k_folds):
                # Define train and test tiles (note that params.test_tiles[0] are training and .test_tiles[1] are test)
                if 'SPARCS' in params.train_dataset:
                    products_per_fold = int(80/k_folds)
                    # Define products for test
                    params.test_tiles[1] = sparcs_products[k*products_per_fold:(k+1)*products_per_fold]
                    # Define products for train by loading all sparcs products and then removing test products
                    params.test_tiles[0] = sparcs_products
                    for product in params.test_tiles[1]:
                        params.test_tiles[0] = [f for f in params.test_tiles[0] if product not in f]

                elif 'Biome' in params.train_dataset:
                    # Swap train and test set for 2-fold CV
                    temp = params.test_tiles[0]
                    params.test_tiles[0] = params.test_tiles[1]
                    params.test_tiles[1] = temp

                # Train and evaluate
                params.modelID = params.modelID[0:12] + '-CV' + str(k+1) + 'of' + str(k_folds)  # Used for saving results
                model = Unet(params)
                print("Training on fold " + str(k + 1) + " of " + str(k_folds))
                model.train(params)

                # Run model on test data set and save output
                evaluate_test_set(model, params.test_dataset, params.num_gpus, params)

    if args.test:
        # If a model has been trained, use that one. If not, load a new one.
        if args.model == 'U-net':
            model = Unet(params)
        evaluate_test_set(model, params.test_dataset, params.num_gpus, params)

    # Print execution time
    exec_time = str(time.time() - start_time)
    print("\n---------------------------------------")
    print("Script executed in: " + exec_time + "s")
    print("---------------------------------------")
