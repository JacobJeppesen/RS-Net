#!/usr/bin/env python
import tensorflow as tf


def get_hparams(model):
    if model == 'U-net':
        return tf.contrib.training.HParams(learning_rate=0.001,
                                           decay=0.0,
                                           threshold=0.5,  # Threshold to create binary cloud mask
                                           patch_size=320,  # Width and height of the patches the img is divided into
                                           overlap=40,  # Overlap in pixels when predicting (to avoid border effects)
                                           batch_size=32,
                                           steps_per_epoch=50,  # = batches per epoch
                                           epochs=4,
                                           data_path="/mnt/dfs/ST_FC-WP1/Data/SentinelSemanticSegmentation/data/")

