#!/usr/bin/env python
import numpy as np
import datetime
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils.training_utils import multi_gpu_model
from ..utils import patch_image, jaccard_coef, jaccard_coef_int, calc_jacc, numpy_generator


class Unet(object):
    def _create_inference(self, n_bands, n_cls, hparams):
        inputs = Input((hparams.patch_size, hparams.patch_size, n_bands))
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv5), conv4])
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv3])
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv2])
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv8), conv1])
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(n_cls, (1, 1), activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=conv10)

        return model

    def train(self, n_bands, n_cls, num_gpus, hparams):
        use_own_generator = True

        if not use_own_generator:
            # Load training data
            x_trn = np.load('data/processed/x_trn_%d.npy' % n_cls)
            y_trn = np.load('data/processed/y_trn_%d.npy' % n_cls)

            # Load validation data
            x_val = np.load('data/processed/x_tmp_%d.npy' % n_cls)
            y_val = np.load('data/processed/y_tmp_%d.npy' % n_cls)

            # Load training data
            #x_trn = np.load('/home/jhj/phd/NetworkDrives/dfs/ST_FC-WP1/Data/SentinelSemanticSegmentation/data/processed/x_trn_%d.npy' % n_cls)
            #y_trn = np.load('/home/jhj/phd/NetworkDrives/dfs/ST_FC-WP1/Data/SentinelSemanticSegmentation/data/processed/y_trn_%d.npy' % n_cls)

            # Load validation data
            #x_val = np.load('/home/jhj/phd/NetworkDrives/dfs/ST_FC-WP1/Data/SentinelSemanticSegmentation/data/processed/x_tmp_%d.npy' % n_cls)
            #y_val = np.load('/home/jhj/phd/NetworkDrives/dfs/ST_FC-WP1/Data/SentinelSemanticSegmentation/data/processed/y_tmp_%d.npy' % n_cls)

            # Divide data into patches
            #x_trn, _, _ = patch_image(x_trn, hparams.patch_size, overlap=0)
            y_trn, _, _ = patch_image(y_trn, hparams.patch_size, overlap=0)
            x_val, _, _ = patch_image(x_val, hparams.patch_size, overlap=0)
            y_val, _, _ = patch_image(y_val, hparams.patch_size, overlap=0)

            # IMPORTANT: SEE https://github.com/fchollet/keras/blob/master/examples/mnist_tfrecord.py FOR TFRECORDS and
            # https://keunwoochoi.wordpress.com/2017/08/24/tip-fit_generator-in-keras-how-to-parallelise-correctly/ and
            # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

            # we create two generator objects with the same arguments
            data_gen_args = dict(horizontal_flip=True,
                                 vertical_flip=True,
                                 data_format="channels_last"
                                 )

            # NOTE: CUSTOM DATA GENERATORS SHOULD BE MADE TO DEAL WITH MORE THAN 3 BANDS. ALSO, USE FLOW_FROM_DIRECTORY.
            #       MAYBE USE TFRECORDS INSTEAD.
            image_datagen = ImageDataGenerator(**data_gen_args)
            mask_datagen = ImageDataGenerator(**data_gen_args)

            # Provide the same seed (important!)
            # For more info, see bottom of https://keras.io/preprocessing/image/
            seed = 1

            # Do statistics to be used in datagen.flow() for pre-processing
            #image_datagen.fit(x_trn, augment=False, seed=seed)
            #mask_datagen.fit(y_trn, augment=False, seed=seed)

            # Create generators
            image_generator = image_datagen.flow(x_trn, batch_size=hparams.batch_size, seed=seed)
            mask_generator = mask_datagen.flow(y_trn, batch_size=hparams.batch_size, seed=seed)

            # Combine generators into one which yields image and masks
            train_generator = zip(image_generator, mask_generator)

        if use_own_generator:
            seed = 1  # Remember to use same seed for img and mask generators!
            img_generator = numpy_generator(type="img",
                                            path=hparams.data_path,
                                            batch_size=hparams.batch_size,
                                            shuffle=True,
                                            seed=seed)

            mask_generator = numpy_generator(type="mask",
                                             path=hparams.data_path,
                                             batch_size=hparams.batch_size,
                                             shuffle=True,
                                             seed=seed)

            train_generator = zip(img_generator, mask_generator)

        # Create the model in keras
        if num_gpus == 1:
            model = self._create_inference(n_bands, n_cls, hparams)  # initialize the model on the CPU

        else:
            with tf.device("/cpu:0"):
                model = self._create_inference(n_bands, n_cls, hparams)  # initialize the model on the CPU
            model = multi_gpu_model(model, gpus=num_gpus)  # Make it run on multiple GPUs

        model.compile(optimizer=Adam(lr=hparams.learning_rate), loss='binary_crossentropy',
                      metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])

        # Define callbacks
        # Must use save_weights_only=True in model_checkpoint (BUG: https://github.com/fchollet/keras/issues/8123)
        model_checkpoint = ModelCheckpoint('models/Unet/weights/unet_tmp.hdf5',
                                           save_weights_only=True,
                                           save_best_only=True)

        tensorboard = TensorBoard(log_dir="models/Unet/tensorboard/{}".
                                  format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
                                  write_graph=True)

        # Load most recent weights
        try:
            model.load_weights('models/Unet/weights/unet_tmp.hdf5')
            print('------------------------------------------')
            print('Loaded weights from most recent checkpoint')
        except:
            print('------------------------------------------')
            print('No saved weights loaded')
            pass

        # Train the model
        print('------------------------------------------')
        print('Start training:')
        model.fit_generator(train_generator,
                            steps_per_epoch=hparams.steps_per_epoch,
                            epochs=hparams.epochs,
                            verbose=1,
                            callbacks=[model_checkpoint, tensorboard])
                            #validation_data=(x_val, y_val))

        # Run model on validation data set
        print('------------------------------------------')
        print("Evaluate model on visualization data set:")
        score = calc_jacc(model,
                          n_cls,
                          patch_size=hparams.patch_size,
                          overlap=hparams.overlap,
                          threshold=hparams.threshold)
        print('Jaccard index (visualization set):', score)

        # Save the weights (append the val score in the name)
        # There is a bug with multi_gpu_model (https://github.com/kuza55/keras-extras/issues/3), hence model.layers[-2]
        if num_gpus != 1:
            model = model.layers[-2]
        model.save_weights('models/Unet/weights/unet_10_jk%.4f' % score)
        model.save_weights('models/Unet/weights/unet_recent.hdf5')

        return model

    def predict(self, img, n_bands, n_cls, num_gpus, hparams):
        # Create the model in keras
        if num_gpus == 1:
            model = self._create_inference(n_bands, n_cls, hparams)  # initialize the model on the CPU
            model.load_weights('models/Unet/weights/unet_recent.hdf5')
        else:
            with tf.device("/cpu:0"):
                model = self._create_inference(n_bands, n_cls, hparams)  # initialize the model on the CPU
                model.load_weights('models/Unet/weights/unet_recent.hdf5')
            model = multi_gpu_model(model, gpus=num_gpus)  # Make it run on multiple GPUs

        # Predict batches of patches
        patches = np.shape(img)[0]  # Total number of patches
        patch_batch_size = 100

        # Do the prediction
        predicted = np.zeros((patches, hparams.patch_size, hparams.patch_size, n_cls))
        for i in range(0, patches, patch_batch_size):
            #print('Processing patch nr.:', i, 'of', patches)
            predicted[i:i + patch_batch_size, :, :, :] = model.predict(img[i:i + patch_batch_size, :, :, :])

        return predicted

