def create_generators(cls, params):
    # Instantiate the generators
    seed = 300  # Remember to use same seed for img and mask generators!
    trn_img_generator = __img_generator__(type="train/img",
                                          path=params.project_path + 'data/processed/',
                                          shuffle=True,
                                          seed=seed,
                                          params=params)
    trn_mask_generator = __mask_generator__(type="train/mask",
                                            path=params.project_path + 'data/processed/',
                                            cls=cls,
                                            shuffle=True,
                                            seed=seed,
                                            params=params)
    val_img_generator = __img_generator__(type="val/img",
                                          path=params.project_path + 'data/processed/',
                                          shuffle=True,
                                          seed=seed,
                                          params=params)
    val_mask_generator = __mask_generator__(type="val/mask",
                                            path=params.project_path + 'data/processed/',
                                            cls=cls,
                                            shuffle=True,
                                            seed=seed,
                                            params=params)

    # Combine the generators to be used with model.fit_generator (see https://keras.io/preprocessing/image/)
    train_generator = zip(trn_img_generator, trn_mask_generator)
    val_generator = zip(val_img_generator, val_mask_generator)
    return train_generator, val_generator


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):  # Py3
        return next(self.it)

        # def next(self):     # Py2
        #    with self.lock:
        #        return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


@threadsafe_generator
def __img_generator__(type, path, shuffle, seed, params):
    """
    Numpy generator for the training data
    """
    # Load config
    bands = params.bands
    batch_size = params.batch_size

    # Load the names of the numpy files, each containing one patch
    files = sorted(os.listdir(path + type))  # os.listdir loads in arbitrary order, hence use sorted()
    files = [f for f in files if '.npy' in f]  # Only use .npy files (e.g. avoid .gitkeep)

    # Create a placeholder for x which can hold batch_size patches
    patch_shape = np.shape(np.load(path + type + "/" + files[0]))
    x_all_bands = np.zeros((batch_size, patch_shape[0], patch_shape[1], patch_shape[2]))
    x = np.zeros((batch_size, patch_shape[0], patch_shape[1], np.size(bands)))

    # Create random generator used for shuffling files and data augmentation
    random_img = random.Random()
    random_img.seed(seed)

    # Shuffle the patches
    if shuffle:
        random_img.shuffle(files)

    # Create the generator
    index = 0
    max_index = len(files)  # max_index is the number of patches available
    while 1:
        for i in range(batch_size):
            if index >= max_index:
                index = 0

            # Load all bands
            x_all_bands[i, :, :, :] = np.load(path + type + "/" + files[index])

            # Extract the wanted bands
            for j, b in enumerate(bands):
                if b == 1:
                    x[i, :, :, j] = x_all_bands[i, :, :, 0]
                elif b == 2:
                    x[i, :, :, j] = x_all_bands[i, :, :, 1]
                elif b == 3:
                    x[i, :, :, j] = x_all_bands[i, :, :, 2]
                elif b == 4:
                    x[i, :, :, j] = x_all_bands[i, :, :, 3]
                elif b == 5:
                    x[i, :, :, j] = x_all_bands[i, :, :, 4]
                elif b == 6:
                    x[i, :, :, j] = x_all_bands[i, :, :, 5]
                elif b == 7:
                    x[i, :, :, j] = x_all_bands[i, :, :, 6]
                elif b == 8:
                    raise ValueError('Band 8 (pan-chromatic band) cannot be included')
                elif b == 9:
                    x[i, :, :, j] = x_all_bands[i, :, :, 7]
                elif b == 10:
                    x[i, :, :, j] = x_all_bands[i, :, :, 8]
                elif b == 11:
                    x[i, :, :, j] = x_all_bands[i, :, :, 9]

            # Brightness augmentation
            if params.brightness_augmentation:
                # Only do augmentation on the training images (i.e. not on val images)
                '''
                if type == "train/img":
                    # Stretch image from 0 to 1 and then multiply by a random number
                    # x[i, :, :, :] = x[i, :, :, :] / np.max(x[i, :, :, :]) * np.random.uniform(low=0.1, high=1)

                    # Change threshold paramater to do something similar to brightness augmentation
                    scaling_parameter = np.random.uniform(low=0.8, high=1.5)
                    x[i, :, :, :] = image_normalizer(x[i, :, :, :],
                                                     threshold=params.norm_threshold * scaling_parameter)

                else:
                    # Normalize
                    x[i, :, :, :] = image_normalizer(x[i, :, :, :], threshold=params.norm_threshold)

                '''

            else:
                # Normalize
                x[i, :, :, :] = image_normalizer(x[i, :, :, :], params, params.norm_method)

            # Data augmentation
            # Todo: Should only be done on training images (and should be selected in params)
            if True:
                if random_img.randint(0, 1):
                    np.flip(x, axis=1)

                if random_img.randint(0, 1):
                    np.flip(x, axis=2)

            index = index + i
        print('\n')
        print(files[index])

        yield x


@threadsafe_generator
def __mask_generator__(type, path, cls, shuffle, seed, params):
    """
    Numpy generator for the true training labels
    """
    # Load config
    batch_size = params.batch_size
    collapse_cls = params.collapse_cls

    # Load the names of the numpy files, each containing one patch
    files = sorted(os.listdir(path + type))  # os.listdir loads in arbitrary order, hence use sorted()
    files = [f for f in files if '.npy' in f]  # Only use .npy files (e.g. avoid .gitkeep)

    if params.satellite == 'Sentinel-2':
        if params.initial_model == 'sen2cor':
            files = [f for f in files if 'sen2cor' in f]
        elif params.initial_model == 'fmask':
            files = [f for f in files if 'fmask' in f]

    # Create a placeholder for x which can hold batch_size patches
    patch_shape = np.shape(np.load(path + type + "/" + files[0]))
    if params.collapse_cls:  # Find the number of classes
        n_cls = 1
    else:
        n_cls = np.size(cls)
    x = np.zeros((batch_size, patch_shape[0], patch_shape[1], n_cls))
    clip_pixels = np.int32(params.overlap / 2)
    x_cropped = np.zeros((batch_size, patch_shape[0] - 2 * clip_pixels, patch_shape[1] - 2 * clip_pixels, n_cls))

    # Create random generator used for shuffling files and data augmentation
    random_mask = random.Random()
    random_mask.seed(seed)

    # Shuffle the patches
    if shuffle:
        random_mask.shuffle(files)

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
                mask = extract_collapsed_cls(mask, cls)

                # Save the binary mask
                x[i, :, :, :] = mask
                x_cropped[i, :, :, :] = x[i,
                                        clip_pixels:patch_shape[0] - clip_pixels,
                                        clip_pixels:patch_shape[1] - clip_pixels,
                                        :]

            else:
                # Get both the class and the iteration index in the for loop using enumerate()
                # https://stackoverflow.com/questions/522563/accessing-the-index-in-python-for-loops
                for j, c in enumerate(cls):
                    y = extract_cls_mask(mask, c)

                    # Save the binary masks as one hot representations
                    x[i, :, :, j] = y[:, :, 0]

                    x_cropped[i, :, :, j] = x[i,
                                            clip_pixels:patch_shape[0] - clip_pixels,
                                            clip_pixels:patch_shape[1] - clip_pixels,
                                            j]

            if True:
                if random_mask.randint(0, 1):
                    np.flip(x_cropped, axis=1)

                if random_mask.randint(0, 1):
                    np.flip(x_cropped, axis=2)

            index = index + i
        print(files[index])
        yield x_cropped