####################################################
# IMPORTANT: THIS SCRIPT MUST BE RUN FROM TERMIAL! #
####################################################
import subprocess
import numpy as np

activation_functions = ['elu']
loss_functions = ['binary_crossentropy']
initializers = ['glorot_normal']
learning_rates = [1e-4]
use_batch_norm = ['True']
l2regs = [1e-4]
dropouts = [0, 0.25, 0.5]
band_combinations = [[1, 2, 3, 4, 5, 6, 7, 9, 10, 11], [1, 2, 3, 4, 5, 6, 7, 9], [2, 3, 4, 5], [2, 3, 4], [3]]
epochs = [3, 10, 20, 40, 80, 160, 200, 200, 200, 200, 200, 200, 200, 200]  # Only used to run random search for longer
dropout_on_last_layer_only=['True']

# Train the models
for activation_func in activation_functions:
    for loss_func in loss_functions:
        for learning_rate in learning_rates:                  
            for l2reg in l2regs:
                for dropout in dropouts:
                    for dropout_on_only_last_layer in dropout_on_last_layer_only:
                        for use_bn in use_batch_norm:
                            for epoch_no in epochs:                                
                                for bands in band_combinations:
                                    for initializer in initializers:
                                        # Hacky way to do to random search by overwriting actual values
                                        learning_rate = int(np.random.uniform(10, 100, 1)[0]) * 1e-5  # Cast to int to avoid round() function (issues with floats)
                                        l2reg = int(np.random.uniform(1, 1000, 1)[0]) * 1e-5  # 1 to 100 when using ground truth, 1 to 1000 when using fmask
                                        dropout = int(np.random.uniform(0, 50, 1)[0]) * 1e-2 * np.random.randint(2, size=1)[0]  # 50% chance for dropout=0
                                        epoch_no = int(np.random.lognormal(3, 0.8, 1)[0])

                                        interpreter = "/home/jhj/anaconda3/envs/CloudDetection/bin/python"
                                        script = "/home/jhj/phd/GitProjects/SentinelSemanticSegmentation/SentinelSemanticSegmentation.py"
                                        # Params string format must fit with the HParams object
                                        # See more at https://www.tensorflow.org/api_docs/python/tf/contrib/training/HParams
                                        params = "--params=activation_func=" + activation_func + \
                                                ",loss_func=" + str(loss_func) + \
                                                ",learning_rate=" + str(learning_rate) + \
                                                ",L2reg=" + str(l2reg) + \
                                                ",dropout=" + str(dropout) + \
                                                ",bands=" + str(bands) + \
                                                ",epochs=" + str(epoch_no) + \
                                                ",use_batch_norm=" + str(use_bn) + \
                                                ",dropout_on_last_layer_only=" + str(dropout_on_only_last_layer) + \
                                                ",initialization=" + str(initializer)

                                        print('-------------STARTING NEW--------------------------')
                                        print('Activation function: ' + activation_func)
                                        print('loss_func: ' + str(loss_func))
                                        print('initializer: ' + str(initializer))
                                        print('learnings_rate: ' + str(learning_rate))
                                        print('L2reg: ' + str(l2reg))
                                        print('dropout: ' + str(dropout))
                                        print('bands: ' + str(bands))
                                        print('epochs: ' + str(epoch_no))
                                        print('use_batch_norm: ' + str(use_bn))
                                        print('dropout_on_last_layer_only: ' + str(dropout_on_only_last_layer))
                                        print('---------------------------------------------------')
                                        subprocess.check_call([interpreter,
                                                            script,
                                                            "--train",
                                                            params])



