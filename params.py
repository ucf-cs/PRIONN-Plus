# params.py
import torch.nn as nn

CPU_COUNT = 24

# VECTOR_SIZE = 4 # params.config['w2v_vec_size']
# BATCH_SIZE = 32 # params.config['cnn_batch_size']
# SCRIPT_LEN = 50 # params.config['script_len']

# TODO: These layer sizes were pulled out of thin air because PRIONN doesn't provide that level of detail. Make sure the sizes are appropriate for the task at hand.
# config = {
#     'cnn_batch_size': 32,
#     'cnn_cl1': 256,
#     'cnn_cl2': 512,
#     'cnn_cl3': 1024,
#     'cnn_epochs': 25,
#     'cnn_f1': 512,
#     'cnn_f2': 256,
#     'cnn_f3': 128,
#     'cnn_lr': 0.007,
#     'output_size': 1440
#     'script_len': 50,
#     'w2v_epochs': 10,
#     'w2v_vec_size': 4,
#     'w2v_window': 2,
# }
config = {
    'cnn_batch_size': 1,
    'cnn_cl1': 128,
    'cnn_cl2': 32,
    'cnn_cl3': 16,
    'cnn_epochs': 2,
    'cnn_f1': 4,
    'cnn_f2': 128,
    'cnn_f3': 128,
    'cnn_lr': 0.00020814735705421613,
    'output_size': 1440,
    'script_len': 10,
    'w2v_epochs': 89,
    'w2v_vec_size': 3,
    'w2v_window': 6,
    }

def print_params(config=config):
    for (key, val) in config:
        print(str(key)+': '+str(val))
    return
