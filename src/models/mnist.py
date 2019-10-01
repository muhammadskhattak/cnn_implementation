import numpy as np


def load_mnist(prefix, folder):
    int_type = np.dtype('int32').newbyteorder('>')
    n_meta_data_bytes = 4 * int_type.itemsize

    data = np.fromfile( folder + "/" + prefix + '-images-idx3-ubyte', dtype='ubyte')
    magic_bytes, n_images, width, height = np.frombuffer(data[:n_meta_data_bytes].tobytes(), int_type)
    data = data[n_meta_data_bytes:].astype(dtype='float32').reshape([n_images, width, height])

    labels = np.fromfile( folder + "/" + prefix + '-labels-idx1-ubyte',
                          dtype='ubyte')[2 * int_type.itemsize:]

    return data, labels

