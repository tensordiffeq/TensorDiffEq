import tensorflow as tf
from tensordiffeq.sampling import LHS
import time as time
import numpy as np


def set_weights(model, w, sizes_w, sizes_b):
    for i, layer in enumerate(model.layers[0:]):
        start_weights = sum(sizes_w[:i]) + sum(sizes_b[:i])
        end_weights = sum(sizes_w[:i + 1]) + sum(sizes_b[:i])
        weights = w[start_weights:end_weights]
        w_div = int(sizes_w[i] / sizes_b[i])
        weights = tf.reshape(weights, [w_div, sizes_b[i]])
        biases = w[end_weights:end_weights + sizes_b[i]]
        weights_biases = [weights, biases]
        layer.set_weights(weights_biases)


def get_weights(model):
    w = []
    for layer in model.layers[0:]:
        weights_biases = layer.get_weights()
        weights = weights_biases[0].flatten()
        biases = weights_biases[1]
        w.extend(weights)
        w.extend(biases)

    w = tf.convert_to_tensor(w)
    return w


def get_sizes(layer_sizes):
    sizes_w = []
    sizes_b = []
    for i, width in enumerate(layer_sizes):
        if i != 1:
            sizes_w.append(int(width * layer_sizes[1]))
            sizes_b.append(int(width if i != 0 else layer_sizes[1]))
    return sizes_w, sizes_b


def MSE(pred, actual, weights=None):
    if weights is not None:
        return tf.reduce_mean(tf.square(weights * tf.math.subtract(pred, actual)))
    return tf.reduce_mean(tf.square(tf.math.subtract(pred, actual)))


def g_MSE(pred, actual, g_lam):
    return tf.reduce_mean(g_lam * tf.square(tf.math.subtract(pred, actual)))


def constant(val, dtype=tf.float32):
    return tf.constant(val, dtype=dtype)


def convertTensor(val, dtype=tf.float32):
    return tf.cast(val, dtype=dtype)


def LatinHypercubeSample(N_f, bounds):
    sampling = LHS(xlimits=bounds)
    return sampling(N_f)


def get_tf_model(model):
    return tf.function(model)


def tensor(x, dtype=tf.float32):
    return tf.convert_to_tensor(x, dtype=dtype)


def multimesh(arrs):
    lens = list(map(len, arrs))
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz *= s

    ans = []
    for i, arr in enumerate(arrs):
        slc = [1] * dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j != i:
                arr2 = arr2.repeat(sz, axis=j)
        ans.append(arr2)

    return ans  # returns like np.meshgrid


# if desired, this flattens and hstacks the output dimensions for feeding into a tf/keras type neural network
def flatten_and_stack(mesh):
    dims = np.shape(mesh)
    output = np.zeros((len(mesh), np.prod(dims[1:])))
    for i, arr in enumerate(mesh):
        output[i] = arr.flatten()
    return output.T  # returns in an [nxm] matrix


final_loss = None
times = []
