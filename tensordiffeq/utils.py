
import tensorflow as tf

def set_weights(self, model, w, sizes_w, sizes_b):
        for i, layer in enumerate(model.layers[0:]):
            start_weights = sum(sizes_w[:i]) + sum(sizes_b[:i])
            end_weights = sum(sizes_w[:i+1]) + sum(sizes_b[:i])
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


def MSE(pred, actual, weights = None):
    if weights == None:
        return tf.reduce_mean(tf.square(tf.math.subtract(pred,actual)))
    return tf.reduce_mean(tf.square(self.weights*tf.math.subtract(pred,actual)))

def constant(val, dtype = tf.float32):
    return tf.constant(val, dtype= dtype)

def LatinHypercubeSample(N_f, bounds):
    sampling = LHS(xlimits=bounds)
    return sampling(N_f)

def _get_tf_model(model):
    #tf.function
    #model
    return tf.function(model)
