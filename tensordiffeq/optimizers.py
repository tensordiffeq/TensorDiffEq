#from https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993

import numpy
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot
import time

def _graph_lbfgs(model, loss):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        loss [in]: a function with signature loss_value = loss(pred_y, true_y).
    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """

    # obtain the shapes of all trainable parameters in the model
    _shapes = tf.shape_n(model.trainable_variables)
    _n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    _count = 0
    _idx = [] # stitch indices
    _part = [] # partition indices
    _start_time = time.time()

    for i, shape in enumerate(_shapes):
        _n = numpy.product(_shape)
        idx.append(tf.reshape(tf.range(_count, _count+_n, dtype=tf.int32), _shape))
        part.extend([i]*_n)
        _count += _n

    _part = tf.constant(_part)

    @tf.function
    def _assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.
        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        _params = tf.dynamic_partition(_params_1d, _part, _n_tensors)
        for i, (_shape, _param) in enumerate(zip(_shapes, _params)):
            model.trainable_variables[i].assign(tf.reshape(_param, _shape))

    # now create a function that will be returned by this factory
    @tf.function
    def f(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        This function is created by function_factory.
        Args:
           params_1d [in]: a 1D tf.Tensor.
        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """
        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as tape:
            # update the parameters in the model
            _assign_new_model_parameters(params_1d)
            # calculate the loss
            _loss_value = loss()[0]

        # calculate gradients and convert to 1D tf.Tensor
        _grads = tape.gradient(_loss_value, model.trainable_variables)
        _grads = tf.dynamic_stitch(_idx, _grads)

        # print out iteration & loss
        f.iter.assign_add(1)

        if f.iter % 300 == 0:

            _elapsed = tf.timestamp() - f.start_time

            tf.print("Iter:", f.iter // 3, "loss:", _loss_value, "time:", _elapsed)
            f.start_time.assign(tf.timestamp())

        # store loss value so we can retrieve later
        tf.py_function(f.history.append, inp=[_loss_value], Tout=[])

        return _loss_value, _grads

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []
    f.start_time = tf.Variable(tf.timestamp())


    return f
