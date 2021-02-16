# from https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993

import numpy
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot
import time


def graph_lbfgs(model, loss):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        loss [in]: a function with signature loss_value = loss(pred_y, true_y).
    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = []  # stitch indices
    part = []  # partition indices
    start_time = time.time()

    for i, shape in enumerate(shapes):
        n = numpy.product(shape)
        idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
        part.extend([i] * n)
        count += n

    part = tf.constant(part)

    @tf.function
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.
        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

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
            assign_new_model_parameters(params_1d)
            # calculate the loss
            loss_value = loss()

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)

        # print out iteration & loss
        f.iter.assign_add(1)

        if f.iter % 300 == 0:
            elapsed = tf.timestamp() - f.start_time

            tf.print("Iter:", f.iter // 3, "loss:", loss_value, "time:", elapsed)
            f.start_time.assign(tf.timestamp())

        # store loss value so we can retrieve later
        tf.py_function(f.history.append, inp=[loss_value], Tout=[])

        return loss_value, grads

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []
    f.start_time = tf.Variable(tf.timestamp())

    return f


def dot(a, b):
    """Dot product function since TensorFlow doesn't have one."""
    return tf.reduce_sum(a * b)


def verbose_func(s):
    print(s)


def eager_lbfgs(opfunc, x, state, maxIter=100, learningRate=1, do_verbose=True):
    """port of lbfgs.lua, using TensorFlow eager mode.
  """

    global final_loss, times

    maxEval = maxIter * 1.25
    tolFun = 1e-12
    tolX = 1e-12
    nCorrection = 50
    isverbose = False
    state.start_time = time.time()

    # verbose function
    if isverbose:
        verbose = verbose_func
    else:
        verbose = lambda x: None

    f, g = opfunc(x)
    g_old = g
    f_old = f

    f_hist = [f]
    currentFuncEval = 1
    state.funcEval = state.funcEval + 1
    p = g.shape[0]

    # check optimality of initial point
    tmp1 = tf.abs(g)
    if tf.reduce_sum(tmp1) <= tolFun:
        verbose("optimality condition below tolFun")
        return x, f_hist

    # optimize for a max of maxIter iterations
    nIter = 0
    times = []
    while nIter < maxIter:
        start_time = time.time()
        if state.nIter == 1:
            tmp1 = tf.abs(g)
            t = min(1, 1 / tf.reduce_sum(tmp1))
        else:
            t = learningRate
        # keep track of nb of iterations
        nIter = nIter + 1
        state.nIter = state.nIter + 1

        ############################################################
        ## compute gradient descent direction
        ############################################################
        if state.nIter == 1:
            d = -g
            old_dirs = []
            old_stps = []
            Hdiag = 1
        else:
            # do lbfgs update (update memory)
            y = g - g_old
            s = d * t
            ys = dot(y, s)

            if ys > 1e-10:
                # updating memory
                if len(old_dirs) == nCorrection:
                    # shift history by one (limited-memory)
                    del old_dirs[0]
                    del old_stps[0]

                # store new direction/step
                old_dirs.append(s)
                old_stps.append(y)

                # update scale of initial Hessian approximation
                Hdiag = ys / dot(y, y)

            # compute the approximate (L-BFGS) inverse Hessian
            # multiplied by the gradient
            k = len(old_dirs)

            # need to be accessed element-by-element, so don't re-type tensor:
            ro = [0] * nCorrection
            for i in range(k):
                ro[i] = 1 / dot(old_stps[i], old_dirs[i])

            # iteration in L-BFGS loop collapsed to use just one buffer
            # need to be accessed element-by-element, so don't re-type tensor:
            al = [0] * nCorrection

            q = -g
            for i in range(k - 1, -1, -1):
                al[i] = dot(old_dirs[i], q) * ro[i]
                q = q - al[i] * old_stps[i]

            # multiply by initial Hessian
            r = q * Hdiag
            for i in range(k):
                be_i = dot(old_stps[i], r) * ro[i]
                r += (al[i] - be_i) * old_dirs[i]

            d = r
            # final direction is in r/d (same object)

        g_old = g
        f_old = f

        ############################################################
        ## compute step length
        ############################################################
        # directional derivative
        gtd = dot(g, d)

        # check that progress can be made along that direction
        if gtd > -tolX:
            verbose("Can not make progress along direction.")
            break

        # reset initial guess for step size
        if state.nIter == 1:
            tmp1 = tf.abs(g)
            t = min(1, 1 / tf.reduce_sum(tmp1))
        else:
            t = learningRate

        x += t * d

        if nIter != maxIter:
            # re-evaluate function only if not in last iteration
            # the reason we do this: in a stochastic setting,
            # no use to re-evaluate that function here
            f, g = opfunc(x)

        lsFuncEval = 1
        f_hist.append(f)

        # update func eval
        currentFuncEval = currentFuncEval + lsFuncEval
        state.funcEval = state.funcEval + lsFuncEval

        ############################################################
        ## check conditions
        ############################################################
        if nIter == maxIter:
            break

        if currentFuncEval >= maxEval:
            # max nb of function evals
            print('max nb of function evals')
            break

        tmp1 = tf.abs(g)
        if tf.reduce_sum(tmp1) <= tolFun:
            # check optimality
            print('optimality condition below tolFun')
            break

        tmp1 = tf.abs(d * t)
        if tf.reduce_sum(tmp1) <= tolX:
            # step size below tolX
            print('step size below tolX')
            break

        if tf.abs(f, f_old) < tolX:
            # function value changing less than tolX
            print('function value changing less than tolX' + str(tf.abs(f - f_old)))
            break

        if do_verbose:
            if nIter % 100 == 0:
                elapsed = time.time() - state.start_time
                print("Step: %3d, loss: %9.8f, time: " % (nIter, f.numpy()), elapsed)
                state.start_time = time.time()

        if nIter == maxIter - 1:
            final_loss = f.numpy()

    # save state
    state.old_dirs = old_dirs
    state.old_stps = old_stps
    state.Hdiag = Hdiag
    state.g_old = g_old
    state.f_old = f_old
    state.t = t
    state.d = d

    return x, f_hist, currentFuncEval


# dummy/Struct gives Lua-like struct object with 0 defaults
class dummy(object):
    pass


class Struct(dummy):
    def __getattribute__(self, key):
        if key == '__dict__':
            return super(dummy, self).__getattribute__('__dict__')
        return self.__dict__.get(key, 0)
