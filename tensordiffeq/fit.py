import tensorflow as tf
import numpy as np
from .networks import *
from .models import *
from .utils import *
from .optimizers import *
import time
import os

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"



def fit(obj, tf_iter, newton_iter, batch_sz = None):
    obj.u_model = neural_net(obj.layer_sizes)
    #Can adjust batch size for collocation points, here we set it to N_f
    if batch_sz is not None:
        obj.batch_sz = batch_sz
    else:
        obj.batch_sz = len(obj.x_f)

    N_f = len(obj.x_f)
    n_batches =  N_f // obj.batch_sz

    start_time = time.time()
    obj.tf_optimizer = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.99)
    obj.tf_optimizer_weights = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.99)

    #these cant be tf.functions on initialization since the distributed strategy requires its own
    #graph using grad and adaptgrad, so they cant be compiled as tf.functions until we know dist/non-dist
    obj.grad = tf.function(obj.grad)
    print("starting Adam training")
    #tf.profiler.experimental.start('../cache/tblogdir1')
    for epoch in range(tf_iter):
        # if obj.isAdaptive:
        #     loss_value, mse_0, mse_b, mse_f, grads, grads_col, grads_u = train_op(obj, n_batches)
        # else:
        loss_value, mse_0, mse_b, mse_f = train_op(obj, n_batches)

        if epoch % 100 == 0:
            elapsed = time.time() - start_time
            print('It: %d, Time: %.2f' % (epoch, elapsed))
            tf.print(f"mse_0: {mse_0}  mse_b  {mse_b}  mse_f: {mse_f}   total loss: {loss_value}")
            start_time = time.time()
    #tf.profiler.experimental.stop()
    #l-bfgs-b optimization
    print("Starting L-BFGS training")
    #tf.profiler.experimental.start('../cache/tblogdir1')
    lbfgs_train(obj, newton_iter)
    #tf.profiler.experimental.stop()


#@tf.function
def lbfgs_train(obj, newton_iter):
    func = graph_lbfgs(obj.u_model, obj.loss)

    init_params = tf.dynamic_stitch(func.idx, obj.u_model.trainable_variables)

    lbfgs_op(func, init_params, newton_iter)

@tf.function
def lbfgs_op(func, init_params, newton_iter):
    results = tfp.optimizer.lbfgs_minimize(
        value_and_gradients_function=func, initial_position=init_params, max_iterations=newton_iter)
    return results


@tf.function
def train_op(obj, n_batches):
    for i in range(n_batches):
        if obj.isAdaptive:
            #unstack = tf.unstack(obj.u_model.trainable_variables, axis = 2)
            obj.variables = obj.u_model.trainable_variables
            obj.variables.extend([obj.u_weights, obj.col_weights])
            loss_value, mse_0, mse_b, mse_f, grads = obj.grad()
            obj.tf_optimizer.apply_gradients(zip(grads[:-2], obj.u_model.trainable_variables))
            obj.tf_optimizer_weights.apply_gradients(zip([-grads[-2], -grads[-1]], [obj.u_weights, obj.col_weights]))
        else:
            obj.variables = obj.u_model.trainable_variables
            loss_value, mse_0, mse_b, mse_f, grads = obj.grad()
            obj.tf_optimizer.apply_gradients(zip(grads, obj.u_model.trainable_variables))
        return loss_value, mse_0, mse_b, mse_f

def fit_dist(obj, tf_iter, newton_iter, batch_sz = None):

    BUFFER_SIZE = len(obj.x_f)
    EPOCHS = tf_iter
    obj.strategy = tf.distribute.MirroredStrategy(cross_device_ops = None)
    print("number of devices: {}".format(obj.strategy.num_replicas_in_sync))

    if batch_sz is not None:
        obj.batch_sz = batch_sz
    else:
        obj.batch_sz = len(obj.x_f)

    N_f = len(obj.x_f)
    n_batches =  N_f // obj.batch_sz

    BATCH_SIZE_PER_REPLICA = obj.batch_sz
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * obj.strategy.num_replicas_in_sync

    train_dataset = tf.data.Dataset.from_tensor_slices((obj.x_f, obj.t_f)).batch(GLOBAL_BATCH_SIZE)
    #col_weights = tf.data.Dataset.from_tensor_slices((obj.col_weights)).batch(GLOBAL_BATCH_SIZE)
    #print(GLOBAL_BATCH_SIZE)
    obj.train_dist_dataset = obj.strategy.experimental_distribute_dataset(train_dataset)
    #col_weights = obj.strategy.experimental_distribute_dataset(col_weights)

    start_time = time.time()

    with obj.strategy.scope():
        obj.u_model = neural_net(obj.layer_sizes)
        obj.tf_optimizer = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.99)
        obj.tf_optimizer_weights = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.99)
        #Can adjust batch size for collocation points, here we set it to N_f

        if obj.isAdaptive:
            obj.col_weights = tf.Variable(tf.random.uniform([5000, 1]))
            obj.u_weights = tf.Variable(100*tf.random.uniform([200, 1]))
    #tf_optimizer_u = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.99)

    print("starting Adam training")
    STEPS = np.max((n_batches // obj.strategy.num_replicas_in_sync,1))
    tf.profiler.experimental.start('../cache/tblogdir1')
    for epoch in range(tf_iter):
        train_loss = train_epoch(obj, obj.train_dist_dataset, obj.col_weights, STEPS)

        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            print('It: %d, Time: %.2f' % (epoch, elapsed))
            tf.print(f"total loss: {train_loss}")
            start_time = time.time()

    tf.profiler.experimental.stop()
    #l-bfgs-b optimization
    print("Starting L-BFGS training")
    lbfgs_train(obj, newton_iter)
    #tf.profiler.experimental.stop()


@tf.function
def train_epoch(obj, dataset, col_weights, STEPS):
    total_loss = 0.0
    num_batches = 0.0
    #dist_col_weights = iter(col_weights)
    dist_dataset_iterator = iter(dataset)
    for _ in range(STEPS):
        total_loss += distributed_train_step(obj, next(dist_dataset_iterator), col_weights)
        num_batches += 1
    train_loss = total_loss / num_batches
    return train_loss

@tf.function
def train_step(obj, inputs, col_weights):
    obj.dist_x_f, obj.dist_t_f = inputs
    obj.dist_col_weights = col_weights

    if obj.isAdaptive:
        unstack = tf.unstack(obj.u_model.trainable_variables)
        obj.grads_all = tf.stack([unstack, obj.u_weights, obj.dist_col_weights], 0)
        loss_value, mse_0, mse_b, mse_f, grads, grads_col, grads_u = obj.adaptgrad()
        obj.tf_optimizer.apply_gradients(zip(grads, obj.u_model.trainable_variables))
        obj.tf_optimizer_weights.apply_gradients(zip([-grads_u, -grads_col], [obj.u_weights, obj.dist_col_weights]))
    else:
        print("non-adaptive")
        loss_value, mse_0, mse_b, mse_f, grads = obj.grad()
        obj.tf_optimizer.apply_gradients(zip(grads, obj.u_model.trainable_variables))
    return loss_value

@tf.function
def distributed_train_step(obj, dataset_inputs, col_weights):
    per_replica_losses = obj.strategy.run(train_step, args=(obj, dataset_inputs, col_weights))
    return obj.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)
