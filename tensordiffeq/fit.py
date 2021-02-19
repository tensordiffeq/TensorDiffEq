import tensorflow as tf
import numpy as np
from .networks import *
from .models import *
from .utils import *
from .optimizers import *
import time
import os
import sys

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"


def fit(obj, tf_iter=0, newton_iter=0, batch_sz=None, newton_eager=True):
    # obj.u_model = neural_net(obj.layer_sizes)
    # obj.build_loss()
    # Can adjust batch size for collocation points, here we set it to N_f
    if batch_sz is not None:
        obj.batch_sz = batch_sz
    else:
        obj.batch_sz = obj.X_f_len
        # obj.batch_sz = len(obj.x_f)

    N_f = obj.X_f_len
    # N_f = len(obj.x_f)
    n_batches = int(N_f // obj.batch_sz)
    start_time = time.time()
    # obj.tf_optimizer = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
    # obj.tf_optimizer_weights = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)

    # these cant be tf.functions on initialization since the distributed strategy requires its own
    # graph using grad and adaptgrad, so they cant be compiled as tf.functions until we know dist/non-dist
    obj.grad = tf.function(obj.grad)
    print("starting Adam training")
    # tf.profiler.experimental.start('../cache/tblogdir1')
    train_op_fn = train_op_inner(obj)
    for epoch in range(tf_iter):
        loss_value = train_op_fn(n_batches, obj)

        if epoch % 100 == 0:
            elapsed = time.time() - start_time
            print('It: %d, Time: %.2f' % (epoch, elapsed))
            # tf.print(f"mse_0: {mse_0}  mse_b  {mse_b}  mse_f: {mse_f}   total loss: {loss_value}")
            tf.print(f"total loss: {loss_value}")
            start_time = time.time()
    # tf.profiler.experimental.stop()


    # tf.profiler.experimental.start('../cache/tblogdir1')
    if newton_iter > 0:
        print("Starting L-BFGS training")
        if newton_eager:
            print("Executing eager-mode L-BFGS")
            loss_and_flat_grad = obj.get_loss_and_flat_grad()
            eager_lbfgs(loss_and_flat_grad,
                        get_weights(obj.u_model),
                        Struct(), maxIter=newton_iter, learningRate=0.8)

        else:
            print("Executing graph-mode L-BFGS\n Building graph...")
            print("Warning: Depending on your CPU/GPU setup, eager-mode L-BFGS may prove faster. If the computational "
                  "graph takes a long time to build, or the computation is slow, try eager-mode L-BFGS (enabled by "
                  "default)")

            lbfgs_train(obj, newton_iter)

    # tf.profiler.experimental.stop()


# @tf.function
def lbfgs_train(obj, newton_iter):
    func = graph_lbfgs(obj.u_model, obj.update_loss)

    init_params = tf.dynamic_stitch(func.idx, obj.u_model.trainable_variables)

    lbfgs_op(func, init_params, newton_iter)


@tf.function
def lbfgs_op(func, init_params, newton_iter):
    return tfp.optimizer.lbfgs_minimize(
        value_and_gradients_function=func,
        initial_position=init_params,
        max_iterations=newton_iter,
        tolerance=1e-20,
    )


def train_op_inner(obj):
    @tf.function
    def apply_grads(n_batches, obj=obj):
        for _ in range(n_batches):
            # unstack = tf.unstack(obj.u_model.trainable_variables, axis = 2)
            obj.variables = obj.u_model.trainable_variables
            if obj.isAdaptive:
                obj.variables.extend([obj.u_weights, obj.col_weights])
                loss_value, grads = obj.grad()
                obj.tf_optimizer.apply_gradients(zip(grads[:-2], obj.u_model.trainable_variables))
                obj.tf_optimizer_weights.apply_gradients(
                    zip([-grads[-2], -grads[-1]], [obj.u_weights, obj.col_weights]))
            else:
                loss_value, grads = obj.grad()
                obj.tf_optimizer.apply_gradients(zip(grads, obj.u_model.trainable_variables))
            return loss_value

    return apply_grads


# TODO Distributed training re-integration
# TODO decouple u_model from being overwritten by calling model.fit

def fit_dist(obj, tf_iter, newton_iter, batch_sz=None, newton_eager=True):
    BUFFER_SIZE = len(obj.x_f)
    EPOCHS = tf_iter
    # devices = ['/gpu:0', '/gpu:1','/gpu:2', '/gpu:3'],
    try:
        obj.strategy = tf.distribute.MirroredStrategy()
    except:
        print("Looks like we cant find any GPUs available, or your GPUs arent responding to Tensorflow's API. If "
              "you're receiving this in error, check that your CUDA, "
              "CUDNN, and other GPU dependencies are installed correctly with correct versioning based on your "
              "version of Tensorflow")

    print("Number of GPU devices: {}".format(obj.strategy.num_replicas_in_sync))

    if batch_sz is not None:
        obj.batch_sz = batch_sz
    else:
        obj.batch_sz = len(obj.x_f)

    weights_idx = tensor(list(range(len(obj.x_f))), dtype=tf.int32)
    print(weights_idx)
    # print(tf.gather(obj.col_weights, weights_idx))
    N_f = len(obj.x_f)
    n_batches = N_f // obj.batch_sz

    BATCH_SIZE_PER_REPLICA = obj.batch_sz
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * obj.strategy.num_replicas_in_sync

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    obj.train_dataset = tf.data.Dataset.from_tensors((weights_idx, obj.x_f, obj.t_f))  # .batch(GLOBAL_BATCH_SIZE)

    obj.train_dataset = obj.train_dataset.with_options(options)

    obj.train_dist_dataset = obj.strategy.experimental_distribute_dataset(obj.train_dataset)

    start_time = time.time()

    with obj.strategy.scope():
        obj.u_model = neural_net(obj.layer_sizes)
        obj.tf_optimizer = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
        obj.tf_optimizer_weights = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
        obj.dist_col_weights = tf.Variable(tf.zeros(batch_sz), validate_shape=True)

        if obj.isAdaptive:
            obj.col_weights = tf.Variable(tf.random.uniform([obj.batch_sz, 1]))
            obj.u_weights = tf.Variable(obj.u_weights)

    def train_epoch(dataset, STEPS):
        total_loss = 0.0
        num_batches = 0.0
        # dist_col_weights = iter(col_weights)
        dist_dataset_iterator = iter(dataset)
        for _ in range(STEPS):
            total_loss += distributed_train_step(obj, next(dist_dataset_iterator))
            num_batches += 1
        train_loss = total_loss / num_batches
        return train_loss

    def train_step(obj, inputs):
        col_idx, obj.dist_x_f, obj.dist_t_f = inputs
        print(obj.dist_x_f, obj.dist_t_f)
        # obj.dist_col_weights = col_weights
        if obj.isAdaptive:
            obj.variables = obj.u_model.trainable_variables
            obj.dist_col_weights = tf.gather(obj.col_weights, col_idx)
            print(obj.dist_col_weights)
            obj.variables.extend([obj.u_weights, obj.dist_col_weights])
            loss_value, grads = obj.grad()
            obj.tf_optimizer.apply_gradients(zip(grads[:-2], obj.u_model.trainable_variables))
            print([grads[-2], grads[-1]])
            obj.tf_optimizer_weights.apply_gradients(
                zip([-grads[-2], -grads[-1]], [obj.u_weights, obj.dist_col_weights]))
            # TODO collocation weight splitting across replicas
            # tf.scatter_nd_add(obj.col_weights, col_idx, obj.dist_col_weights)
        else:
            obj.variables = obj.u_model.trainable_variables
            loss_value, grads = obj.grad()
            obj.tf_optimizer.apply_gradients(zip(grads, obj.u_model.trainable_variables))
        return loss_value

    @tf.function
    def distributed_train_step(obj, dataset_inputs):
        per_replica_losses = obj.strategy.run(train_step, args=(obj, dataset_inputs))
        return obj.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                   axis=None)

    @tf.function
    def dist_loop(obj, STEPS):
        total_loss = 0.0
        num_batches = 0.0
        # dist_col_weights = iter(col_weights)
        dist_dataset_iterator = iter(obj.train_dist_dataset)
        for _ in range(STEPS):
            total_loss += distributed_train_step(obj, next(dist_dataset_iterator))
            num_batches += 1
        train_loss = total_loss / num_batches

        return train_loss

    def train_loop(obj, tf_iter, STEPS):

        start_time = time.time()
        for epoch in range(tf_iter):
            loss = dist_loop(obj, STEPS)

            if epoch % 100 == 0:
                elapsed = time.time() - start_time
                template = ("Epoch {}, Time: {}, Loss: {}")
                print(template.format(epoch, elapsed, loss))
                # print('It: %d, Time: %.2f, loss: %.9f' % (epoch, elapsed, tf.get_static_value(loss)))
                start_time = time.time()

    print("starting Adam training")
    STEPS = np.max((n_batches // obj.strategy.num_replicas_in_sync, 1))
    # tf.profiler.experimental.start('../cache/tblogdir1')
    train_loop(obj, tf_iter, STEPS)
    # tf.profiler.experimental.stop()

    # l-bfgs-b optimization
    print("Starting L-BFGS training")
    # lbfgs_train(obj, newton_iter)
    # tf.profiler.experimental.stop()
