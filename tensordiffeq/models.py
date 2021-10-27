import tensorflow as tf
import numpy as np
import time
from .utils import *
from .networks import *
from .plotting import *
from .fit import *
from tqdm.auto import tqdm, trange
from .output import print_screen


class CollocationSolverND:
    def __init__(self, assimilate=False, verbose=True):
        self.assimilate = assimilate
        self.verbose = verbose

    def compile(self, layer_sizes, f_model, domain, bcs, isAdaptive=False,
                dict_adaptive=None, init_weights=None, g=None, dist=False):
        """
        Args:
            layer_sizes: A list of layer sizes, can be overwritten via resetting u_model to a keras model
            f_model: PDE definition
            domain: a Domain object containing the information on the domain of the system
            bcs: a list of ICs/BCs for the problem
            isAdaptive: Boolean value determining whether to implement self-adaptive solving
            dict_adaptive: a dictionary with boollean indicating adaptive loss for every loss function
            init_weights: a dictionary with keys "residual" and "BCs". Values must be a tuple with dimension
                          equal to the number of  residuals and boundares conditions, respectively
            g: a function in terms of `lambda` for self-adapting solving. Defaults to lambda^2
            dist: A boolean value determining whether the solving will be distributed across multiple GPUs

        Returns:
            None
        """
        self.tf_optimizer = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
        self.tf_optimizer_weights = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
        self.layer_sizes = layer_sizes
        self.sizes_w, self.sizes_b = get_sizes(layer_sizes)
        self.bcs = bcs
        self.f_model = get_tf_model(f_model)
        self.g = g
        self.domain = domain
        self.dist = dist
        self.X_f_dims = tf.shape(self.domain.X_f)
        self.X_f_len = tf.slice(self.X_f_dims, [0], [1]).numpy()
        # must explicitly cast data into tf.float32 for stability
        # tmp = [tf.cast(np.reshape(vec, (-1, 1)), tf.float32) for i, vec in enumerate(self.domain.X_f.T)]
        # self.X_f_in = np.asarray(tmp)
        self.X_f_in = [tf.cast(np.reshape(vec, (-1, 1)), tf.float32) for i, vec in enumerate(self.domain.X_f.T)]
        self.u_model = neural_net(self.layer_sizes)
        self.batch = None
        self.batch_indx_map = None
        self.lambdas = self.dict_adaptive = self.lambdas_map = None
        self.isAdaptive = isAdaptive

        if self.isAdaptive:
            self.dict_adaptive = dict_adaptive
            self.lambdas, self.lambdas_map = initialize_weights_loss(init_weights)

            if dict_adaptive is None and init_weights is None:
                raise Exception("Adaptive weights selected but no inputs were specified!")
        if (
                self.isAdaptive is False
                and self.dict_adaptive is not None
                and self.lambdas is not None
            ):
            raise Exception(
                "Adaptive weights are turned off but weight vectors were provided. Set the weight vectors to "
                "\"none\" to continue")

    def compile_data(self, x, t, y):
        if not self.assimilate:
            raise Exception(
                "Assimilate needs to be set to 'true' for data assimilation. Re-initialize CollocationSolver1D with "
                "assimilate=True.")
        self.data_x = x
        self.data_t = t
        self.data_s = y

    def update_loss(self):
        loss_bcs = 0.

        #####################################
        # BOUNDARIES and INIT conditions
        #####################################
        # Check if adaptive is allowed
        if self.isAdaptive:
            if len(self.lambdas_map['bcs']) > 0:
                idx_lambda_bcs = self.lambdas_map['bcs'][0]

        for counter_bc, bc in enumerate(self.bcs):
            loss_bc = 0.
            # Check if the current BC is adaptive
            if self.isAdaptive:
                isBC_adaptive = self.dict_adaptive["BCs"][counter_bc]
            else:
                isBC_adaptive = False

            # Periodic BC iteration for all components of deriv_model
            if bc.isPeriodic:
                if isBC_adaptive:
                    # TODO: include Adapative Periodic Boundaries Conditions
                    raise Exception('TensorDiffEq is currently not accepting Adapative Periodic Boundaries Conditions')
                else:
                    for i, dim in enumerate(bc.var):
                        for j, lst in enumerate(dim):
                            for k, tup in enumerate(lst):
                                upper = bc.u_x_model(self.u_model, bc.upper[i])[j][k]
                                lower = bc.u_x_model(self.u_model, bc.lower[i])[j][k]
                                msq = MSE(upper, lower)
                                loss_bc = tf.math.add(loss_bc, msq)
            # initial BCs, including adaptive model
            elif bc.isInit:
                if isBC_adaptive:
                    loss_bc = MSE(self.u_model(bc.input), bc.val, self.lambdas[idx_lambda_bcs])
                    idx_lambda_bcs += 1
                else:
                    loss_bc = MSE(self.u_model(bc.input), bc.val)
            # BC types are added
            elif bc.isNeumann:
                if isBC_adaptive:
                    #TODO: include Adapative Neumann Boundaries Conditions
                    raise Exception('TensorDiffEq is currently not accepting Adapative Neumann Boundaries Conditions')
                else:
                    for i, dim in enumerate(bc.var):
                        for j, lst in enumerate(dim):
                            for k, tup in enumerate(lst):
                                target = tf.cast(bc.u_x_model(self.u_model, bc.input[i])[j][k], dtype=tf.float32)
                                msq = MSE(bc.val, target)
                                loss_bc = tf.math.add(loss_bc, msq)

            elif bc.isDirichlect:
                if isBC_adaptive:
                    loss_bc = MSE(self.u_model(bc.input), bc.val, self.lambdas[idx_lambda_bcs])
                    idx_lambda_bcs += 1
                else:
                    loss_bc = MSE(self.u_model(bc.input), bc.val)

            else:
                raise Exception('Boundary condition type is not acceptable')

            loss_bcs = tf.add(loss_bcs, loss_bc)

        #####################################
        # Residual Equations
        #####################################
        # pass thorough the forward method
        if self.n_batches > 1:
            # The collocation points will be split based on the batch_indx_map
            # generated on the beginning of this epoch on models.train_op_inner.apply_grads
            X_batch = []
            for x_in in self.X_f_in:
                indx_on_batch = self.batch_indx_map[self.batch * self.batch_sz:(self.batch + 1) * self.batch_sz]
                X_batch.append(tf.gather(x_in,indx_on_batch))
            f_u_preds = self.f_model(self.u_model, *X_batch)
        else:
            f_u_preds = self.f_model(self.u_model, *self.X_f_in)

        # If it is only one residual, just convert it to a tuple of one element
        if not isinstance(f_u_preds, tuple):
            f_u_preds = f_u_preds,

        loss_res = 0.
        for counter_res, f_u_pred in enumerate(f_u_preds):
            # Check if the current Residual is adaptive
            if self.isAdaptive:
                isRes_adaptive = self.dict_adaptive["residual"][counter_res]
                if isRes_adaptive:
                    idx_lambda_res = self.lambdas_map['residual'][0]
                    lambdas2loss = self.lambdas[idx_lambda_res]

                    if self.n_batches > 1:
                        # select lambdas on minebatch
                        lambdas2loss = tf.gather(lambdas2loss,indx_on_batch)

                    if self.g is not None:
                        loss_r = g_MSE(f_u_pred, constant(0.0), self.g(lambdas2loss))
                    else:
                        loss_r = MSE(f_u_pred, constant(0.0), lambdas2loss)
                    idx_lambda_res += 1
                else:
                    # In the case where the model is Adaptive but the residual
                    # is not adaptive, the residual loss should be computed.
                    loss_r = MSE(f_u_pred, constant(0.0))
            else:
                loss_r = MSE(f_u_pred, constant(0.0))

            loss_res = tf.math.add(loss_r, loss_res)

        loss_total = tf.math.add(loss_res, loss_bcs)

        return loss_total

    # @tf.function
    def grad(self):
        with tf.GradientTape() as tape:
            loss_value = self.update_loss()
            grads = tape.gradient(loss_value, self.variables)
        return loss_value, grads

    def fit(self, tf_iter=0, newton_iter=0, batch_sz=None, newton_eager=True):

        # Can adjust batch size for collocation points, here we set it to N_f
        N_f = self.X_f_len[0]
        self.batch_sz = batch_sz if batch_sz is not None else N_f
        self.n_batches = N_f // self.batch_sz

        if self.isAdaptive and self.dist:
            raise Exception("Currently we dont support distributed training for adaptive PINNs")

        if self.n_batches > 1 and self.dist:
            raise Exception("Currently we dont support distributed minibatching training")

        if self.dist:
            BUFFER_SIZE = len(self.X_f_in[0])
            EPOCHS = tf_iter
            # devices = ['/gpu:0', '/gpu:1','/gpu:2', '/gpu:3'],
            try:
                self.strategy = tf.distribute.MirroredStrategy()
            except:
                print(
                    "Looks like we cant find any GPUs available, or your GPUs arent responding to Tensorflow's API. If "
                    "you're receiving this in error, check that your CUDA, "
                    "CUDNN, and other GPU dependencies are installed correctly with correct versioning based on your "
                    "version of Tensorflow")

            print("Number of GPU devices: {}".format(self.strategy.num_replicas_in_sync))

            BATCH_SIZE_PER_REPLICA = self.batch_sz
            GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * self.strategy.num_replicas_in_sync

            # options = tf.data.Options()
            # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

            self.train_dataset = tf.data.Dataset.from_tensor_slices(
                self.X_f_in).batch(GLOBAL_BATCH_SIZE)

            # self.train_dataset = self.train_dataset.with_options(options)

            self.train_dist_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)

            start_time = time.time()

            with self.strategy.scope():
                self.u_model = neural_net(self.layer_sizes)
                self.tf_optimizer = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
                self.tf_optimizer_weights = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
                # self.dist_col_weights = tf.Variable(tf.zeros(batch_sz), validate_shape=True)

                if self.isAdaptive:
                    # self.col_weights = tf.Variable(tf.random.uniform([self.batch_sz, 1]))
                    self.u_weights = tf.Variable(self.u_weights)

            fit_dist(self, tf_iter=tf_iter, newton_iter=newton_iter, batch_sz=batch_sz, newton_eager=newton_eager)

        else:
            fit(self, tf_iter=tf_iter, newton_iter=newton_iter, newton_eager=newton_eager)

    # L-BFGS implementation from https://github.com/pierremtb/PINNs-TF2.0
    def get_loss_and_flat_grad(self):
        def loss_and_flat_grad(w):
            with tf.GradientTape() as tape:
                set_weights(self.u_model, w, self.sizes_w, self.sizes_b)
                loss_value = self.update_loss()
            grad = tape.gradient(loss_value, self.u_model.trainable_variables)
            grad_flat = []
            for g in grad:
                grad_flat.append(tf.reshape(g, [-1]))
            grad_flat = tf.concat(grad_flat, 0)
            return loss_value, grad_flat

        return loss_and_flat_grad

    def predict(self, X_star):
        # predict using concatenated data
        u_star = self.u_model(X_star)
        # split data into tuples for ND support
        # must explicitly cast data into tf.float32 for stability
        # tmp = [tf.cast(np.reshape(vec, (-1, 1)), tf.float32) for i, vec in enumerate(X_star.T)]
        # X_star = np.asarray(tmp)
        # X_star = tuple(X_star)
        X_star = [tf.cast(np.reshape(vec, (-1, 1)), tf.float32) for i, vec in enumerate(X_star.T)]
        f_u_star = self.f_model(self.u_model, *X_star)
        return u_star.numpy(), f_u_star.numpy()

    def save(self, path):
        self.u_model.save(path)

    def load_model(self, path, compile_model=False):
        self.u_model = tf.keras.models.load_model(path, compile=compile_model)


# WIP
# TODO Distributed Discovery Model
class DiscoveryModel():
    def compile(self, layer_sizes, f_model, X, u, var, col_weights=None):
        self.layer_sizes = layer_sizes
        self.f_model = get_tf_model(f_model)
        self.X = X
        self.u = u
        self.vars = var
        self.len_ = len(var)
        self.u_model = neural_net(self.layer_sizes)
        self.tf_optimizer = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
        self.tf_optimizer_vars = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
        self.tf_optimizer_weights = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
        self.col_weights = col_weights
        # tmp = [np.reshape(vec, (-1,1)) for i, vec in enumerate(self.X)]
        self.X_in = tuple(X)
        # self.X_in = np.asarray(tmp).T

    # print(np.shape(self.X_in))

    @tf.function
    def loss(self):
        u_pred = self.u_model(tf.concat(self.X, 1))
        f_u_pred = self.f_model(self.u_model, self.vars, *self.X_in)
        if self.col_weights is not None:
            return MSE(u_pred, self.u) + g_MSE(f_u_pred, constant(0.0), self.col_weights ** 2)
        else:
            return MSE(u_pred, self.u) + MSE(f_u_pred, constant(0.0))

    @tf.function
    def grad(self):
        with tf.GradientTape() as tape:
            loss_value = self.loss()
            grads = tape.gradient(loss_value, self.variables)
        return loss_value, grads

    @tf.function
    def train_op(self):
        self.variables = self.u_model.trainable_variables
        len_ = self.len_
        if self.col_weights is not None:

            self.variables.extend([self.col_weights])
            self.variables.extend(self.vars)
            loss_value, grads = self.grad()
            self.tf_optimizer.apply_gradients(zip(grads[:-(len_ + 2)], self.u_model.trainable_variables))
            self.tf_optimizer_weights.apply_gradients(zip([-grads[-(len_ + 1)]], [self.col_weights]))
            self.tf_optimizer_vars.apply_gradients(zip(grads[-len_:], self.vars))
        else:
            self.variables.extend(self.vars)
            loss_value, grads = self.grad()

            self.tf_optimizer.apply_gradients(zip(grads[:-(len_ + 1)], self.u_model.trainable_variables))

            self.tf_optimizer_vars.apply_gradients(zip(grads[-len_:], self.vars))

        return loss_value

    def fit(self, tf_iter):
        self.train_loop(tf_iter)

    def train_loop(self, tf_iter):  # sourcery skip: move-assign
        start_time = time.time()
        print_screen(self, discovery_model=True)
        with trange(tf_iter) as t:
            for i in t:
                loss_value = self.train_op()
                if i % 10 == 0:
                    # elapsed = time.time() - start_time
                    # print('It: %d, Time: %.2f' % (i, elapsed))
                    # tf.print(f"loss_value: {loss_value}")
                    var = [var.numpy() for var in self.vars]
                    t.set_postfix(loss=loss_value.numpy())
                    t.set_postfix(vars=var)
                    # tf.print(f"vars estimate(s): {var}")
                    # start_time = time.time()
