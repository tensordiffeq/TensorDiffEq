class CollocationModel1D:
    def__init(self):
        self.layer_sizes = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

        self.sizes_w = []
        self.sizes_b = []

        for i, width in enumerate(self.layer_sizes):
            if i != 1:
                self.sizes_w.append(int(width * layer_sizes[1]))
                self.sizes_b.append(int(width if i != 0 else layer_sizes[1]))


    def set_weights(self, model, w, sizes_w, sizes_b):
            for i, layer in enumerate(model.layers[0:]):
                #print(w)
               # print(i, layer)
                start_weights = sum(sizes_w[:i]) + sum(sizes_b[:i])
                #print("start weights",np.shape(start_weights), start_weights)

                end_weights = sum(sizes_w[:i+1]) + sum(sizes_b[:i])
                #print("end weights", np.shape(end_weights), end_weights)

                weights = w[start_weights:end_weights]
                #print("weights", np.shape(weights), weights)

                w_div = int(sizes_w[i] / sizes_b[i])
                #print("w_div", w_div)

                weights = tf.reshape(weights, [w_div, sizes_b[i]])
                #print("weights", np.shape(weights), weights)

                biases = w[end_weights:end_weights + sizes_b[i]]
                #print("biases", np.shape(biases), biases)

                weights_biases = [weights, biases]
                #print("weights_biases", np.shape(weights_biases), weights_biases)

                layer.set_weights(weights_biases)

    def get_weights(self, model):
        w = []
        for layer in model.layers[0:]:
            weights_biases = layer.get_weights()
            weights = weights_biases[0].flatten()
            biases = weights_biases[1]
            w.extend(weights)
            w.extend(biases)

        w = tf.convert_to_tensor(w)
        return w

            #custom nn with regularized inputs in the domain [lb,ub] = [-5,5]
    # regular dense nn with tanh activations, reorder this cell with the above to select which type of network to use
    # the second neural_net function definition will overwrite the first, making it the one used in the model
    def neural_net(self, layer_sizes):
        model = Sequential()
        model.add(layers.InputLayer(input_shape=(layer_sizes[0],)))
        for width in layer_sizes[1:-1]:
            model.add(layers.Dense(
                width, activation=tf.nn.tanh,
                kernel_initializer="glorot_normal"))
        model.add(layers.Dense(
                layer_sizes[-1], activation=None,
                kernel_initializer="glorot_normal"))
        return model


    u_model = self.neural_net(self.layer_sizes)

    def loss(self, x_f_batch, t_f_batch,
                 x0, t0, u0, x_lb,
                 t_lb, x_ub, t_ub, col_weights, u_weights):


        u0_pred = self.u_model(tf.concat([x0, t0],1))
        #u0_pred = u0_pred[:, 0:1]
        #v0_pred = uv0_pred[:, 1:2]
        u_lb_pred, u_x_lb_pred = u_x_model(x_lb, t_lb)
        u_ub_pred, u_x_ub_pred = u_x_model(x_ub, t_ub)
        #mu, mu_xx = mu_xx_model(x_f_batch, t_f_batch)
        #u_t_mapping = u_t_map(x_f_batch,t_f_batch, mu_xx)
        f_u_pred = f_model(x_f_batch, t_f_batch)

        #h0_pred = tf.sqrt(tf.square(u0_pred) + tf.square(v0_pred))

        #mse_0_u = tf.reduce_mean(tf.square(u_weights*(u0 - u0_pred)))
        mse_0_u = tf.reduce_mean(tf.square((u0 - u0_pred)))
        #mse_0_v = tf.reduce_mean(tf.square(v0 - v0_pred))

        mse_b_u = tf.reduce_mean(tf.square(tf.math.subtract(u_lb_pred, u_ub_pred))) + \
                tf.reduce_mean(tf.square(tf.math.subtract(u_x_lb_pred, u_x_ub_pred)))
        #mse_b_v = tf.reduce_mean(tf.square(v_lb_pred - v_ub_pred)) + \
                #tf.reduce_mean(tf.square(v_x_lb_pred-v_x_ub_pred))

        mse_f_u = tf.reduce_mean(tf.square(col_weights*f_u_pred))
        #mse_f_u = tf.reduce_mean(tf.square(f_u_pred))
        #mse_f_v = tf.reduce_mean(tf.square(col_weights*f_v_pred))


        return  mse_0_u + mse_b_u + mse_f_u , mse_0_u, mse_f_u

    def f_model(x, t):
        g1 = tf.constant(0.01, dtype = tf.float32)
        g2 = tf.constant(10e-6, dtype = tf.float32)

        u = u_model(tf.concat([x, t],1))
        u_x = tf.gradients(u, x)
        u_xx = tf.gradients(u_x, x)
        u_t = tf.gradients(u, t)

        tmp = g1*(u**3 - u) - g2*u_xx
        tmp_x = tf.gradients(tmp, x)
        tmp_xx = tf.gradients(tmp_x, x)

        f_u = tf.math.subtract(u_t, tmp_xx)

        return f_u

    def u_x_model(x, t):
        u = u_model(tf.concat([x, t],1))
        u_x = tf.gradients(u, x)

        return u, u_x


    @tf.function
    def grad(model, x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights):
        with tf.GradientTape(persistent=True) as tape:
            #tape.watch(col_weights)
            #tape.watch(u_weights)
            loss_value, mse_0, mse_f = loss(x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights)
            grads = tape.gradient(loss_value, u_model.trainable_variables)
            #print(grads)
            grads_col = tape.gradient(loss_value, col_weights)
            grads_u = tape.gradient(loss_value, u_weights)

        return loss_value, mse_0, mse_f, grads, grads_col

    def fit(x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights, tf_iter, newton_iter):

            # Creating the tensors
    #     X_u = tf.cast(X_u, dtype = tf.float32)
    #     u = tf.cast(u, dtype = tf.float32)

        start_time = time.time()
        tf_optimizer = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.99)
        tf_optimizer_coll = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.99)
        tf_optimizer_u = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.99)

        print("starting Adam training")
        #Adam optimization
        with tf.summary.create_file_writer('/tensorflow/logdir').as_default() as writer:
            tf.summary.trace_on()
            for epoch in range(tf_iter):

                with tf.GradientTape(persistent=True) as tape:
                    #tape.watch(col_weights)
                    #tape.watch(u_weights)
                    loss_value, mse_0, mse_f, grads, grads_col = grad(u_model, x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights)

                    #grads_col = tape.gradient(loss_value, col_weights)
                    #grads_u = tape.gradient(loss_value, u_weights)

                tf_optimizer.apply_gradients(zip(grads, u_model.trainable_variables))
                tf_optimizer_coll.apply_gradients(zip([-grads_col], [col_weights]))
               # tf_optimizer_u.apply_gradients(zip([-grads_u], [u_weights]))


                del tape
                #loss_value = tf_optimization_step(X_u, u, x_f_batch, t_f_batch) #iterate across full X_u and u data for each batch
                if epoch % 10 == 0:
                    elapsed = time.time() - start_time
                    print('It: %d, Time: %.2f' % (epoch, elapsed))
                    tf.print(f"mse_0: {mse_0}  mse_f: {mse_f}   total loss: {loss_value}")
                    start_time = time.time()
                if epoch == 0:
                    tf.summary.trace_export(name="all", step=epoch, profiler_outdir='/tensorflow/logdir')
                #tf.summary.trace_export(name="all", step=epoch, profiler_outdir='/tensorflow/logdir')
                tf.summary.scalar('loss', loss_value, step=epoch)
                writer.flush()


        print(col_weights)
        #l-bfgs-b optimization
        print("Starting L-BFGS training")

        loss_and_flat_grad = get_loss_and_flat_grad(x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights)

        lbfgs(loss_and_flat_grad,
          get_weights(u_model),
          Struct(), maxIter=newton_iter, learningRate=0.8)

    #         lbfgs(loss_and_flat_grad_col,
    #           col_weights,
    #           Struct(), maxIter=10, learningRate=0.8)
