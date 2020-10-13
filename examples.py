import tensordiffeq as tdq




@tf.function
def f_model(u, x, t):
    u_x = tf.gradients(u, x)
    u_xx = tf.gradients(u_x, x)
    u_t = tf.gradients(u,t)
    c1 = tf.constant(.0001, dtype = tf.float32)
    c2 = tf.constant(5.0, dtype = tf.float32)
    f_u = u_t - c1*u_xx + c2*u*u*u - c2*u
    return f_u

@tf.function
def u_x_model(x, t):
    u = u_model(tf.concat([x, t],1))
    u_x = tf.gradients(u, x)
    return u, u_x
