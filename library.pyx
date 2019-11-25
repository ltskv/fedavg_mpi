cimport numpy as np
import numpy as np
import mynet as mn


ctr = []
X_train, y_train, X_test, y_test = mn.load_mnist()


cdef public char * greeting():
    return f'The value is {3**3**3}'.encode('utf-8')


cdef public void debug_print(object o):
    print(o.flags)
    # print(o)


cdef public np.ndarray[np.float32_t, ndim=2, mode='c'] dot(
    np.ndarray[np.float32_t, ndim=2, mode='c'] x,
    np.ndarray[np.float32_t, ndim=2, mode='c'] y
):
    return x @ y


cdef public np.ndarray[np.float32_t, ndim=2, mode='c'] predict(
    object net,
    np.ndarray[np.float32_t, ndim=2, mode='c'] X
):
    try:
        return net(X)
    except Exception as e:
        print(e)


cdef public object create_network():
    return mn.Network((784, 10), mn.relu, mn.sigmoid, mn.bin_x_entropy)


cdef public object combo_net(list nets):
    return mn.combo_net(nets)


cdef public object make_like(object neta, object netb):
    netb.be_like(neta)


cdef public void step_net(
    object net,
    np.ndarray[np.float32_t, ndim=2, mode='c'] batch
):
    opt = mn.SGDOptimizer(lr=0.1)
    net.step(batch[:, :784], batch[:, 784:], opt)


cdef public float eval_net(
    object net
):
    return net.evaluate(X_test, y_test, 'cls')


cdef public np.ndarray[np.float32_t, ndim=2, mode='c'] mnist_batch(
    Py_ssize_t bs
):
    idx = np.random.choice(len(X_train), bs, replace=False)
    arr = np.concatenate([X_train[idx], y_train[idx]], axis=1)
    return arr

cdef public float arrsum(
    np.ndarray[np.float32_t, ndim=2, mode='c'] a
):
    return np.sum(a)
