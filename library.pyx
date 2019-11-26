cimport numpy as np
import numpy as np
import mynet as mn

from libc.stdlib cimport malloc


ctr = []
X_train, y_train, X_test, y_test = mn.load_mnist()
opt = mn.SGDOptimizer(lr=0.1)


cdef extern from "numpy/arrayobject.h":
    object PyArray_SimpleNewFromData(
        int nd, long* dims, int typenum, void* data
    )
    void *PyArray_DATA(np.ndarray arr)


ctypedef public struct Dense:
    long[2] shape
    int ownmem
    float* W
    float* b


ctypedef public struct Network:
    Py_ssize_t n_layers;
    Dense* layers;


cdef public char * greeting():
    return f'The value is {3**3**3}'.encode('utf-8')


cdef public void debug_print(object o):
    print(o.flags)


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
    float* batch_data,
    Py_ssize_t batch_size
):
    cdef Py_ssize_t in_dim = net.geometry[0]
    cdef Py_ssize_t out_dim = net.geometry[-1]
    batch = np.asarray(<float[:batch_size,:in_dim+out_dim]>batch_data)
    net.step(batch[:, :in_dim], batch[:, in_dim:], opt)


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


cdef public void inspect_array(
    np.ndarray[np.float32_t, ndim=2, mode='c'] a
):
    print(a.flags)
    print(a.dtype)
    print(a.sum())


cdef public void be_like_cified(
    object net,
    Network* c_net
):
   """WARNING this function makes an assumption that `net` and `c_net`
   have the same shape and hopefully is going to crash horribly otherwise."""
   for i, l in enumerate(net.layers):
       w1, w2 = l.W.shape
       l.W[:] = <float[:w1,:w2]>c_net.layers[i].W
       l.b[:] = <float[:w2]>c_net.layers[i].b


cdef public void cify_network(
    object net, Network* c_net
):
    """WARNING `c_net` is valid as long as `net` is

    Whoever has `c_net` is responsible for freeing c_net.layers list
    Layers themselves don't need any de-init.
    """
    c_net.n_layers = len(net.layers)
    c_net.layers = <Dense*>malloc(len(net.layers) * sizeof(Dense))
    for i, l in enumerate(net.layers):
        w1, w2 = l.W.shape
        c_net.layers[i].shape[0] = w1
        c_net.layers[i].shape[1] = w2
        c_net.layers[i].W = <float*>PyArray_DATA(l.W)
        c_net.layers[i].b = <float*>PyArray_DATA(l.b)
        c_net.layers[i].ownmem = 0
