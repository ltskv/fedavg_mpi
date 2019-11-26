cimport numpy as np
import numpy as np
import mynet as mn

from libc.stdlib cimport malloc
from libc.string cimport memcpy


ctr = []
X_train, y_train, X_test, y_test = mn.load_mnist()
opt = mn.SGDOptimizer(lr=0.1)


cdef extern from "numpy/arrayobject.h":
    void *PyArray_DATA(np.ndarray arr)


ctypedef public struct Dense:
    long[2] shape
    int ownmem
    float* W
    float* b


ctypedef public struct Network:
    size_t n_layers;
    Dense* layers;


cdef public char * greeting():
    return f'The value is {3**3**3}'.encode('utf-8')


cdef public void debug_print(object o):
    print(o)


cdef public void predict(
    Network* net,
    float* X,
    size_t batch_size
):
    pass
    # try:
        # return net(X)
    # except Exception as e:
        # print(e)


cdef public void step_net(
    Network* c_net,
    float* batch_data,
    size_t batch_size
):
    net = wrap_c_network(c_net)
    cdef size_t in_dim = net.geometry[0]
    cdef size_t out_dim = net.geometry[-1]
    batch = np.asarray(<float[:batch_size,:in_dim+out_dim]>batch_data)
    net.step(batch[:, :in_dim], batch[:, in_dim:], opt)


cdef public float eval_net(Network* c_net):
    net = wrap_c_network(c_net)
    return net.evaluate(X_test, y_test, 'cls')


cdef public void mnist_batch(float* batch, size_t bs):
    idx = np.random.choice(len(X_train), bs, replace=False)
    arr = np.concatenate([X_train[idx], y_train[idx]], axis=1)
    memcpy(batch, <float*>PyArray_DATA(arr), arr.size*sizeof(float))


cdef public void create_c_network(Network* c_net):
    net = _create_network()
    c_net.n_layers = len(net.layers)
    c_net.layers = <Dense*>malloc(sizeof(Dense) * c_net.n_layers)
    for i, l in enumerate(net.layers):
        d0, d1 = l.W.shape
        c_net.layers[i].shape[0] = d0
        c_net.layers[i].shape[1] = d1
        c_net.layers[i].W = <float*>malloc(sizeof(float) * d0 * d1)
        c_net.layers[i].b = <float*>malloc(sizeof(float) * d1)
        memcpy(c_net.layers[i].W, PyArray_DATA(l.W), sizeof(float) * d0 * d1)
        memcpy(c_net.layers[i].b, PyArray_DATA(l.b), sizeof(float) * d1)
        c_net.layers[i].ownmem = 1


cdef public void be_like(Network* c_dst, Network* c_src):
    dst = wrap_c_network(c_dst)
    src = wrap_c_network(c_src)
    dst.be_like(src)


cdef object wrap_c_network(Network* c_net):
    net = _create_network(init=False)
    for i, l in enumerate(net.layers):
        d0, d1 = l.W.shape
        l.W = np.asarray(<float[:d0,:d1]>c_net.layers[i].W)
        l.b = np.asarray(<float[:d1]>c_net.layers[i].b)
    return net


cdef void inspect_array(
    np.ndarray[np.float32_t, ndim=2, mode='c'] a
):
    print(a.flags, flush=True)
    print(a.dtype, flush=True)
    print(a.sum(), flush=True)


def _create_network(init=True):
    return mn.Network((784, 10), mn.relu, mn.sigmoid, mn.bin_x_entropy,
                      initialize=init)


def combo_net(nets):
    return mn.combo_net(nets)
