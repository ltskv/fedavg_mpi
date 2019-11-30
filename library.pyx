cimport numpy as np
import numpy as np

from sys import stderr

from libc.stdlib cimport malloc
from libc.string cimport memcpy

import nn


X_train, y_train, X_test, y_test = nn.load_mnist()


cdef extern from "numpy/arrayobject.h":
    void *PyArray_DATA(np.ndarray arr)


ctypedef public struct Weight:
    size_t dims
    long* shape
    float* W


ctypedef public struct WeightList:
    size_t n_weights;
    Weight* weights;


cdef public char *greeting():
    return f'The value is {3**3**3}'.encode('utf-8')


cdef public void debug_print(object o):
    print(o)


cdef public object create_network():
    return nn.create_mnist_network()


cdef public void set_net_weights(object net, WeightList* wl):
    net.set_weights(wrap_weight_list(wl))


cdef public void step_net(
    object net, float* X, float* y, size_t batch_size
):
    in_shape = (batch_size,) + net.layers[0].input_shape[1:]
    out_shape = (batch_size,) + net.layers[-1].output_shape[1:]
    X_train = np.asarray(<float[:np.prod(in_shape)]>X).reshape(in_shape)
    y_train = np.asarray(<float[:np.prod(out_shape)]>y).reshape(out_shape)

    net.train_on_batch(X_train, y_train)


cdef public float eval_net(object net):
    return net.evaluate(X_test, y_test, verbose=False)[1]


cdef public void mnist_batch(float* X, float* y, size_t bs,
                             int part, int total):
    if total == 0:
        X_pool, y_pool = X_train, y_train
    else:
        partsize = len(X_train) // total
        X_pool = X_train[part*partsize:(part+1)*partsize]
        y_pool = y_train[part*partsize:(part+1)*partsize]

    idx = np.random.choice(len(X_pool), bs, replace=True)

    X_r = X_pool[idx]
    y_r = y_pool[idx]

    assert X_r.flags['C_CONTIGUOUS']
    assert y_r.flags['C_CONTIGUOUS']
    memcpy(X, <float*>PyArray_DATA(X_r), X_r.size * sizeof(float))
    memcpy(y, <float*>PyArray_DATA(y_r), y_r.size * sizeof(float))


cdef public void init_weightlist_like(WeightList* wl, object net):
    weights = net.get_weights()
    wl.n_weights = len(weights)
    wl.weights = <Weight*>malloc(sizeof(Weight) * wl.n_weights)
    for i, w in enumerate(weights):
        sh = np.asarray(w.shape, dtype=long)
        wl.weights[i].dims = sh.size
        wl.weights[i].shape = <long*>malloc(sizeof(long) * sh.size)
        wl.weights[i].W = <float*>malloc(sizeof(float) * w.size)

        assert sh.flags['C_CONTIGUOUS']
        memcpy(wl.weights[i].shape, <long*>PyArray_DATA(sh),
               sh.size * sizeof(long))


cdef public void update_weightlist(WeightList* wl, object net):
    weights = net.get_weights()
    for i, w in enumerate(weights):
        w = w.astype(np.float32)

        assert w.flags['C_CONTIGUOUS']
        memcpy(wl.weights[i].W, <float*>PyArray_DATA(w),
               w.size * sizeof(float))


cdef public void combo_weights(
    WeightList* wl_frank, WeightList* wls, size_t num_weights
):
    """Not a one-liner anymore :/"""
    alpha = 1. / num_weights
    frank = wrap_weight_list(wl_frank)
    for w in frank:
        w[:] = 0
    for i in range(num_weights):
        for wf, ww in zip(frank, wrap_weight_list(&wls[i])):
            wf += alpha * ww


cdef list wrap_weight_list(WeightList* wl):
    weights = []
    for i in range(wl.n_weights):
        w_shape = <long[:wl.weights[i].dims]>wl.weights[i].shape
        w_numel = np.prod(w_shape)
        weights.append(
            np.asarray(<float[:w_numel]>wl.weights[i].W).reshape(w_shape)
        )
    return weights


def inspect_array(a):
    print(a.flags, flush=True)
    print(a.dtype, flush=True)
    print(a.sum(), flush=True)
