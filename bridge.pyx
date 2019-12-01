cimport numpy as np
import numpy as np

from sys import stderr

from libc.stdlib cimport malloc, realloc
from libc.string cimport memcpy

import library as nn


X_train, y_train, X_test, y_test = nn.load_mnist()
tokenizers = {}


cdef extern from "numpy/arrayobject.h":
    void *PyArray_DATA(np.ndarray arr)


ctypedef public struct Weight:
    size_t dims
    long* shape
    float* W


ctypedef public struct WeightList:
    size_t n_weights;
    Weight* weights;


ctypedef public struct Word:
    size_t mem
    char* data


ctypedef public struct WordList:
    size_t mem
    size_t n_words
    Word* words


cdef public char *greeting():
    return f'The value is {3**3**3}'.encode('utf-8')


cdef public int get_tokens(WordList* wl, const char *filename):
    fnu = filename.decode('utf-8')
    if fnu not in tokenizers:
        tokenizers[fnu] = nn.token_generator(fnu)
    g = tokenizers[fnu]
    try:
        words = next(g)
    except StopIteration:
        return 0
    words_into_wordlist(wl, words)
    return 1


cdef public long vocab_idx_of(Word* w):
    word = w.data.decode('utf-8')
    try:
        return nn.vocab[word]
    except KeyError:
        return -1


cdef public void f_idx_list_to_print(float* f_idxs, size_t num):
    idxs = np.asarray(<float[:num]>f_idxs).astype(np.int)
    cdef str pyuni = ' '.join(nn.inv_vocab[i] for i in idxs)
    print(pyuni)
    # cdef bytes b = pyuni.encode('utf-8')
    # cdef char* retval = <char*>malloc((len(b) + 1) * sizeof(char))
    # retval[len(b)] = 0
    # return retval


cdef public void c_onehot(float* y, float* idxs, size_t n_idx):
    oh = nn.onehot(np.asarray(<float[:n_idx]>idxs), nc=len(nn.vocab))
    ensure_contiguous(oh)
    memcpy(y, PyArray_DATA(oh), oh.size * sizeof(float))
    # eprint(np.argmax(oh, axis=1))


cdef public void c_slices(float* X, float* idxs, size_t bs, size_t win):
    X_np = np.asarray(<float[:bs,:2*win]>X)
    idxs_np = np.asarray(<float[:bs + 2*win]>idxs)
    for r in range(bs):
        X_np[r, :win] = idxs_np[r:r+win]
        X_np[r, win:] = idxs_np[r+win+1:r+win+1+win]
    # eprint(X_np)


cdef public void debug_print(object o):
    eprint(o)


cdef public object create_network(int win, int embed):
    return nn.create_cbow_network(win, embed)


cdef public void set_net_weights(object net, WeightList* wl):
    net.set_weights(wrap_weight_list(wl))


cdef public void step_net(
    object net, float* X, float* y, size_t batch_size
):
    in_shape = (batch_size,) + net.input_shape[1:]
    out_shape = (batch_size,) + net.output_shape[1:]
    X_train = np.asarray(<float[:np.prod(in_shape)]>X).reshape(in_shape)
    y_train = np.asarray(<float[:np.prod(out_shape)]>y).reshape(out_shape),

    net.train_on_batch(X_train, y_train)


cdef public size_t out_size(object net):
    return np.prod(net.output_shape[1:])


cdef public float eval_net(object net):
    return net.evaluate(X_test, y_test, verbose=False)


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
    memcpy(X, PyArray_DATA(X_r), X_r.size * sizeof(float))
    memcpy(y, PyArray_DATA(y_r), y_r.size * sizeof(float))


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
        memcpy(wl.weights[i].shape, PyArray_DATA(sh), sh.size * sizeof(long))


cdef public void update_weightlist(WeightList* wl, object net):
    weights = net.get_weights()
    for i, w in enumerate(weights):
        w = w.astype(np.float32)

        assert w.flags['C_CONTIGUOUS']
        memcpy(wl.weights[i].W, PyArray_DATA(w), w.size * sizeof(float))


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


cdef void words_into_wordlist(WordList* wl, list words):
    if wl.mem < len(words):
        old = wl.mem
        wl.mem = len(words)
        wl.words = <Word*>realloc(wl.words, wl.mem * sizeof(Word))
        for i in range(old, wl.mem):
            wl.words[i].mem = 0
            wl.words[i].data = <char*>0

    wl.n_words = len(words)
    for i, w in enumerate(words):
        wenc = w.encode('utf-8')
        if wl.words[i].mem < len(wenc) + 1:
            wl.words[i].mem = len(wenc) + 1
            wl.words[i].data = <char*>realloc(
                wl.words[i].data, wl.words[i].mem * sizeof(char)
            )
        memcpy(wl.words[i].data, <char*>wenc, len(wenc) * sizeof(char))
        wl.words[i].data[len(wenc)] = 0


def inspect_array(a):
    print(a.flags, flush=True)
    print(a.dtype, flush=True)
    print(a.sum(), flush=True)


def ensure_contiguous(a):
    assert a.flags['C_CONTIGUOUS']


def eprint(*args, **kwargs):
    return print(*args, flush=True, **kwargs)
