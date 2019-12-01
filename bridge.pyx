cimport numpy as np
import numpy as np

from sys import stderr

from libc.stdlib cimport malloc, realloc
from libc.string cimport memcpy

import library as nn


tokenizers = {}
X_test = None
y_test = None


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


cdef public void debug_print(object o):
    eprint(o)


cdef public object create_network(int win, int embed):
    try:
        return nn.create_cbow_network(win, embed)
    except Exception as e:
        eprint(e)


cdef public void set_net_weights(object net, WeightList* wl):
    net.set_weights(wrap_weight_list(wl))


cdef public void step_net(
    object net, float* batch, size_t bs
):
    X_train, y_train = cbow_batch(net, batch, bs)
    net.train_on_batch(X_train, y_train)


cdef public size_t out_size(object net):
    return np.prod(net.output_shape[1:])


cdef public float eval_net(object net):
    try:
        return net.evaluate(X_test, y_test, verbose=False)
    except Exception as e:
        eprint(e)


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


cdef public void create_test_dataset(size_t win):
    _create_test_dataset(win)


cdef tuple cbow_batch(
    object net, float* batch, size_t bs
):
    win = net.input_shape[1] // 2
    batch_np = np.asarray(<float[:bs,:2*win+1]>batch)
    X_np = np.concatenate([batch_np[:, :win], batch_np[:, win+1:]], axis=1)
    y_np = nn.onehot(batch_np[:, win], nc=len(nn.vocab))
    return X_np, y_np


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


def _create_test_dataset(win):
    global X_test, y_test
    if X_test is None or y_test is None:
        X_test, y_test = nn.create_test_dataset(win)
