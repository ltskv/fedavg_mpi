cimport numpy as np
import numpy as np

from sys import stderr

from libc.stdlib cimport malloc, realloc
from libc.string cimport memcpy

import library as nn


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


cdef public size_t getwin():
    return nn.WIN


cdef public size_t getemb():
    return nn.EMB


cdef public size_t getbs():
    return nn.CFG['bs']


cdef public size_t getbpe():
    return nn.CFG['bpe']


cdef public float gettarget():
    return nn.CFG['target']


cdef public size_t getvocsize():
    return len(nn.vocab)


cdef public int get_tokens(WordList* wl, const char *filename):
    fnu = filename.decode('utf-8')
    if fnu not in tokenizers:
        tokenizers[fnu] = nn.token_generator(fnu)
    g = tokenizers[fnu]
    try:
        words = next(g)
    except StopIteration:
        eprint(f'Text {fnu} depleted, restarting...')
        tokenizers[fnu] = nn.token_generator(fnu)
        g = tokenizers[fnu]
        words = next(g)
    words_into_wordlist(wl, words)
    return 1


cdef public long vocab_idx_of(Word* w):
    word = w.data.decode('utf-8')
    try:
        return nn.vocab[word]
    except KeyError:
        return -1


cdef public void _dbg_idx_list_to_print(long* f_idxs, size_t num):
    idxs = np.asarray(<long[:num]>f_idxs)
    cdef str pyuni = ' '.join(nn.inv_vocab[i] for i in idxs)
    eprint(pyuni)


cdef public void _dbg_print(object o):
    eprint(o)


cdef public void _dbg_print_cbow_batch(float* batch):
    X_np, y_np = cbow_batch(batch)
    eprint(X_np)
    eprint(y_np)


cdef public void randidx(int* idx, size_t l, size_t how_much):
    i_np = np.random.choice(l, how_much, replace=False).astype(np.intc)
    memcpy(idx, PyArray_DATA(i_np), how_much * sizeof(int))


cdef public object create_network():
    try:
        net = nn.create_cbow_network()
        return net
    except Exception as e:
        eprint(e)


cdef public void set_net_weights(object net, WeightList* wl):
    net.set_weights(wrap_weight_list(wl))


cdef public void step_net(object net, float* batch):
    X_train, y_train = cbow_batch(batch)
    net.train_on_batch(X_train, y_train)


cdef public size_t out_size(object net):
    return np.prod(net.output_shape[1:])


cdef public float eval_net(object net):
    return nn.eval_network(net)


cdef public void ckpt_net(object net):
    nn.ckpt_network(net)


cdef public void save_emb(object net):
    nn.save_embeddings(nn.get_embeddings(net))


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
    """Not a one-liner anymore \_(".)_/"""
    alpha = 1. / num_weights
    frank = wrap_weight_list(wl_frank)
    for w in frank:
        w[:] = 0
    for i in range(num_weights):
        for wf, ww in zip(frank, wrap_weight_list(&wls[i])):
            wf += alpha * ww


cdef tuple cbow_batch(float* batch):
    win = getwin()
    bs = getbs()
    batch_np = np.asarray(<float[:bs,:2*win+1]>batch)
    X_np = batch_np[:, [*range(win), *range(win+1, win+win+1)]]
    y_np = nn.onehot(batch_np[:, win], nc=len(nn.vocab))
    return X_np, y_np


cdef list wrap_weight_list(WeightList* wl):
    """Thinly wraps a WeightList struct into a list of NumPy arrays."""
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
            wl.words[i].data = NULL

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
    return print(*args, flush=True, file=stderr, **kwargs)
