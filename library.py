import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # STFU!

from mynet import onehot


HERE = os.path.abspath(os.path.dirname(__file__))
CORPUS = os.path.join(HERE, 'melville-moby_dick.txt')
VOCAB = os.path.join(HERE, 'vocab.txt')

vocab = {
    w: i for i, w in enumerate(open(VOCAB).read().splitlines(keepends=False))
}
inv_vocab = sorted(vocab, key=vocab.get)


def word_tokenize(s: str):
    l = ''.join(c.lower() if c.isalpha() else ' ' for c in s)
    return l.split()


def create_test_dataset(win):
    S = 1000
    with open(CORPUS) as f:
        ds = np.array([vocab[w] for w in word_tokenize(f.read())
                       if w in vocab])
    idx = np.random.choice(np.arange(win, len(ds) - win), S)
    return (
        # X
        np.stack([
            np.concatenate([ds[i-win:i], ds[i+1:i+win+1]])
            for i in idx
        ], axis=0).astype(np.float32),

        #y
        onehot(ds[idx], nc=len(vocab))
    )

def create_mnist_network():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(30, input_shape=(784,), activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='sgd',
                  metrics=['accuracy'])
    return model


def create_cbow_network(win, embed):
    ctxt = tf.keras.layers.Input(shape=[2*win])
    ed = tf.keras.layers.Embedding(len(vocab), embed, input_length=2*win)(ctxt)
    cbow = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(ed)
    blowup = tf.keras.layers.Dense(len(vocab), activation='softmax')(cbow)
    mod = tf.keras.Model(inputs=ctxt, outputs=blowup)
    mod.compile(
        optimizer='sgd',
        loss='categorical_crossentropy',
    )
    return mod


def token_generator(filename):
    with open(filename) as f:
        for i, l in enumerate(f.readlines()):
            if not l.isspace():
                tok = word_tokenize(l)
                if tok:
                    yield tok
