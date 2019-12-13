import os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from mynet import onehot


WIN = 2
EMB = 32

HERE = os.path.abspath(os.path.dirname(__file__))


def read_cfg():
    with open(os.path.join(HERE, 'cfg.json')) as f:
        return json.load(f)


CFG = read_cfg()
DATA = os.path.join(HERE, CFG['data'])
CORPUS = os.path.join(DATA, 'corpus.txt')
VOCAB = os.path.join(DATA, 'vocab.txt')
TEST = os.path.join(DATA, 'test.txt')


def read_vocab_list():
    with open(VOCAB) as f:
        return f.read().split()


inv_vocab = read_vocab_list()
vocab = {w: i for i, w in enumerate(inv_vocab)}

X_test = None
y_test = None


def word_tokenize(s: str):
    l = ''.join(c.lower() if c.isalpha() else ' ' for c in s)
    return l.split()


def create_test_dataset():
    import numpy as np
    test_dataset = np.vectorize(vocab.get)(np.genfromtxt(TEST, dtype=str))
    assert test_dataset.shape[1] == 2*WIN + 1

    global X_test, y_test
    X_test = test_dataset[:, [*range(0, WIN), *range(WIN+1, WIN+WIN+1)]]
    y_test = onehot(test_dataset[:, WIN], nc=len(vocab))


def create_mnist_network():
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # STFU!
    tf.random.set_random_seed(42)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(30, input_shape=(784,), activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


def create_cbow_network():
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # STFU!
    tf.random.set_random_seed(42)

    ctxt = tf.keras.layers.Input(shape=[2*WIN])
    ed = tf.keras.layers.Embedding(len(vocab), EMB, input_length=2*WIN)(ctxt)
    cbow = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(ed)
    blowup = tf.keras.layers.Dense(len(vocab), activation='softmax')(cbow)
    mod = tf.keras.Model(inputs=ctxt, outputs=blowup)
    mod.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
    )
    return mod


def eval_network(net):
    if X_test is None or y_test is None:
        create_test_dataset()
    return net.evaluate(X_test, y_test, verbose=False)


def token_generator(filename):
    with open(filename) as f:
        for l in f:
            if not l.isspace():
                tok = word_tokenize(l)
                if tok:
                    yield tok
