import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from mynet import onehot


HERE = os.path.abspath(os.path.dirname(__file__))
DATA = os.path.join(HERE, 'data')
CORPUS = os.path.join(DATA, 'corpus.txt')
VOCAB = os.path.join(DATA, 'vocab.txt')
TEST = os.path.join(DATA, 'test.txt')

vocab = {
    w: i for i, w in enumerate(open(VOCAB).read().splitlines(keepends=False))
}
inv_vocab = sorted(vocab, key=vocab.get)


def word_tokenize(s: str):
    l = ''.join(c.lower() if c.isalpha() else ' ' for c in s)
    return l.split()


def create_test_dataset(win):
    import numpy as np
    test_dataset = np.vectorize(vocab.get)(np.genfromtxt(TEST, dtype=str))
    assert test_dataset.shape[1] == 2*win + 1
    X_test = test_dataset[:, [*range(0, win), *range(win+1, win+win+1)]]
    y_test = onehot(test_dataset[:, win], nc=len(vocab))
    return X_test, y_test


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


def create_cbow_network(win, embed):
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # STFU!
    tf.random.set_random_seed(42)

    ctxt = tf.keras.layers.Input(shape=[2*win])
    ed = tf.keras.layers.Embedding(len(vocab), embed, input_length=2*win)(ctxt)
    cbow = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(ed)
    blowup = tf.keras.layers.Dense(len(vocab), activation='softmax')(cbow)
    mod = tf.keras.Model(inputs=ctxt, outputs=blowup)
    mod.compile(
        optimizer='adam',
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
