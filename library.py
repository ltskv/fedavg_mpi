import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # STFU!
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
from mynet import load_mnist, onehot


def word_tokenize(s: str):
    l = ''.join(c.lower() if c.isalpha() else ' ' for c in s)
    return l.split()


HERE = os.path.abspath(os.path.dirname(__file__))
CORPUS = os.path.join(HERE, 'melville-moby_dick.txt')
VOCAB = os.path.join(HERE, 'vocab.txt')

vocab = {
    w: i for i, w in enumerate(open(VOCAB).read().splitlines(keepends=False))
}
# inv_vocab = [vocab[i] for i in range(len(vocab))]
inv_vocab = sorted(vocab, key=vocab.get)


def create_mnist_network():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(30, input_shape=(784,), activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='sgd',
                  metrics=['accuracy'])
    return model


def create_cbow_network(win, embed):
    ctxt = tf.keras.layers.Input(shape=[win])
    ed = tf.keras.layers.Embedding(len(vocab), embed, input_length=win)(ctxt)
    cbow = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(ed)
    blowup = tf.keras.layers.Dense(len(vocab), activation='softmax')(cbow)
    mod = tf.keras.Model(inputs=ctxt, outputs=blowup)
    mod.compile(
        optimizer='sgd',
        loss='categorical_crossentropy',
    )
    print(mod, flush=True)
    return mod


def token_generator(filename):
    with open(filename) as f:
        for i, l in enumerate(f.readlines()):
            if not l.isspace():
                tok = word_tokenize(l)
                if tok:
                    yield tok
