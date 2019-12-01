import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # STFU!
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
from mynet import load_mnist, onehot


def word_tokenize(s: str):
    l = ''.join(c if c.isalpha() else ' ' for c in s)
    return l.split()


HERE = os.path.abspath(os.path.dirname(__file__))
CORPUS = os.path.join(HERE, 'melville-moby_dick.txt')
# sw = set(stopwords.words('english'))
sw = ['the']
vocab = list(set(
    w.lower() for w in word_tokenize(open(CORPUS).read())
    if w.isalpha() and not w.lower() in sw
))


def create_mnist_network():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(30, input_shape=(784,), activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='sgd',
                  metrics=['accuracy'])
    return model


def create_cbow_network(win, vocab, embed):
    ctxt = tf.keras.layers.Input(shape=[win])
    ed = tf.keras.layers.Embedding(vocab, embed, input_length=win)(ctxt)
    avgd = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(ed)
    mod = tf.keras.Model(inputs=ctxt, outputs=avgd)
    mod.compile(
        optimizer='sgd',
        loss='categorical_crossentropy',
    )
    return mod


def token_generator(filename):
    with open(filename) as f:
        for l in f.readlines(500):
            if not l.isspace():
                tok = word_tokenize(l)
                if tok:
                    yield tok
