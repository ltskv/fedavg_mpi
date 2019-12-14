# Implementation of Federated Averaging with MPI, Keras and Cython

(_for educational purposes_)

## What's it doing?

The system implemented in this project learns word embeddings with CBOW
approach, and furthermore, tries to do it in a distributed fashion. There are
two flavors of distribution present here:

1. Reading tokens (words) from a source (a text file for now), filtering and
   looking up vocabulary indices for words, windowing and batching are all
implemented in separate processes and form an *input pipeline*.
2. Neural Network training is done in parallel across several nodes
   (*learners*), with the learned weights periodically gathered, averaged and
   distributed by the central node, a.k.a. *dispatcher*. 

In this framework each learner can have its own input pipeline or all learners
can tap a single input pipeline or something in between can also work. It's not
possible in current version for one learner to tap more than one pipeline
though.

## How to make this work

### Requirements

* A recent UNIX-y system
* A recent GCC (default macOS clang also seems to work)
* MPICH 3
* Python 3.6 with dev headers and libraries (e.g. `python3-dev` on Ubuntu)
* Meson and ninja for building
* TensorFlow 1.14
* flask
* Cython

### Compiling

Compilation is supposed to be as simple as: (run in project root)

```sh
meson build && cd build && ninja
```

If this fails then either fix it yourself or let me know I guess.

### Running

Now this isn't without some quirks (due to this being a course project and
all). First you have to run *FROM PROJECT ROOT* using the following command
(don't run it yet as there are more instructions coming):

```sh
mpiexec -n NUM_PROC ./build/fedavg_mpi /path/to/training/data/textfile{1,2,3}
```

This program **expects a couple of things**:

First, **in the project root** create a directory `data` and put in there
the following three files:
- `vocab.txt` -- a whitespace-separated list of words, for which the embeddings
  will be learned. The words can only contain lowercase alphabetic ASCII chars
(you can try lowercase UTF-8 and see what happens but no guarantees here).
- `test.txt` -- a testing dataset with context windows of size 5, one line per
  window. The central (so third) word in the context window will be used as the
target and the surrounding words as the source. The same requirements apply
here as for the vocabulary, and furthermore only words present in the
`vocab.txt` are allowed in `test.txt`. This file will be used to track the loss
of the network during training. An example of the `test.txt` format.

```
the quick brown fox jumped
over a lazy dog padword
```

There also needs to be a file `cfg.json` **in the project root** containing the
following fields:
    
* `"data"`: `some_name` -- the name of the directory in which you put
`vocab.txt` and `test.txt`;
* `"bpe"`: Number of independent learner SGD iterations per communication
  round;
* `"bs"`: batch size (the number of context windows in a batch);
* `"target"`: The float value for the loss that you want to achieve, once the
network reaches this loss it will stop training, save the embeddings and exit.

Then, for each training data file passed as an argument (these can reside
wherever you want them to), an input pipeline will be constructed in the
program. There are 3 nodes in the input pipeline (tokenizer, filter, batcher).
Then there's this rule that one learner isn't allowed to tap more than one
pipeline, so each pipeline will need at least one learner. There also needs to
be a dispatcher process and a visualizer process.

**TLDR:** The formula for the number of processes that you need to request from
`mpiexec -n` looks like this:

```
NUM_PROCS >= 4*num_data_files + 2
```

There is also a convenient (well, somewhat) formula to determine how many
learners you will get depending on the arguments you passed:

```
learners = NUM_PROCS - 2 - 3*num_data_files
```

The good thing is, the program will complain if it doesn't like the numbers you
passed it and tell you how to fix it. 
