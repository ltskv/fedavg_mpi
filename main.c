#include <Python.h>
#include <stdio.h>
#include <mpi.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "library.h"

#define P_READER 0
#define P_SLAVE 1
#define P_MASTER 2

#define COMM 50
#define ITER 32
#define BS 32

typedef enum{
    DATA,
    SLAVE,
    MASTER
} Role;


// Reads some data and converts it to 2D float array
void data_reader() {
    while (1) {
        PyArrayObject* batch = mnist_batch(BS);

        long* shape = PyArray_SHAPE(batch);
        MPI_Send(shape, 2, MPI_LONG, P_SLAVE, 0, MPI_COMM_WORLD);
        MPI_Send(PyArray_DATA(batch), PyArray_SIZE(batch), MPI_FLOAT,
                P_SLAVE, 0, MPI_COMM_WORLD);
        Py_DECREF(batch);
    }
}

void send_network(Network* c_net, int dest, int tag) {
    Py_ssize_t n_layers = c_net->n_layers;
    MPI_Send(&n_layers, 1, MPI_LONG, dest, tag, MPI_COMM_WORLD);
    for (Py_ssize_t i = 0; i < n_layers; i++) {
        long d0 = c_net->layers[i].shape[0];
        long d1 = c_net->layers[i].shape[1];

        MPI_Send(c_net->layers[i].shape, 2, MPI_LONG, dest, tag,
                MPI_COMM_WORLD);
        MPI_Send(c_net->layers[i].W, d0 * d1, MPI_FLOAT, dest, tag,
                MPI_COMM_WORLD);
        MPI_Send(c_net->layers[i].b, d1, MPI_FLOAT, dest, tag,
                MPI_COMM_WORLD);
    }
}

void recv_network(Network* c_net, int src, int tag) {
    MPI_Recv(&c_net->n_layers, 1, MPI_LONG, src, tag, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
    c_net->layers = malloc(sizeof(Dense) * c_net->n_layers);
    for (Py_ssize_t i = 0; i < c_net->n_layers; i++) {
        MPI_Recv(&c_net->layers[i].shape, 2, MPI_LONG, src, tag,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        long d0 = c_net->layers[i].shape[0];
        long d1 = c_net->layers[i].shape[1];
        c_net->layers[i].ownmem = 1;
        c_net->layers[i].W = malloc(sizeof(float) * d0 * d1);
        c_net->layers[i].b = malloc(sizeof(float) * d1);
        MPI_Recv(c_net->layers[i].W, d0 * d1, MPI_FLOAT, src, tag,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(c_net->layers[i].b, d1, MPI_FLOAT, src, tag,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void free_network_contents(Network* c_net) {
    for (Py_ssize_t i = 0; i < c_net->n_layers; i++) {
        if (c_net->layers[i].ownmem) {
            free(c_net->layers[i].b);
            free(c_net->layers[i].W);
        }
    }
    free(c_net->layers);
}

// Receives weight updates and trains, sends learned weights back to master
void slave_node() {
    PyObject* net = create_network();
    for (int i = 0; i < COMM; i++) {
        char go;
        MPI_Recv(&go, 1, MPI_CHAR, P_MASTER, MPI_ANY_TAG, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
        for (int k = 0; k < ITER; k++) {
            long shape[2];
            MPI_Recv(shape, 2, MPI_LONG, P_READER, MPI_ANY_TAG, MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
            long size = shape[0] * shape[1];
            float* batch = malloc(shape[0] * shape[1] * sizeof(float));
            MPI_Recv(batch, size, MPI_FLOAT, P_READER, MPI_ANY_TAG,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            step_net(net, batch, BS);
            free(batch);
        }
        Network c_net;
        cify_network(net, &c_net);
        send_network(&c_net, P_MASTER, 0);
        free_network_contents(&c_net);
    }
    Py_DECREF(net);
}

// Stores most up-to-date model, sends it to slaves for training
void master_node() {
    PyObject* frank = create_network();
    for (int i = 0; i < COMM; i++) {
        char go;
        MPI_Send(&go, 1, MPI_CHAR, P_SLAVE, 0, MPI_COMM_WORLD);
        Network c_net;
        recv_network(&c_net, P_SLAVE, MPI_ANY_TAG);
        be_like_cified(frank, &c_net);
        free_network_contents(&c_net);
        printf("Frank: %f\n", eval_net(frank));
    }
    Py_DECREF(frank);
}

Role map_node() {
    int node;
    MPI_Comm_rank(MPI_COMM_WORLD, &node);
    if (node == 0) return DATA;
    if (node == 1) return SLAVE;
    if (node == 2) return MASTER;
    return SLAVE;
}

int main (int argc, const char **argv) {
    MPI_Init(NULL, NULL);

    // Cython Boilerplate
    PyImport_AppendInittab("library", PyInit_library);
    Py_Initialize();
    import_array();
    PyRun_SimpleString("import sys\nsys.path.insert(0,'')");
    PyObject* library_module = PyImport_ImportModule("library");

    // Actual Code
    switch (map_node()) {
        case DATA: data_reader();
                   break;
        case SLAVE: slave_node();
                    break;
        case MASTER: master_node();
                     break;
    }

    // Finalizing Boilerplate
    Py_DECREF(library_module);
    Py_Finalize();
    MPI_Finalize();
}
