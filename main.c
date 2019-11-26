#include "library.h"

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define P_READER 0
#define P_SLAVE 1
#define P_MASTER 2

#define COMM 500
#define ITER 32
#define BS 32

typedef enum{
    DATA,
    SLAVE,
    MASTER
} Role;

void data_reader() {
    // Reads some data and converts it to a float array
    printf("Start reader\n");
    size_t batch_numel = (784 + 10) * BS;
    float* batch = malloc(batch_numel * sizeof(float));
    while (1) {
        mnist_batch(batch, BS);
        MPI_Send(batch, batch_numel, MPI_FLOAT, P_SLAVE, 0, MPI_COMM_WORLD);
    }
    free(batch);
}

void send_network(const Network* c_net, int dest, int tag) {
    // Send a network to the expecting destination
    // It's best to receive with `recv_network`
    size_t n_layers = c_net->n_layers;
    MPI_Send(&n_layers, 1, MPI_LONG, dest, tag, MPI_COMM_WORLD);
    for (size_t i = 0; i < n_layers; i++) {
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
    // Creates a new network at c_net (all pointers will be lost so beware)
    MPI_Recv(&c_net->n_layers, 1, MPI_LONG, src, tag, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
    c_net->layers = malloc(sizeof(Dense) * c_net->n_layers);
    for (size_t i = 0; i < c_net->n_layers; i++) {
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
    // Cleans up the net
    for (size_t i = 0; i < c_net->n_layers; i++) {
        if (c_net->layers[i].ownmem) {
            free(c_net->layers[i].b);
            free(c_net->layers[i].W);
        }
    }
    free(c_net->layers);
    c_net->layers = NULL;  // So that you don't get any ideas
}

// Receives weight updates and trains, sends learned weights back to master
void slave_node() {
    printf("Start slave\n");
    Network net;
    create_c_network(&net);

    size_t batch_numel = (784 + 10) * BS;
    float* batch = malloc(batch_numel * sizeof(float));

    for (int i = 0; i < COMM; i++) {
        char go;
        MPI_Recv(&go, 1, MPI_CHAR, P_MASTER, MPI_ANY_TAG, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
        for (int k = 0; k < ITER; k++) {
            MPI_Recv(batch, batch_numel, MPI_FLOAT, P_READER, MPI_ANY_TAG,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            step_net(&net, batch, BS);
        }
        printf("Net: %f\n", eval_net(&net));
        send_network(&net, P_MASTER, 0);
    }

    free(batch);
    free_network_contents(&net);
}

// Stores most up-to-date model, sends it to slaves for training
void master_node() {
    printf("Start master\n");
    Network frank;
    create_c_network(&frank);
    for (int i = 0; i < COMM; i++) {
        char go;
        MPI_Send(&go, 1, MPI_CHAR, P_SLAVE, 0, MPI_COMM_WORLD);
        Network net;
        recv_network(&net, P_SLAVE, MPI_ANY_TAG);
        frankenstein(&frank, &net, 1);
        free_network_contents(&net);
        printf("Frank: %f\n", eval_net(&frank));
    }
    free_network_contents(&frank);
}

Role map_node() {
    int node;
    MPI_Comm_rank(MPI_COMM_WORLD, &node);
    if (node == P_READER) return DATA;
    if (node == P_MASTER) return MASTER;
    if (node == P_SLAVE) return SLAVE;

    exit(1);  // this is bad
}

int main (int argc, const char **argv) {
    MPI_Init(NULL, NULL);

    // Cython Boilerplate
    PyImport_AppendInittab("library", PyInit_library);
    Py_Initialize();
    PyRun_SimpleString("import sys\nsys.path.insert(0,'')");
    PyObject* library_module = PyImport_ImportModule("library");

    // Actual Code
    switch (map_node()) {
        case DATA: data_reader(); break;
        case SLAVE: slave_node(); break;
        case MASTER: master_node(); break;
    }

    // Finalizing Boilerplate
    Py_DECREF(library_module);
    Py_Finalize();
    MPI_Finalize();
}
