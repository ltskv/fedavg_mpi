#include "cythoned/library.h"

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define P_READER 0
#define P_MASTER 1
#define P_SLAVE 2

#define TAG_IDGAF 0
#define TAG_BATCH 1
#define TAG_NETWK 2
#define TAG_WEIGH 3
#define TAG_READY 4

#define COMM 500
#define ITER 40
#define BS 50
#define FSPC 0.2

#define sid(s) s + P_SLAVE

#define s_in_slaves(w) (size_t s = 0; s < w - P_SLAVE; s++)
#define i_in_range(x) (size_t i = 0; i < x; i++)
// I am honestly VERY sorry for this but power corrupts even the best of us

typedef enum{
    DATA,
    SLAVE,
    MASTER
} Role;


typedef struct IntQueue IntQueue;
struct IntQueue {
    int head;
    int tail;
    size_t size;
    int* data;
};

void queue_from_size(IntQueue* q, size_t s) {
    q->data = malloc(s * sizeof(int));
    q->size = s+1;
    q->head = 0;
    q->tail = 0;
}

void push_queue(IntQueue *q, int d) {
    // Assuming queue is not full
    q->data[q->tail] = d;
    q->tail = (q->tail + 1) % q->size;
}

int pop_queue(IntQueue *q) {
    int d = q->data[q->head];
    q->head = (q->head + 1) % q->size;
    return d;
}

int queue_empty(IntQueue *q) {
    return q->head == q->tail;
}

int queue_full(IntQueue *q) {
    return ((q->tail + 1) % q->size) == q->head;
}

void data_reader() {
    // Reads some data and converts it to a float array
    printf("Start reader\n");
    size_t batch_numel = (784 + 10) * BS;
    float* batch = malloc(batch_numel * sizeof(float));
    int s = 0;

    while (1) {
        MPI_Recv(&s, 1, MPI_INT, MPI_ANY_SOURCE, TAG_READY, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
        mnist_batch(batch, BS);
        MPI_Send(batch, batch_numel, MPI_FLOAT, s, TAG_BATCH, MPI_COMM_WORLD);
    }
    free(batch);
}

void send_weights(const Network* c_net, int dest, int tag) {
    // This assumes that the receiving end has a fully initialized network
    // Of the same arch as `c_net`
    for i_in_range(c_net->n_layers) {
        long d0 = c_net->layers[i].shape[0];
        long d1 = c_net->layers[i].shape[1];
        MPI_Send(c_net->layers[i].W, d0 * d1, MPI_FLOAT, dest, tag,
                MPI_COMM_WORLD);
        MPI_Send(c_net->layers[i].b, d1, MPI_FLOAT, dest, tag,
                MPI_COMM_WORLD);
    }
}

void recv_weights(const Network* c_net, int src, int tag) {
    // This assumes that the sender is going to send stuff that is going
    // To fit exactly into the c_net
    for i_in_range(c_net->n_layers) {
        long d0 = c_net->layers[i].shape[0];
        long d1 = c_net->layers[i].shape[1];
        MPI_Recv(c_net->layers[i].W, d0 * d1, MPI_FLOAT, src, tag,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(c_net->layers[i].b, d1, MPI_FLOAT, src, tag,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void send_network(const Network* c_net, int dest, int tag) {
    // Send a network to the expecting destination
    // It's best to receive with `recv_network`
    size_t n_layers = c_net->n_layers;
    MPI_Send(&n_layers, 1, MPI_LONG, dest, tag, MPI_COMM_WORLD);
    for i_in_range(c_net->n_layers) {
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
    // c_net HAS TO BE a fresh empty Network struct
    MPI_Recv(&c_net->n_layers, 1, MPI_LONG, src, tag, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
    c_net->layers = malloc(sizeof(Dense) * c_net->n_layers);
    for i_in_range(c_net->n_layers) {
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
    for i_in_range(c_net->n_layers) {
        if (c_net->layers[i].ownmem) {
            free(c_net->layers[i].b);
            free(c_net->layers[i].W);
        }
    }
    if (c_net->layers != NULL) {
        free(c_net->layers);
        c_net->layers = NULL;  // So that you don't get any ideas
    }
}

// Receives weight updates and trains, sends learned weights back to master
void slave_node() {
    printf("Start slave\n");

    int me;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    size_t batch_numel = (784 + 10) * BS;
    float* batch = malloc(batch_numel * sizeof(float));
    Network net;
    create_c_network(&net);

    for i_in_range(COMM) {
        MPI_Send(&me, 1, MPI_INT, P_MASTER, TAG_READY, MPI_COMM_WORLD);
        recv_weights(&net, P_MASTER, TAG_NETWK);
        for (int k = 0; k < ITER; k++) {
            MPI_Send(&me, 1, MPI_INT, P_READER, TAG_READY, MPI_COMM_WORLD);
            MPI_Recv(batch, batch_numel, MPI_FLOAT, P_READER, TAG_BATCH,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            step_net(&net, batch, BS);
        }
        printf("Net: %f\n", eval_net(&net));
        send_weights(&net, P_MASTER, TAG_WEIGH);
    }
    free_network_contents(&net);
    free(batch);
}

void master_node() {
    // Stores most up-to-date model, sends it to slaves for training
    // First do it synchronously
    // Need a "slave registry"
    printf("Start master\n");

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    Network frank;
    create_c_network(&frank);

    // It's better to have more memory than needed
    // Than less memory than needed
    // Kong Fuzi
    Network* nets = malloc(sizeof(Network) * world_size);
    for s_in_slaves(world_size) create_c_network(nets + s);

    IntQueue slave_queue;
    queue_from_size(&slave_queue, world_size - P_SLAVE);

    for i_in_range(COMM) {
        for s_in_slaves(world_size) {
            send_weights(&frank, sid(s), TAG_WEIGH);
        }
        for s_in_slaves(world_size) {
            recv_weights(nets + s, sid(s), TAG_WEIGH);
        }
        combo_c_net(&frank, nets, world_size - P_SLAVE);
        printf("Frank: %f\n", eval_net(&frank));
    }
    free_network_contents(&frank);
    free(nets);
}

Role map_node() {
    int node;
    MPI_Comm_rank(MPI_COMM_WORLD, &node);
    if (node == P_READER) return DATA;
    if (node == P_MASTER) return MASTER;
    if (node >= P_SLAVE) return SLAVE;

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
