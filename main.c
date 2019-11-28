#include "cythoned/library.h"

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define TAG_IDGAF 0
#define TAG_BATCH 1
#define TAG_NETWK 2
#define TAG_WEIGH 3
#define TAG_READY 4

#define COMM 500
#define ITER 120
#define BS 50
#define FSPC 0.4

#define in_range(i, x) (size_t (i) = 0; (i) < (x); (i)++)
// I am honestly VERY sorry for this but power corrupts even the best of us

#define INFO_PRINTF(fmt, ...) \
    do { fprintf(stderr, fmt, __VA_ARGS__); } while(0)
#define INFO_PRINTLN(what) \
    do { fprintf(stderr, "%s\n", what); } while(0)


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

int number_of_nodes() {
    int n;
    MPI_Comm_size(MPI_COMM_WORLD, &n);
    return n;
}

int number_of_masters() {
    return 1;
}

int number_of_readers() {
    return 1;
}

int number_of_slaves() {
    return number_of_nodes() - number_of_masters() - number_of_readers();
}

int my_id() {
    int i;
    MPI_Comm_rank(MPI_COMM_WORLD, &i);
    return i;
}

int master_id(int m) {
    return m;
}

int reader_id(int r) {
    return r + number_of_masters();
}

int slave_id(int s) {
    return s + number_of_masters() + number_of_readers();
}

Role map_node() {
    int node;
    MPI_Comm_rank(MPI_COMM_WORLD, &node);
    if (node >= reader_id(0) && node <= reader_id(number_of_readers()-1)) {
        return DATA;
    }
    if (node >= master_id(0) && node <= master_id(number_of_masters()-1)) {
        return MASTER;
    }
    if (node >= slave_id(0) && node <= slave_id(number_of_slaves()-1)) {
        return SLAVE;
    }
    exit(1);  // this is bad
}

int rid(int id, Role what) {
    int z;
    switch (what) {
        case DATA: z = reader_id(0); break;
        case SLAVE: z = slave_id(0); break;
        case MASTER: z = master_id(0); break;
    }
    return id - z;
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
        mnist_batch(batch, BS, rid(s, SLAVE), number_of_slaves());
        MPI_Send(batch, batch_numel, MPI_FLOAT, s, TAG_BATCH, MPI_COMM_WORLD);
    }
    free(batch);
}

void send_weights(const Network* c_net, int dest, int tag) {
    // This assumes that the receiving end has a fully initialized network
    // Of the same arch as `c_net`
    for in_range(i, c_net->n_layers) {
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
    for in_range(i, c_net->n_layers) {
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
    for in_range(i, c_net->n_layers) {
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
    for in_range(i, c_net->n_layers) {
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
    for in_range(i, c_net->n_layers) {
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

void slave_node() {
    // 0. Announce readiness?
    // 1. Receive weights from master ([ ] has to know its master)
    // 2. Request batch from reader ([ ] has to choose a reader)
    // 3. Do computations
    // 4. Send weights back to master
    printf("Start slave\n");

    int me;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    size_t batch_numel = (784 + 10) * BS;
    float* batch = malloc(batch_numel * sizeof(float));
    Network net;
    create_c_network(&net);

    for in_range(i, COMM) {
        // INFO_PRINTF("%d announcing itself\n", my_id());
        MPI_Send(&me, 1, MPI_INT, master_id(0), TAG_READY, MPI_COMM_WORLD);
        // INFO_PRINTF("%d waitng for weights from %d\n", my_id(), master_id(0));
        recv_weights(&net, master_id(0), TAG_WEIGH);
        // INFO_PRINTF("%d an answer!\n", my_id());
        for in_range(k, ITER) {
            MPI_Send(&me, 1, MPI_INT, reader_id(0), TAG_READY, MPI_COMM_WORLD);
            MPI_Recv(batch, batch_numel, MPI_FLOAT, reader_id(0), TAG_BATCH,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            step_net(&net, batch, BS);
        }
        printf("%d net: %f\n", my_id(), eval_net(&net));
        send_weights(&net, master_id(0), TAG_WEIGH);
    }
    free_network_contents(&net);
    free(batch);
}

void master_node() {
    // 0. Initialize model

    // 1. Send it to some slaves for processing (synchronous)
    // 2. Receive weights back (synchronous)
    // 3. Average the weights

    printf("Start master\n");

    Network frank;
    create_c_network(&frank);

    int spr = number_of_slaves() * FSPC;  // Slaves per round
    int s;

    Network *nets = malloc(sizeof(Network) * spr);
    int *handles = malloc(sizeof(int) * spr);

    for in_range(i, spr) create_c_network(nets + i);
    for in_range(i, COMM) {

        for in_range(k, spr) {
            MPI_Recv(&s, 1, MPI_INT, MPI_ANY_SOURCE, TAG_READY, MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
            send_weights(&frank, s, TAG_WEIGH);
            handles[k] = s;
        }
        for in_range(k, spr) {
            recv_weights(nets + k, handles[k], TAG_WEIGH);
        }
        combo_c_net(&frank, nets, spr);
        printf("Frank: %f\n", eval_net(&frank));
    }
    free_network_contents(&frank);
    free(nets);
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
