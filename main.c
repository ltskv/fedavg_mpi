#include "cythoned/library.h"

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define TAG_IDGAF 0
#define TAG_BATCH 1
#define TAG_NETWK 2
#define TAG_WEIGH 3
#define TAG_READY 4
#define TAG_BREAK 5

#define COMM 100
#define ITER 20
#define BS 50
#define FSPC 1

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

void free_weightlist(WeightList* wl) {
    for in_range(i, wl->n_weights) {
        free(wl->weights[i].shape);
        free(wl->weights[i].W);
    }
    free(wl->weights);
}

void data_reader() {
    // Reads some data and converts it to a float array
    INFO_PRINTF("Starting reader %d\n", getpid());

    size_t X_numel = 784 * BS;
    size_t y_numel = 10 * BS;
    float* X = malloc(X_numel * sizeof(float));
    float* y = malloc(y_numel * sizeof(float));
    int s = 0;

    while (s != -1) {
        MPI_Recv(&s, 1, MPI_INT, MPI_ANY_SOURCE, TAG_READY, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
        if (s != -1) {
            mnist_batch(X, y, BS, rid(s, SLAVE), number_of_slaves());
            MPI_Send(X, X_numel, MPI_FLOAT, s, TAG_BATCH, MPI_COMM_WORLD);
            MPI_Send(y, y_numel, MPI_FLOAT, s, TAG_BATCH, MPI_COMM_WORLD);
        }
    }
    free(X);
    free(y);
}

void send_weights(const WeightList* wl, int dest, int tag) {
    // This assumes that the receiving end knows exactly
    // the number of elements being sent and has memory ready
    // for it.
    for in_range(i, wl->n_weights) {
        long n_el = 1;
        for in_range(k, wl->weights[i].dims) {
            n_el *= wl->weights[i].shape[k];
        }
        MPI_Send(wl->weights[i].W, n_el, MPI_FLOAT, dest, tag, MPI_COMM_WORLD);
    }
}

void recv_weights(WeightList* wl, int src, int tag) {
    // This assumes that the sender sends stuff that is going
    // to fit into memory in correct order too.
    for in_range(i, wl->n_weights) {
        long n_el = 1;
        for in_range(d, wl->weights[i].dims) {
            n_el *= wl->weights[i].shape[d];
        }
        MPI_Recv(wl->weights[i].W, n_el, MPI_FLOAT, src, tag, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
    }
}

void slave_node() {
    // 0. Announce readiness?
    // 1. Receive weights from master ([ ] has to know its master)
    // 2. Request batch from reader ([ ] has to choose a reader)
    // 3. Do computations
    // 4. Send weights back to master
    INFO_PRINTF("Starting slave %d\n", getpid());

    int me;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    size_t X_numel = 784 * BS;
    size_t y_numel = 10 * BS;
    float* X = malloc(X_numel * sizeof(float));
    float* y = malloc(y_numel * sizeof(float));

    PyObject* net = create_network();
    WeightList wl;
    init_weightlist_like(&wl, net);

    for in_range(i, COMM) {
        MPI_Send(&me, 1, MPI_INT, master_id(0), TAG_READY, MPI_COMM_WORLD);
        recv_weights(&wl, master_id(0), TAG_WEIGH);
        set_net_weights(net, &wl);
        for in_range(k, ITER) {
            MPI_Send(&me, 1, MPI_INT, reader_id(0), TAG_READY, MPI_COMM_WORLD);
            MPI_Recv(X, X_numel, MPI_FLOAT, reader_id(0), TAG_BATCH,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(y, y_numel, MPI_FLOAT, reader_id(0), TAG_BATCH,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            step_net(net, X, y, BS);
        }
        printf("%d net: %f\n", my_id(), eval_net(net));
        update_weightlist(&wl, net);
        send_weights(&wl, master_id(0), TAG_WEIGH);
    }
    Py_DECREF(net);
    free_weightlist(&wl);
}

void master_node() {
    // 0. Initialize model

    // 1. Send it to some slaves for processing (synchronous)
    // 2. Receive weights back (synchronous)
    // 3. Average the weights


    PyObject* frank = create_network();
    WeightList wl;
    init_weightlist_like(&wl, frank);
    update_weightlist(&wl, frank);

    int spr = number_of_slaves() * FSPC;  // Slaves per round
    int s;

    WeightList *wls = malloc(sizeof(WeightList) * spr);
    int *handles = malloc(sizeof(int) * spr);

    for in_range(i, spr) {
        init_weightlist_like(wls + i, frank);
    }
    for in_range(i, COMM) {

        for in_range(k, spr) {
            MPI_Recv(&s, 1, MPI_INT, MPI_ANY_SOURCE, TAG_READY, MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
            send_weights(&wl, s, TAG_WEIGH);
            handles[k] = s;
        }
        for in_range(k, spr) {
            recv_weights(wls + k, handles[k], TAG_WEIGH);
        }
        combo_weights(&wl, wls, spr);
        set_net_weights(frank, &wl);
        printf("Frank: %f\n", eval_net(frank));
    }
    Py_DECREF(frank);
    free_weightlist(&wl);
    for in_range(i, spr) free_weightlist(wls + i);
    free(wls);
    if (rid(my_id(), MASTER) == 0) {
        for in_range(r, number_of_readers()) {
            int stop = -1;
            MPI_Send(&stop, 1, MPI_INT, reader_id(r), TAG_READY,
                    MPI_COMM_WORLD);
        }
    }
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
