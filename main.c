#include "cythoned/bridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define TAG_IDGAF 0
#define TAG_BATCH 1
#define TAG_NETWK 2
#define TAG_WEIGH 3
#define TAG_READY 4
#define TAG_BREAK 5
#define TAG_STLEN 6
#define TAG_SWORD 7
#define TAG_IWORD 8
#define TAG_INSTR 9

#define COMM 50
#define ITER 50
#define BS 32
#define EMB 20
#define WIN 2
#define FLPC 0.8

#define in_range(i, x) (size_t (i) = 0; (i) < (x); (i)++)
// I am honestly VERY sorry for this but power corrupts even the best of us

#define INFO_PRINTF(fmt, ...) \
    do { fprintf(stderr, fmt, __VA_ARGS__); } while(0)
#define INFO_PRINTLN(what) \
    do { fprintf(stderr, "%s\n", what); } while(0)
#define INFO_PRINT(what) \
    do { fprintf(stderr, "%s", what); } while(0)

int g_argc = 1;

typedef enum{
    TOKENIZER,
    FILTERER,
    BATCHER,
    LEARNER,
    DISPATCHER
} Role;

int world_size() {
    int n;
    MPI_Comm_size(MPI_COMM_WORLD, &n);
    return n;
}

int my_mpi_id() {
    int i;
    MPI_Comm_rank(MPI_COMM_WORLD, &i);
    return i;
}

size_t number_of(Role what) {
    switch (what) {
        case TOKENIZER:
            if (g_argc < 2) {
                INFO_PRINTLN("NOT ENOUGH INPUTS!");
                exit(1);
            }
            return g_argc - 1;
        case FILTERER:
            return 1;
        case BATCHER:
            return 1;
        case LEARNER:
            return world_size()
                - number_of(TOKENIZER)
                - number_of(FILTERER)
                - number_of(BATCHER)
                - number_of(DISPATCHER);
        case DISPATCHER:
            return 1;
    }
}

int mpi_id_from_role_id(Role role, int rid) {
    int base = 0;
    for (Role r = TOKENIZER; r < role; r++) {
        base += number_of(r);
    }
    return rid + base;
}

int role_id_from_mpi_id(Role role, int mid) {
    int z = mpi_id_from_role_id(role, 0);
    int rid = mid - z;
    if (rid >= number_of(role) || rid < 0) {
        INFO_PRINTF("%d is not a %d\n", mid, role);
        exit(1);
    }
    return rid;
}

Role map_node() {
    int node = my_mpi_id();
    size_t base = 0;
    for (Role r = TOKENIZER; r <= DISPATCHER; r++) {
        if (node < number_of(r) + base) return r;
        base += number_of(r);
    }
    INFO_PRINTF("Something went wrong for node %d\n", node);
    exit(1);  // this is bad
}

void free_word(Word* w) {
    free(w->data);
    w->data = NULL;
    w->mem = 0;
}

void free_wordlist(WordList* wl) {
    for in_range(i, wl->mem) {
        free_word(wl->words + i);
    }
    free(wl->words);
    wl->words = NULL;
    wl->n_words = 0;
}

void send_word(Word* w, int dest) {
    long len = strlen(w->data);
    MPI_Send(&len, 1, MPI_LONG, dest, TAG_STLEN, MPI_COMM_WORLD);
    MPI_Send(w->data, len + 1, MPI_CHAR, dest, TAG_SWORD, MPI_COMM_WORLD);
}

int recv_word(Word* w, int src) {
    long len;
    MPI_Status stat;
    MPI_Recv(&len, 1, MPI_LONG, src, TAG_STLEN, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
    if (w->mem < len + 1) {
        w->mem = len + 1;
        w->data = realloc(w->data, sizeof(char) * w->mem);
    }
    MPI_Recv(w->data, len + 1, MPI_CHAR, src, TAG_SWORD, MPI_COMM_WORLD,
            &stat);
    return stat.MPI_SOURCE;
}

void tokenizer(const char* source) {
    WordList wl = {0, 0, NULL};
    while (get_tokens(&wl, source)) {
        for in_range(i, wl.n_words) {
            send_word(&wl.words[i], mpi_id_from_role_id(FILTERER, 0));
        }
    }
    Word terminator = {1, ""};
    send_word(&terminator, mpi_id_from_role_id(FILTERER, 0));
    free_wordlist(&wl);
}

void filterer() {
    Word w = {0, NULL};
    const size_t bufsize = 2 * WIN + 1;
    long* idx = malloc(bufsize * sizeof(long));
    size_t have = 0;
    while (1) {
        while (have < bufsize) {
            recv_word(&w, role_id_from_mpi_id(TOKENIZER, 0));
            if (!strlen(w.data)) break;
            idx[have] = vocab_idx_of(&w);
            if (idx[have] != -1) have++;
        }
        if (!strlen(w.data)) break;
        have = 0;
        MPI_Send(idx, bufsize, MPI_LONG, mpi_id_from_role_id(BATCHER, 0),
                TAG_IWORD, MPI_COMM_WORLD);
    }
    idx[0] = -1;
    MPI_Send(idx, bufsize, MPI_LONG, mpi_id_from_role_id(BATCHER, 0),
            TAG_IWORD, MPI_COMM_WORLD);
    free_word(&w);
    free(idx);
}

void batcher() {
    int s = 0;
    const size_t entry_size = 2 * WIN + 1;
    const size_t bufsize = BS * entry_size;
    float* batch = malloc(bufsize * sizeof(float));
    long* l_wid = malloc(entry_size * sizeof(long));

    while (1) {
        for in_range(r, BS) {
            MPI_Recv(l_wid, entry_size, MPI_LONG,
                    mpi_id_from_role_id(FILTERER, 0),
                    TAG_IWORD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (l_wid[0] == -1) break;
            for in_range(c, entry_size) {
                batch[r*entry_size + c] = (float)l_wid[c];
            }
        }
        if (l_wid[0] == -1) break;
        INFO_PRINT(".");
        MPI_Recv(&s, 1, MPI_INT, MPI_ANY_SOURCE, TAG_READY, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
        MPI_Send(batch, bufsize, MPI_FLOAT, s, TAG_BATCH, MPI_COMM_WORLD);
        INFO_PRINTLN("!");
    }
    free(l_wid);
    free(batch);
}

void free_weightlist(WeightList* wl) {
    for in_range(i, wl->n_weights) {
        free(wl->weights[i].shape);
        free(wl->weights[i].W);
    }
    free(wl->weights);
}

void send_weights(const WeightList* wl, int dest) {
    // This assumes that the receiving end knows exactly
    // the number of elements being sent and has memory ready
    // for it.
    for in_range(i, wl->n_weights) {
        long n_el = 1;
        for in_range(k, wl->weights[i].dims) {
            n_el *= wl->weights[i].shape[k];
        }
        MPI_Send(wl->weights[i].W, n_el, MPI_FLOAT, dest,
                TAG_WEIGH, MPI_COMM_WORLD);
    }
}

void recv_weights(WeightList* wl, int src) {
    // This assumes that the sender sends stuff that is going
    // to fit into memory in correct order too.
    for in_range(i, wl->n_weights) {
        long n_el = 1;
        for in_range(d, wl->weights[i].dims) {
            n_el *= wl->weights[i].shape[d];
        }
        MPI_Recv(wl->weights[i].W, n_el, MPI_FLOAT, src,
                TAG_WEIGH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void learner() {
    INFO_PRINTF("Starting slave %d\n", getpid());
    int me = my_mpi_id();

    PyObject* net = create_network(WIN, EMB);
    create_test_dataset(WIN);
    WeightList wl;
    init_weightlist_like(&wl, net);
    size_t entry_size = (2*WIN + 1);
    size_t bufsize = BS * entry_size;

    float* batch = malloc(bufsize * sizeof(float));

    for in_range(i, COMM) {
        recv_weights(&wl, mpi_id_from_role_id(DISPATCHER, 0));
        set_net_weights(net, &wl);
        for in_range(k, ITER) {
            MPI_Send(&me, 1, MPI_INT, mpi_id_from_role_id(BATCHER, 0),
                    TAG_READY, MPI_COMM_WORLD);
            MPI_Recv(batch, bufsize, MPI_FLOAT,
                    mpi_id_from_role_id(BATCHER, 0), TAG_BATCH, MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
            step_net(net, batch, BS);
        }
        // printf("%d net: %f\n", my_mpi_id(), eval_net(net));
        update_weightlist(&wl, net);
        send_weights(&wl, mpi_id_from_role_id(DISPATCHER, 0));
    }
    Py_DECREF(net);
    free_weightlist(&wl);
    free(batch);
}

void dispatcher() {
    PyObject* frank = create_network(WIN, EMB);
    create_test_dataset(WIN);
    WeightList wl;
    init_weightlist_like(&wl, frank);
    update_weightlist(&wl, frank);

    int lpr = number_of(LEARNER) * FLPC;  // Learners per round

    WeightList *wls = malloc(sizeof(WeightList) * lpr);
    int *round = malloc(sizeof(int) * lpr);

    for in_range(i, lpr) {
        init_weightlist_like(wls + i, frank);
    }
    for in_range(i, COMM) {
        randidx(round, number_of(LEARNER), lpr);

        for in_range(k, lpr) {
            // INFO_PRINTF(" %5d", round[k]);
            send_weights(&wl, mpi_id_from_role_id(LEARNER, round[k]));
        }
        // INFO_PRINTLN("");
        for in_range(k, lpr) {
            recv_weights(wls + k, mpi_id_from_role_id(LEARNER, round[k]));
        }
        combo_weights(&wl, wls, lpr);
        set_net_weights(frank, &wl);
        // printf("Frank: %f\n", eval_net(frank));
    }
    Py_DECREF(frank);
    free_weightlist(&wl);
    for in_range(i, lpr) free_weightlist(wls + i);
    free(wls);
    free(round);
}

int main (int argc, const char **argv) {
    MPI_Init(NULL, NULL);

    // Cython Boilerplate
    PyImport_AppendInittab("bridge", PyInit_bridge);
    Py_Initialize();
    PyRun_SimpleString("import sys\nsys.path.insert(0,'')");
    PyObject* bridge_module = PyImport_ImportModule("bridge");

    // Actual Code
    int role_id;
    g_argc = argc;
    switch (map_node()) {
        case TOKENIZER:
            role_id = role_id_from_mpi_id(TOKENIZER, my_mpi_id());
            tokenizer(argv[role_id + 1]);
            break;
        case FILTERER:
            filterer();
            break;
        case BATCHER:
            batcher();
            break;
        case LEARNER:
            learner();
            break;
        case DISPATCHER:
            dispatcher();
            break;
        default:
            INFO_PRINTLN("DYING HORRIBLY!");
    }

    // Finalizing Boilerplate
    Py_DECREF(bridge_module);
    Py_Finalize();
    MPI_Finalize();
}
