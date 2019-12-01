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

#define COMM 1
#define ITER 1000
#define BS 10
#define EMB 20
#define WIN 2
#define FSPC 1

#define in_range(i, x) (size_t (i) = 0; (i) < (x); (i)++)
// I am honestly VERY sorry for this but power corrupts even the best of us

#define INFO_PRINTF(fmt, ...) \
    do { fprintf(stderr, fmt, __VA_ARGS__); } while(0)
#define INFO_PRINTLN(what) \
    do { fprintf(stderr, "%s\n", what); } while(0)

// char_stream -> tokenize -> word_strem -> filter + batch -> slave network

typedef enum{
    TOKENIZER,
    FILTERER,
    BATCHER,
    SLAVE,
    MASTER
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
            return 1;
        case FILTERER:
            return 1;
        case BATCHER:
            return 1;
        case SLAVE:
            return world_size()
                - number_of(TOKENIZER)
                - number_of(FILTERER)
                - number_of(BATCHER)
                - number_of(MASTER);
        case MASTER:
            return 0;
#warning "set to real number of masters!"
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
    for (Role r = TOKENIZER; r <= MASTER; r++) {
        if (node < number_of(r) + base) return r;
        base += number_of(r);
    }
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

void recv_word(Word* w, int src) {
    long len;
    MPI_Recv(&len, 1, MPI_LONG, src, TAG_STLEN, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
    if (w->mem < len + 1) {
        w->mem = len + 1;
        w->data = realloc(w->data, sizeof(char) * w->mem);
    }
    MPI_Recv(w->data, len + 1, MPI_CHAR, src, TAG_SWORD, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
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
    // Reads some data and converts it to a float array
    // INFO_PRINTF("Starting batcher %d\n", getpid());
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
        cbow_batch(batch, BS, WIN);

        // MPI_Recv(&s, 1, MPI_INT, MPI_ANY_SOURCE, TAG_READY, MPI_COMM_WORLD,
                // MPI_STATUS_IGNORE);
        // MPI_Send(batch, bufsize, MPI_FLOAT, s, TAG_BATCH, MPI_COMM_WORLD);
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
    int me = my_mpi_id();

    PyObject* net = create_network(WIN, EMB);
    create_test_dataset(WIN);
    WeightList wl;
    init_weightlist_like(&wl, net);

    size_t vocab = out_size(net);
    size_t n_words = (BS + WIN + WIN);
    size_t X_numel = BS * (WIN + WIN);
    size_t y_numel = BS * vocab;

    float* X = malloc(X_numel * sizeof(float));
    float* y = malloc(y_numel * sizeof(float));
    float* f_widx = malloc(n_words * sizeof(float));

    for in_range(i, COMM) {
        // MPI_Send(&me, 1, MPI_INT, mpi_id_from_role_id(MASTER, 0),
                // TAG_READY, MPI_COMM_WORLD);
        // recv_weights(&wl, mpi_id_from_role_id(MASTER, 0), TAG_WEIGH);
        // set_net_weights(net, &wl);
        for in_range(k, ITER) {
            MPI_Send(&me, 1, MPI_INT, mpi_id_from_role_id(BATCHER, 0),
                    TAG_READY, MPI_COMM_WORLD);
            MPI_Recv(f_widx, n_words, MPI_FLOAT,
                    mpi_id_from_role_id(BATCHER, 0), TAG_BATCH, MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
            // cbow_batch(X, y, f_widx, BS, WIN);
            step_net(net, X, BS);
#warning "fix this"
            INFO_PRINTLN(".");
        }
        printf("%d net: %f\n", my_mpi_id(), eval_net(net));
        update_weightlist(&wl, net);
        // send_weights(&wl, mpi_id_from_role_id(MASTER, 0), TAG_WEIGH);
    }
    Py_DECREF(net);
    free_weightlist(&wl);
}

void master_node() {
    // 0. Initialize model

    // 1. Send it to some slaves for processing (synchronous)
    // 2. Receive weights back (synchronous)
    // 3. Average the weights


    PyObject* frank = create_network(WIN, EMB);
    WeightList wl;
    init_weightlist_like(&wl, frank);
    update_weightlist(&wl, frank);

    int spr = number_of(SLAVE) * FSPC;  // Slaves per round
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
    // if (role_id_from_mpi_id(my_mpi_id(), MASTER) == 0) {
        // for in_range(r, number_of(BATCHER)) {
            // int stop = -1;
            // MPI_Send(&stop, 1, MPI_INT, reader_id(r), TAG_READY,
                    // MPI_COMM_WORLD);
        // }
    // }
}

int main (int argc, const char **argv) {
    MPI_Init(NULL, NULL);

    // Cython Boilerplate
    PyImport_AppendInittab("bridge", PyInit_bridge);
    Py_Initialize();
    PyRun_SimpleString("import sys\nsys.path.insert(0,'')");
    PyObject* bridge_module = PyImport_ImportModule("bridge");

    // Actual Code
    switch (map_node()) {
        case TOKENIZER:
            tokenizer(argv[1]);
            break;
        case FILTERER:
            filterer();
            break;
        case BATCHER:
            batcher();
            break;
        // case SLAVE:
            // slave_node();
            // break;
        default:
            INFO_PRINTLN("DYING HORRIBLY!");
        // case SLAVE: slave_node(); break;
        // case MASTER: master_node(); break;
    }

    // Finalizing Boilerplate
    Py_DECREF(bridge_module);
    Py_Finalize();
    MPI_Finalize();
}
