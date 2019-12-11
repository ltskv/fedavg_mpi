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
#define TAG_TERMT 10

#define COMM 1
#define ITER 690
#define BS 32
#define EMB 32
#define WIN 2
#define FLPC 1

#define in_range(i, x) (size_t i = 0; i < (x); i++)
// I am honestly VERY sorry for this but power corrupts even the best of us

#define INFO_PRINTF(fmt, ...) \
    do { fprintf(stderr, fmt, __VA_ARGS__); } while(0)
#define INFO_PRINTLN(what) \
    do { fprintf(stderr, "%s\n", what); } while(0)
#define INFO_PRINT(what) \
    do { fprintf(stderr, "%s", what); } while(0)

int g_argc;  // sorry!

typedef enum{
    TOKENIZER,
    FILTERER,
    BATCHER,
    LEARNER,
    VISUALIZER,
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
            return g_argc - 1;
        case FILTERER:
            return number_of(TOKENIZER);
        case BATCHER:
            return number_of(TOKENIZER);
        case LEARNER:
            return world_size()
                - number_of(TOKENIZER)
                - number_of(FILTERER)
                - number_of(BATCHER)
                - number_of(DISPATCHER)
                - number_of(VISUALIZER);
        case VISUALIZER:
            return 0;
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
        MPI_Abort(MPI_COMM_WORLD, 1);
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
    MPI_Abort(MPI_COMM_WORLD, 1);  // this is bad
    return -1;  // Not going to happen anyway (i hope)
}

void announce_ready(int dest) {
    int me = my_mpi_id();
    MPI_Send(&me, 1, MPI_INT, dest, TAG_READY, MPI_COMM_WORLD);
}

int wait_for_ready() {
    int ready;
    MPI_Recv(&ready, 1, MPI_INT, MPI_ANY_SOURCE, TAG_READY, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
    return ready;
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

void ssend_word(Word* w, int dest) {
    long len = strlen(w->data);
    MPI_Ssend(&len, 1, MPI_LONG, dest, TAG_STLEN, MPI_COMM_WORLD);
    MPI_Ssend(w->data, len + 1, MPI_CHAR, dest, TAG_SWORD, MPI_COMM_WORLD);
}

int recv_word(Word* w, int src) {
    // WAT is going on here I have no idea
    long len;
    MPI_Status stat;
    MPI_Recv(&len, 1, MPI_LONG, src, TAG_STLEN, MPI_COMM_WORLD,
            &stat);
    int the_src = stat.MPI_SOURCE;
    if (w->mem < len + 1) {
        w->mem = len + 1;
        w->data = realloc(w->data, sizeof(char) * w->mem);
    }
    MPI_Recv(w->data, len + 1, MPI_CHAR, the_src, TAG_SWORD, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
    return the_src;
}

void tokenizer(const char* source) {
    INFO_PRINTF("Starting tokenizer %d\n", getpid());
    int rid = role_id_from_mpi_id(TOKENIZER, my_mpi_id());

    WordList wl = {0, 0, NULL};
    size_t sync_ctr = 0;

    Word terminator = {1, ""};
    MPI_Request stop_req;
    int stop;
    MPI_Irecv(&stop, 1, MPI_INT, MPI_ANY_SOURCE, TAG_TERMT, MPI_COMM_WORLD,
            &stop_req);
    MPI_Test(&stop_req, &stop, MPI_STATUS_IGNORE);

    while (!stop && get_tokens(&wl, source)) {
        for in_range(i, wl.n_words) {
            if (sync_ctr == 10000) {
                ssend_word(&wl.words[i], mpi_id_from_role_id(FILTERER, rid));
                sync_ctr = 0;
            } else {
                send_word(&wl.words[i], mpi_id_from_role_id(FILTERER, rid));
            }
            sync_ctr++;
        }
        MPI_Test(&stop_req, &stop, MPI_STATUS_IGNORE);
    }
    free_wordlist(&wl);
    send_word(&terminator, mpi_id_from_role_id(FILTERER, rid));
    INFO_PRINTF("Finishing tokenizer %d\n", getpid());
}

void filterer() {
    INFO_PRINTF("Starting filterer %d\n", getpid());
    int rid = role_id_from_mpi_id(FILTERER, my_mpi_id());

    Word w = {0, NULL};
    const size_t entry_size = 2 * WIN + 1;
    const size_t bufsize = entry_size;
    long* buffer = malloc(bufsize * sizeof(long));
    size_t have = 0;

    while (1) {
        while (have != entry_size) {  // TODO FLATTEN PIPELINE
            recv_word(&w, mpi_id_from_role_id(TOKENIZER, rid));

            if (!strlen(w.data)) break;

            buffer[have] = vocab_idx_of(&w);
            if (buffer[have] != -1) have++;
        }

        if (!strlen(w.data)) break;

        have = 0;
        MPI_Send(buffer, entry_size, MPI_LONG,
                mpi_id_from_role_id(BATCHER, rid),
                TAG_IWORD, MPI_COMM_WORLD);
    }
    free_word(&w);
    free(buffer);
    INFO_PRINTF("Finishing filterer %d\n", getpid());
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
            for in_range(c, entry_size) {
                batch[r*entry_size + c] = (float)l_wid[c];
            }
        }
        MPI_Recv(&s, 1, MPI_INT, MPI_ANY_SOURCE, TAG_READY, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
        MPI_Send(batch, bufsize, MPI_FLOAT, s, TAG_BATCH, MPI_COMM_WORLD);
        printf("!\n");
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
    INFO_PRINTF("Starting learner %d\n", getpid());
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
        update_weightlist(&wl, net);
        send_weights(&wl, mpi_id_from_role_id(DISPATCHER, 0));
    }
    Py_DECREF(net);
    free_weightlist(&wl);
    free(batch);
}

void dispatcher() {
    INFO_PRINTF("Starting dispatcher %d\n", getpid());
    PyObject* frank = create_network(WIN, EMB);
    create_test_dataset(WIN);
    WeightList wl;
    init_weightlist_like(&wl, frank);
    update_weightlist(&wl, frank);

    int lpr = number_of(LEARNER) * FLPC;  // Learners per round
    WeightList *wls = malloc(sizeof(WeightList) * lpr);
    for in_range(i, lpr) {
        init_weightlist_like(wls + i, frank);
    }
    int *round = malloc(sizeof(int) * lpr);

    float first_loss = eval_net(frank);
    float crt_loss = first_loss;
    float min_loss = crt_loss;
    time_t start = time(NULL);
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
        crt_loss = eval_net(frank);
        min_loss = crt_loss < min_loss ? crt_loss : min_loss;
        INFO_PRINTF("Round %ld, validation loss %f\n", i, crt_loss);
    }

    int stop = 1;
    MPI_Send(&stop, 1, MPI_INT, mpi_id_from_role_id(TOKENIZER, 0), TAG_TERMT,
            MPI_COMM_WORLD);

    time_t finish = time(NULL);
    float delta_t = finish - start;
    float delta_l = first_loss - eval_net(frank);
    INFO_PRINTF(
            "Laptop MPI sgd consecutive_batch W%d E%d BS%d R%d bpe%d LPR%d,"
            "%f,%f,%f\n", WIN, EMB, BS, COMM, ITER, lpr,
            delta_l / COMM, delta_l / delta_t, min_loss);
    Py_DECREF(frank);
    free_weightlist(&wl);
    for in_range(i, lpr) free_weightlist(wls + i);
    free(wls);
    free(round);
}

void visualizer() {
    INFO_PRINTF("Starting visualizer %d\n", getpid());
    serve();
}

int main (int argc, const char **argv) {
    MPI_Init(NULL, NULL);

    if (my_mpi_id() == 0) {
        if (argc < 2) {
            INFO_PRINTLN("NOT ENOUGH INPUTS!");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    g_argc = argc;

    // Cython Boilerplate
    PyImport_AppendInittab("bridge", PyInit_bridge);
    Py_Initialize();
    PyRun_SimpleString("import sys\nsys.path.insert(0,'')");
    PyObject* bridge_module = PyImport_ImportModule("bridge");

    // Actual Code
    int role_id;
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
        case VISUALIZER:
            visualizer();
            break;
        default:
            INFO_PRINTLN("DYING HORRIBLY!");
    }

    // Finalizing Boilerplate
    Py_DECREF(bridge_module);
    Py_Finalize();
    MPI_Finalize();
}
