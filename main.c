#include <Python.h>
#include <stdio.h>
#include <mpi.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "library.h"

#define P_READER 0
#define P_SLAVE 1
#define P_MASTER 2

#define COMM 100
#define ITER 20
#define BS 50

// Reads some data and converts it to 2D float array
void data_reader() {
    while (1) {
        PyArrayObject* batch = mnist_batch(10);

        long* shape = PyArray_SHAPE(batch);
        MPI_Send(shape, 2, MPI_LONG, P_SLAVE, 0, MPI_COMM_WORLD);
        MPI_Send(PyArray_DATA(batch), PyArray_SIZE(batch), MPI_FLOAT,
                P_SLAVE, 0, MPI_COMM_WORLD);
        Py_DECREF(batch);
    }
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
            float* data = malloc(shape[0] * shape[1] * sizeof(float));
            MPI_Recv(data, size, MPI_FLOAT, P_READER, MPI_ANY_TAG,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            PyArrayObject* batch = PyArray_SimpleNewFromData(
                    2, shape, NPY_FLOAT32, data);
            step_net(net, batch);
        }
        printf("%f\n", eval_net(net));
    }
}

// Stores most up-to-date model, sends it to slaves for training
void master_node() {
    for (int i = 0; i < COMM; i++) {
        char go;
        MPI_Send(&go, 1, MPI_CHAR, P_SLAVE, 0, MPI_COMM_WORLD);
    }
}

int main (int argc, const char **argv) {
    int node;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &node);

    // Cython Boilerplate
    PyImport_AppendInittab("library", PyInit_library);
    Py_Initialize();
    import_array();
    PyRun_SimpleString("import sys\nsys.path.insert(0,'')");
    PyObject* library_module = PyImport_ImportModule("library");

    // Actual Code
    if (node == 0) {
        data_reader();
    }
    else if (node == 1) {
        slave_node();
    }
    else if (node == 2) {
        master_node();
    }

    // Cython Finalizing Boilerplate
    Py_DECREF(library_module);
    Py_Finalize();
    MPI_Finalize();
}
