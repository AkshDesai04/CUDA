/*
Write a program that deliberately triggers a `cudaErrorInvalidValue` by passing a `nullptr` as the destination for `cudaMemcpy`. Use your `CHECK()` macro to catch and report the error cleanly.

Thinking:
To provoke a `cudaErrorInvalidValue`, we simply call `cudaMemcpy` with a null destination pointer. The program will allocate a small chunk of device memory as the source, then attempt the copy. The `CHECK()` macro will intercept the returned error code, print a helpful message (file, line, and the CUDA error string), and exit the program. This demonstrates how to handle CUDA errors cleanly and ensures that the error is caught early during development or debugging.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define CHECK(call)                                            \
    do {                                                       \
        cudaError_t err = (call);                              \
        if (err != cudaSuccess) {                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

int main(void) {
    const size_t N = 10;
    const size_t size = N * sizeof(int);

    int *devSrc = NULL;
    CHECK(cudaMalloc(&devSrc, size));

    // Intentionally cause cudaErrorInvalidValue by passing nullptr as destination
    CHECK(cudaMemcpy(nullptr, devSrc, size, cudaMemcpyDeviceToDevice));

    // This point will never be reached because CHECK will exit on error
    CHECK(cudaFree(devSrc));

    return 0;
}
