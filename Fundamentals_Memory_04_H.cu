/*
Aim of the program: Allocate memory and use cudaMemsetAsync with a non-default stream.

Thinking:
The goal is to demonstrate how to allocate device memory, create a CUDA stream, use cudaMemsetAsync to set memory asynchronously on that stream, copy the data back to the host on the same stream, and finally synchronize and verify the result. We will allocate an array of integers on the device, set all bytes to 0xFF (which for an int becomes 0xFFFFFFFF). We will then copy the data back to the host asynchronously and print the first few elements to confirm the operation. Error checking will be performed after each CUDA API call to catch any issues early. The program is written in standard C with CUDA extensions and can be compiled with nvcc to produce a .cu file. */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

int main(void)
{
    const size_t N = 10;                     // number of integers
    int *d_array = NULL;
    int *h_array = (int *)malloc(N * sizeof(int));
    if (!h_array) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_array, N * sizeof(int)));

    // Create a non-default stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Asynchronously set all bytes of the device array to 0xFF on the created stream
    CUDA_CHECK(cudaMemsetAsync(d_array, 0xFF, N * sizeof(int), stream));

    // Asynchronously copy the data back to the host on the same stream
    CUDA_CHECK(cudaMemcpyAsync(h_array, d_array, N * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));

    // Wait for the stream operations to complete
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Print the result to verify
    printf("First %zu elements of the array after cudaMemsetAsync:\n", N);
    for (size_t i = 0; i < N; ++i) {
        printf("h_array[%zu] = 0x%X\n", i, h_array[i]);
    }

    // Clean up
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_array));
    free(h_array);

    return EXIT_SUCCESS;
}
