/*
Aim of the program: In the loop, `thread_id` is the global index `blockIdx.x * blockDim.x + threadIdx.x`. `grid_size` is the total number of threads launched `gridDim.x * blockDim.x`. Implement this.

Thinking:
The goal is to show how to compute the global thread index (`thread_id`) and the total number of threads (`grid_size`) inside a CUDA kernel, and use them in a loop that covers an array of size N. Each thread will write its own `thread_id` value into the output array at positions that it owns, stepping by `grid_size` so that the whole array is covered without overlap. The host code sets up the data, launches the kernel with a chosen configuration, copies back the results, and prints a few entries to verify correctness. This example is intentionally simple so it can be compiled with nvcc and run on any CUDA-enabled device.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define N 1024           // Size of the array
#define THREADS_PER_BLOCK 256

// Error checking macro
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, err, cudaGetErrorName(err), cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Kernel that writes its global thread ID into the output array
__global__ void write_thread_id(int *out) {
    // Compute the global thread index and the total number of threads launched
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = gridDim.x * blockDim.x;

    // Each thread processes elements spaced by grid_size
    for (int i = thread_id; i < N; i += grid_size) {
        out[i] = thread_id;
    }
}

int main() {
    int *d_out = nullptr;
    int *h_out = new int[N];

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));

    // Launch kernel
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    write_thread_id<<<blocks, THREADS_PER_BLOCK>>>(d_out);
    CUDA_CHECK(cudaGetLastError());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print first 16 results
    printf("First 16 output values (index : thread_id):\n");
    for (int i = 0; i < 16 && i < N; ++i) {
        printf("%d : %d\n", i, h_out[i]);
    }

    // Clean up
    delete[] h_out;
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
