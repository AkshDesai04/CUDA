/*
Allocate on host, allocate on device, but forget the `cudaMemcpy` call. What is in the device memory?

My thinking:
We want to demonstrate that device memory is uninitialized if we don't copy data from host.
We'll allocate a host array, initialize it with known values, allocate a device array,
and skip the cudaMemcpy from host to device. Then we copy from device back to a second
host array and print it. We expect the device memory to contain garbage or zeros
depending on the GPU driver implementation. We'll add basic error checking and free
memory. The code will compile with nvcc and run on a CUDA-capable GPU.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

int main(void) {
    const int N = 10;
    size_t size = N * sizeof(int);

    // Allocate host memory and initialize
    int *h_src = (int*)malloc(size);
    int *h_dst = (int*)malloc(size);
    if (!h_src || !h_dst) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < N; ++i) {
        h_src[i] = i + 1;          // Fill with 1,2,3,...
    }

    // Allocate device memory
    int *d_data;
    CHECK_CUDA(cudaMalloc((void**)&d_data, size));

    // Intentionally forget to copy h_src to d_data
    // (no cudaMemcpy(h_src, d_data, size, cudaMemcpyHostToDevice))

    // Copy from device to host (this will copy uninitialized data)
    CHECK_CUDA(cudaMemcpy(h_dst, d_data, size, cudaMemcpyDeviceToHost));

    // Print the host source data (for reference)
    printf("Host source data (initialized):\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_src[i]);
    }
    printf("\n\n");

    // Print the host destination data (copied from uninitialized device memory)
    printf("Host destination data (copied from device memory without prior memcpy):\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_dst[i]);
    }
    printf("\n");

    // Clean up
    free(h_src);
    free(h_dst);
    CHECK_CUDA(cudaFree(d_data));

    return EXIT_SUCCESS;
}
