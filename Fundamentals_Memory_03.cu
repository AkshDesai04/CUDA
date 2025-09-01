/*
Write a program that allocates two integer arrays, d_a and d_b, on the device. Initialize d_a by copying data from the host. Then, use cudaMemcpy with the cudaMemcpyDeviceToDevice kind to copy the contents of d_a directly to d_b without involving the host.

Thinking process:
- The program needs to demonstrate device-to-device memory copy on CUDA.
- We will create a small array on the host, copy it to device array d_a, then perform a cudaMemcpy with cudaMemcpyDeviceToDevice to copy d_a to d_b.
- Finally, copy d_b back to the host to verify that the data was correctly transferred.
- We'll use standard CUDA API calls: cudaMalloc, cudaMemcpy, cudaFree.
- Include error checking via a simple macro to simplify debugging.
- Output the arrays before and after the copy to confirm correctness.
- The code is self-contained in a single .cu file and can be compiled with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 10

// Helper macro for error checking
#define CHECK(call)                                                   \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",         \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

int main(void) {
    // Host array initialization
    int h_a[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = i * 10; // arbitrary values
    }

    // Device pointers
    int *d_a = NULL;
    int *d_b = NULL;

    // Allocate device memory for d_a and d_b
    CHECK(cudaMalloc((void**)&d_a, N * sizeof(int)));
    CHECK(cudaMalloc((void**)&d_b, N * sizeof(int)));

    // Copy host array to device array d_a
    CHECK(cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice));

    // Perform device-to-device copy from d_a to d_b
    CHECK(cudaMemcpy(d_b, d_a, N * sizeof(int), cudaMemcpyDeviceToDevice));

    // Allocate host array to retrieve d_b
    int h_b[N];

    // Copy device array d_b back to host
    CHECK(cudaMemcpy(h_b, d_b, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    printf("Contents of host array h_a:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_a[i]);
    }
    printf("\n");

    printf("Contents of host array h_b after device-to-device copy:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_b[i]);
    }
    printf("\n");

    // Clean up
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));

    return 0;
}
