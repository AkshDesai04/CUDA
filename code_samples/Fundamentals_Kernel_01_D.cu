/*
Use `dim3` variables to define a grid of (1,1,1) and a block of (64,1,1) and launch the kernel.

Thinking:
1. The user explicitly asks to define a CUDA kernel launch configuration using dim3 for grid and block.
2. The grid must be (1,1,1) and the block must be (64,1,1). This means only 64 threads will run in total.
3. We will write a simple kernel that performs an operation on an array of 64 integers so that the effect of the launch can be observed.
4. Allocate a host array of 64 integers initialized to zero. Allocate a device array of the same size.
5. The kernel will add 1 to each element of the device array based on its thread index.
6. After launching the kernel, copy the result back to the host and print the array to verify that all elements are incremented.
7. Use standard CUDA error checking macros to ensure the program fails gracefully if something goes wrong.
8. The program will be a self‑contained .cu file that can be compiled with nvcc and run on any CUDA‑capable GPU.
9. No external files or libraries are required beyond the CUDA runtime.
10. The code follows the user instruction strictly and includes the required aim statement verbatim in a multiline comment at the top.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple CUDA kernel that adds 1 to each element of the array
__global__ void increment_kernel(int *d_array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_array[idx] += 1;
    }
}

// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

int main(void) {
    const int N = 64;  // Number of elements matches block size
    int h_array[N];

    // Initialize host array to zeros
    for (int i = 0; i < N; ++i) {
        h_array[i] = 0;
    }

    // Allocate device memory
    int *d_array;
    CUDA_CHECK(cudaMalloc((void **)&d_array, N * sizeof(int)));

    // Copy host array to device
    CUDA_CHECK(cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(64, 1, 1);

    // Launch kernel
    increment_kernel<<<gridDim, blockDim>>>(d_array, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_array));

    // Print results
    printf("Resulting array after kernel execution:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    return 0;
}
