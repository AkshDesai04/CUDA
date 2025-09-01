```cuda
/*
Aim: Calculate the size in bytes required for the allocation (256 * sizeof(int))
and use that variable in both cudaMalloc and cudaMemcpy.

Thinking:
- Determine the number of elements: 256.
- Compute the byte size: size_t size = N * sizeof(int).
- Allocate host array and fill it with some values.
- Allocate device memory using cudaMalloc with the same size variable.
- Copy data from host to device using cudaMemcpy with the size variable.
- Launch a simple kernel (e.g., double each element) to demonstrate usage.
- Copy the result back to host using cudaMemcpy and verify.
- Use error checking macros for robustness.
- Print the first few results for verification.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 256
#define CHECK(call)                                                         \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Simple kernel that doubles each element
__global__ void doubleElements(int *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] *= 2;
    }
}

int main(void) {
    int h_arr[N];
    int *d_arr = NULL;
    size_t size = N * sizeof(int);  // Size in bytes

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        h_arr[i] = i;
    }

    // Allocate device memory
    CHECK(cudaMalloc((void **)&d_arr, size));

    // Copy from host to device
    CHECK(cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice));

    // Launch kernel: one block of 256 threads
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    doubleElements<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    // Check for kernel launch errors
    CHECK(cudaGetLastError());

    // Copy result back to host
    CHECK(cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost));

    // Print first 10 results to verify
    printf("First 10 results after doubling:\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_arr[%d] = %d\n", i, h_arr[i]);
    }

    // Clean up
    CHECK(cudaFree(d_arr));

    printf("Program completed successfully.\n");
    return 0;
}
```