```cu
/* 
The exchange involves GPU 0 sending its rightmost element to GPU 1's 'left halo' region, 
and GPU 1 sending its leftmost element to GPU 0's 'right halo' region. 
I reasoned that to demonstrate this, I need a simple CUDA program that:
1. Allocates an array on each GPU with at least one element on each side that can be considered a halo.
2. Initializes the arrays with distinct values so we can easily see the exchange.
3. Enables peer access between the two GPUs if supported.
4. Performs a direct GPU-to-GPU copy using cudaMemcpyPeer to transfer the rightmost element of GPU 0 to the leftmost position of GPU 1, and vice versa for the leftmost element of GPU 1 to the rightmost position of GPU 0.
5. Copies the results back to the host and prints them to verify the exchange.
6. Handles all error checking and cleans up resources.
The program is self‑contained, uses only CUDA APIs, and can be compiled with nvcc to produce an executable that runs on a system with two GPUs. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 5   // number of elements in each array (excluding halo positions)

// Macro for error checking
#define CHECK_CUDA(call)                                                    \
    {                                                                       \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

int main(void)
{
    int device0 = 0, device1 = 1;
    int *dA = NULL, *dB = NULL;
    int hA[N], hB[N];
    size_t bytes = N * sizeof(int);

    // Initialize host arrays with distinct values
    for (int i = 0; i < N; ++i) {
        hA[i] = i;          // GPU 0 array: 0,1,2,3,4
        hB[i] = 10 + i;     // GPU 1 array: 10,11,12,13,14
    }

    // Allocate and copy data on GPU 0
    CHECK_CUDA(cudaSetDevice(device0));
    CHECK_CUDA(cudaMalloc((void**)&dA, bytes));
    CHECK_CUDA(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));

    // Allocate and copy data on GPU 1
    CHECK_CUDA(cudaSetDevice(device1));
    CHECK_CUDA(cudaMalloc((void**)&dB, bytes));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    // Enable peer access if possible
    int canAccessPeer01 = 0, canAccessPeer10 = 0;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccessPeer01, device0, device1));
    CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccessPeer10, device1, device0));

    if (canAccessPeer01) {
        CHECK_CUDA(cudaSetDevice(device0));
        CHECK_CUDA(cudaDeviceEnablePeerAccess(device1, 0));
    }
    if (canAccessPeer10) {
        CHECK_CUDA(cudaSetDevice(device1));
        CHECK_CUDA(cudaDeviceEnablePeerAccess(device0, 0));
    }

    // Perform the halo exchange
    // GPU 0's rightmost element (index N-1) -> GPU 1's left halo (index 0)
    CHECK_CUDA(cudaSetDevice(device0));
    CHECK_CUDA(cudaMemcpyPeer(dB + 0, device1, dA + (N-1), device0, sizeof(int)));

    // GPU 1's leftmost element (index 0) -> GPU 0's right halo (index N-1)
    CHECK_CUDA(cudaSetDevice(device1));
    CHECK_CUDA(cudaMemcpyPeer(dA + (N-1), device0, dB + 0, device1, sizeof(int)));

    // Copy results back to host
    CHECK_CUDA(cudaSetDevice(device0));
    CHECK_CUDA(cudaMemcpy(hA, dA, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaSetDevice(device1));
    CHECK_CUDA(cudaMemcpy(hB, dB, bytes, cudaMemcpyDeviceToHost));

    // Print results
    printf("After halo exchange:\n");
    printf("GPU 0 array: ");
    for (int i = 0; i < N; ++i)
        printf("%d ", hA[i]);
    printf("\n");

    printf("GPU 1 array: ");
    for (int i = 0; i < N; ++i)
        printf("%d ", hB[i]);
    printf("\n");

    // Clean up
    CHECK_CUDA(cudaSetDevice(device0));
    CHECK_CUDA(cudaFree(dA));

    CHECK_CUDA(cudaSetDevice(device1));
    CHECK_CUDA(cudaFree(dB));

    return 0;
}
```