```cuda
/* 
Aim: 
Run a program that has an out-of-bounds memory access inside the kernel (an asynchronous error) 
with and without proper post-launch error checking (cudaDeviceSynchronize). 
Observe that without it, the program may complete without any visible error, 
but potentially with incorrect results. 

Thinking:
- The kernel will intentionally write beyond the bounds of a device array to trigger an illegal memory access. 
- CUDA launches kernels asynchronously, so errors in the kernel do not surface immediately after the kernel launch. 
- The host code must explicitly synchronize (cudaDeviceSynchronize) to flush any asynchronous errors to the host. 
- In the first test we will launch the kernel and *not* synchronize before copying results. The error will not be reported until a later API call, and the data may be corrupted or partially written. 
- In the second test we will launch the same kernel but immediately call cudaDeviceSynchronize. This forces the host to wait for kernel completion and will surface the illegal memory access as a CUDA error. 
- We will use a helper macro CUDA_CHECK to catch synchronous errors, and we will also manually check cudaGetLastError after kernel launch to demonstrate that without synchronization the error is still pending. 
- The host array will be initialized to zeros; after a faulty kernel we expect some non-zero values. After synchronization, we expect the program to terminate with an error and the array may remain unchanged. 
- This demonstrates the importance of proper post-launch error checking in CUDA programming. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for checking CUDA API calls
#define CUDA_CHECK(call)                                               \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Kernel that writes to an array, but intentionally goes out of bounds
__global__ void outOfBoundsKernel(int *d_arr, int N) {
    int idx = threadIdx.x;
    // Intentionally access out of bounds by adding an offset
    int outIdx = idx + N;  // This will be >= N for all threads
    d_arr[outIdx] = idx;   // Illegal write
}

// Simple kernel that writes valid indices (for comparison)
__global__ void correctKernel(int *d_arr, int N) {
    int idx = threadIdx.x;
    if (idx < N) {
        d_arr[idx] = idx;
    }
}

int main(void) {
    const int N = 5;
    const int SIZE = N * sizeof(int);
    int *h_arr = (int *)malloc(SIZE);
    int *d_arr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_arr, SIZE));

    // ---------- Test 1: Launch outOfBoundsKernel without synchronization ----------
    printf("=== Test 1: Launch kernel without cudaDeviceSynchronize ===\n");
    // Initialize host array to zero
    for (int i = 0; i < N; ++i) h_arr[i] = 0;
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, SIZE, cudaMemcpyHostToDevice));

    // Launch kernel with N threads (will write out of bounds)
    outOfBoundsKernel<<<1, N>>>(d_arr, N);

    // Immediately check for errors after launch (will not catch the async error yet)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error after kernel launch (should be success here): %s\n", cudaGetErrorString(err));
    } else {
        printf("No error reported immediately after kernel launch.\n");
    }

    // Copy results back to host without synchronization
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, SIZE, cudaMemcpyDeviceToHost));

    // Print host array
    printf("Host array after kernel (without sync): ");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Check for any error that might have been detected during memcpy
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error after memcpy (Test 1): %s\n", cudaGetErrorString(err));
    } else {
        printf("No error reported after memcpy (Test 1).\n");
    }

    // ---------- Test 2: Launch outOfBoundsKernel with synchronization ----------
    printf("\n=== Test 2: Launch kernel with cudaDeviceSynchronize ===\n");
    // Reinitialize host array to zero
    for (int i = 0; i < N; ++i) h_arr[i] = 0;
    // Copy to device again
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, SIZE, cudaMemcpyHostToDevice));

    // Launch kernel
    outOfBoundsKernel<<<1, N>>>(d_arr, N);

    // Immediately check for errors after launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error after kernel launch: %s\n", cudaGetErrorString(err));
    } else {
        printf("No error reported immediately after kernel launch.\n");
    }

    // Synchronize to force error detection
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize detected error: %s\n", cudaGetErrorString(err));
    } else {
        printf("cudaDeviceSynchronize completed without errors.\n");
    }

    // Attempt to copy results back (will fail if previous error was fatal)
    err = cudaMemcpy(h_arr, d_arr, SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error during cudaMemcpy after sync (Test 2): %s\n", cudaGetErrorString(err));
    } else {
        // Print host array
        printf("Host array after kernel (with sync): ");
        for (int i = 0; i < N; ++i) {
            printf("%d ", h_arr[i]);
        }
        printf("\n");
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_arr));
    free(h_arr);

    return 0;
}
```