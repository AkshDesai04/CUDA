```cpp
/*
Aim of the program: reverse an array using only atomic swaps in global memory? Compare the performance.

Thinking:
- We want to implement an array reversal on the GPU where each thread performs a swap of two elements using atomic operations.
- CUDA provides atomicExch which exchanges the value at a memory location with a new value atomically. 
  We can use two atomicExch calls per pair to perform a swap: 
      old = atomicExch(&a, b);
      atomicExch(&b, old);
- We'll create two kernels:
  1. reverse_atomic: each thread swaps arr[i] with arr[n-1-i] using atomicExch.
  2. reverse_non_atomic: each thread swaps arr[i] with arr[n-1-i] using normal memory accesses (no atomics).
- Since each swap pair is distinct (i and n-1-i), there is no race condition in the non-atomic kernel; it should be faster.
- We will time both kernels using cudaEventRecord and compute elapsed time.
- We will also verify correctness by comparing the GPU result with the expected reversed array computed on the host.
- The code will:
  * Allocate host array of size N (e.g., 10 million elements).
  * Initialize it with sequential values.
  * Allocate device memory and copy data.
  * Run reverse_atomic kernel, time it, copy result back, verify correctness.
  * Run reverse_non_atomic kernel, time it, copy result back, verify correctness.
  * Print timing results and a success/failure message.
- Error checking will be performed after CUDA API calls.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

#define N 10000000          // Number of elements in the array
#define BLOCK_SIZE 256      // Threads per block

// Kernel that reverses an array using atomic swaps
__global__ void reverse_atomic(int *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n / 2) {
        int j = n - 1 - idx;
        // First atomic exchange: arr[idx] becomes arr[j], old value of arr[idx] is stored
        int old = atomicExch(&arr[idx], arr[j]);
        // Second atomic exchange: arr[j] becomes old value
        atomicExch(&arr[j], old);
    }
}

// Kernel that reverses an array using normal memory operations
__global__ void reverse_non_atomic(int *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n / 2) {
        int j = n - 1 - idx;
        int temp = arr[idx];
        arr[idx] = arr[j];
        arr[j] = temp;
    }
}

// Helper function for CUDA error checking
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Host function to verify that array is reversed correctly
int verify_reverse(const int *original, const int *reversed, int n) {
    for (int i = 0; i < n; ++i) {
        if (reversed[i] != original[n - 1 - i]) {
            return 0; // Mismatch found
        }
    }
    return 1; // All elements match
}

int main() {
    // Allocate host memory
    int *h_arr = (int *)malloc(N * sizeof(int));
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize array with sequential values
    for (int i = 0; i < N; ++i) {
        h_arr[i] = i;
    }

    // Copy original array for later verification
    int *h_orig = (int *)malloc(N * sizeof(int));
    if (!h_orig) {
        fprintf(stderr, "Failed to allocate host memory for original array.\n");
        free(h_arr);
        return EXIT_FAILURE;
    }
    memcpy(h_orig, h_arr, N * sizeof(int));

    // Allocate device memory
    int *d_arr;
    cudaError_t err = cudaMalloc((void **)&d_arr, N * sizeof(int));
    checkCudaError(err, "cudaMalloc for d_arr");

    // Copy data from host to device
    err = cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);
    checkCudaError(err, "cudaMemcpy H2D");

    // Determine grid dimensions
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ====================== Atomic Kernel ======================
    // Reset device array to original
    err = cudaMemcpy(d_arr, h_orig, N * sizeof(int), cudaMemcpyHostToDevice);
    checkCudaError(err, "cudaMemcpy reset H2D for atomic kernel");

    cudaEventRecord(start, 0);
    reverse_atomic<<<gridSize, BLOCK_SIZE>>>(d_arr, N);
    err = cudaGetLastError();
    checkCudaError(err, "reverse_atomic kernel launch");

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float atomic_time_ms = 0.0f;
    cudaEventElapsedTime(&atomic_time_ms, start, stop);

    // Copy result back to host
    int *h_result_atomic = (int *)malloc(N * sizeof(int));
    if (!h_result_atomic) {
        fprintf(stderr, "Failed to allocate host memory for atomic result.\n");
        cudaFree(d_arr);
        free(h_arr);
        free(h_orig);
        return EXIT_FAILURE;
    }
    err = cudaMemcpy(h_result_atomic, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    checkCudaError(err, "cudaMemcpy D2H for atomic result");

    // Verify correctness
    int atomic_correct = verify_reverse(h_orig, h_result_atomic, N);

    // ====================== Non-Atomic Kernel ======================
    // Reset device array to original
    err = cudaMemcpy(d_arr, h_orig, N * sizeof(int), cudaMemcpyHostToDevice);
    checkCudaError(err, "cudaMemcpy reset H2D for non-atomic kernel");

    cudaEventRecord(start, 0);
    reverse_non_atomic<<<gridSize, BLOCK_SIZE>>>(d_arr, N);
    err = cudaGetLastError();
    checkCudaError(err, "reverse_non_atomic kernel launch");

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float non_atomic_time_ms = 0.0f;
    cudaEventElapsedTime(&non_atomic_time_ms, start, stop);

    // Copy result back to host
    int *h_result_non_atomic = (int *)malloc(N * sizeof(int));
    if (!h_result_non_atomic) {
        fprintf(stderr, "Failed to allocate host memory for non-atomic result.\n");
        cudaFree(d_arr);
        free(h_arr);
        free(h_orig);
        free(h_result_atomic);
        return EXIT_FAILURE;
    }
    err = cudaMemcpy(h_result_non_atomic, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    checkCudaError(err, "cudaMemcpy D2H for non-atomic result");

    // Verify correctness
    int non_atomic_correct = verify_reverse(h_orig, h_result_non_atomic, N);

    // ====================== Results ======================
    printf("Array size: %d elements\n", N);
    printf("Atomic kernel time: %.3f ms\n", atomic_time_ms);
    printf("Non-atomic kernel time: %.3f ms\n", non_atomic_time_ms);
    printf("Atomic kernel correctness: %s\n", atomic_correct ? "PASS" : "FAIL");
    printf("Non-atomic kernel correctness: %s\n", non_atomic_correct ? "PASS" : "FAIL");

    // ====================== Cleanup ======================
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_arr);
    free(h_arr);
    free(h_orig);
    free(h_result_atomic);
    free(h_result_non_atomic);

    return 0;
}
```