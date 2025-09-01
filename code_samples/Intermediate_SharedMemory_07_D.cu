```cuda
/*
Aim of the program:
Can you reverse an array using only atomic swaps in global memory? Compare the performance.

Thinking:
The goal is to reverse an array in place on the GPU by using only atomic swap operations (cuda atomic functions) on global memory, and then compare its performance against a conventional non-atomic reverse kernel that simply reads and writes to global memory. 
To achieve an atomic reverse we need to swap pairs of elements (i, N-1-i).  CUDA provides atomicExch which atomically exchanges the value at a memory address with a new value.  We can perform a two-step swap: 
    temp = atomicExch(&a[i], a[j]);   // a[i] becomes a[j], temp holds original a[i]
    atomicExch(&a[j], temp);         // a[j] becomes original a[i]
This guarantees that two threads that might try to swap overlapping pairs do not corrupt data because each atomicExch is a single read-modify-write that is indivisible.  The kernel assigns one thread per pair and only uses atomicExch, ensuring no other global memory accesses that could race.

For comparison we implement a naive reverse kernel that just reads a[i] into a temporary register and writes it to a[N-1-i] without any atomic operations.  This will typically be faster because it uses normal memory traffic but may suffer from memory coalescing issues.  We will time both kernels using CUDA events, run them on a large array (e.g. 2^20 elements), and print the elapsed time for each.  We also validate that both kernels produce the same result.

The program includes:
- Host code to allocate and initialize data, copy to device, launch kernels, time them, copy back, verify, and free resources.
- Two device kernels: one using atomic swaps, one naive.
- Utility error checking.

All code is contained in a single .cu file. The program can be compiled with nvcc and run on any CUDA-capable device.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1 << 20)          // Array size (1,048,576 elements)
#define THREADS_PER_BLOCK 256

// Utility macro for CUDA error checking
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel that reverses array using only atomic swaps in global memory
__global__ void atomicSwapReverse(int *d_arr, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = idx;
    size_t j = n - 1 - idx;

    if (i < j) {
        // Perform atomic swap between d_arr[i] and d_arr[j]
        int temp = atomicExch(&d_arr[i], d_arr[j]);
        atomicExch(&d_arr[j], temp);
    }
}

// Naive kernel that reverses array without atomic operations
__global__ void naiveReverse(int *d_arr, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = idx;
    size_t j = n - 1 - idx;

    if (i < j) {
        int temp = d_arr[i];
        d_arr[i] = d_arr[j];
        d_arr[j] = temp;
    }
}

// Host function to verify that two arrays are identical
int verify(const int *a, const int *b, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            printf("Mismatch at index %zu: %d != %d\n", i, a[i], b[i]);
            return 0;
        }
    }
    return 1;
}

int main()
{
    // Allocate host memory
    int *h_orig = (int *)malloc(N * sizeof(int));
    int *h_result_atomic = (int *)malloc(N * sizeof(int));
    int *h_result_naive = (int *)malloc(N * sizeof(int));
    if (!h_orig || !h_result_atomic || !h_result_naive) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize array with some pattern
    for (size_t i = 0; i < N; ++i) {
        h_orig[i] = (int)i;
    }

    // Allocate device memory
    int *d_arr_atomic = NULL;
    int *d_arr_naive = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_arr_atomic, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_arr_naive, N * sizeof(int)));

    // Copy original data to both device arrays
    CUDA_CHECK(cudaMemcpy(d_arr_atomic, h_orig, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_arr_naive, h_orig, N * sizeof(int), cudaMemcpyHostToDevice));

    // Determine grid size
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    // Events for timing
    cudaEvent_t start_atomic, stop_atomic;
    cudaEvent_t start_naive, stop_naive;
    CUDA_CHECK(cudaEventCreate(&start_atomic));
    CUDA_CHECK(cudaEventCreate(&stop_atomic));
    CUDA_CHECK(cudaEventCreate(&start_naive));
    CUDA_CHECK(cudaEventCreate(&stop_naive));

    // Launch atomic swap kernel and time it
    CUDA_CHECK(cudaEventRecord(start_atomic));
    atomicSwapReverse<<<grid, block>>>(d_arr_atomic, N);
    CUDA_CHECK(cudaEventRecord(stop_atomic));
    CUDA_CHECK(cudaEventSynchronize(stop_atomic));

    // Launch naive kernel and time it
    CUDA_CHECK(cudaEventRecord(start_naive));
    naiveReverse<<<grid, block>>>(d_arr_naive, N);
    CUDA_CHECK(cudaEventRecord(stop_naive));
    CUDA_CHECK(cudaEventSynchronize(stop_naive));

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_result_atomic, d_arr_atomic, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_result_naive, d_arr_naive, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify correctness
    int ok_atomic = verify(h_orig, h_result_atomic, N);
    int ok_naive = verify(h_orig, h_result_naive, N);

    // Compute elapsed times
    float ms_atomic = 0.0f, ms_naive = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_atomic, start_atomic, stop_atomic));
    CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start_naive, stop_naive));

    // Print results
    printf("Atomic swap reverse: %s, time = %.3f ms\n",
           ok_atomic ? "Success" : "Failure", ms_atomic);
    printf("Naive reverse:       %s, time = %.3f ms\n",
           ok_naive ? "Success" : "Failure", ms_naive);

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start_atomic));
    CUDA_CHECK(cudaEventDestroy(stop_atomic));
    CUDA_CHECK(cudaEventDestroy(start_naive));
    CUDA_CHECK(cudaEventDestroy(stop_naive));
    CUDA_CHECK(cudaFree(d_arr_atomic));
    CUDA_CHECK(cudaFree(d_arr_naive));
    free(h_orig);
    free(h_result_atomic);
    free(h_result_naive);

    return 0;
}
```