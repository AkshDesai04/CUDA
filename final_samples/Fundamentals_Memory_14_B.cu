/*
Aim: In a kernel launched with N threads for an array of size N, write to d_ptr[threadIdx.x + 1]. Thread N-1 will go out of bounds. Run with cuda-memcheck.

Thinking:
1. We need a simple CUDA program that allocates an array of size N on the device.
2. Launch a kernel with N threads (one block for simplicity). Each thread writes to d_ptr[threadIdx.x + 1].
3. For threadIdx.x == N-1, this index will be N, which is out of bounds because valid indices are 0..N-1.
4. The program should compile and run, and when executed under cuda-memcheck it should report an out-of-bounds write.
5. Include basic error checking for CUDA API calls to ensure we can see errors from the API side as well.
6. Keep the code minimal but complete: include headers, error checking macro, main function, kernel, memory allocation, launch, copy back, free.
7. The kernel should write a simple value (e.g., the thread index) to demonstrate the write operation.
8. After copying back, we can print a few elements to confirm that the array was written to.
9. The code is self-contained and can be compiled with nvcc and run with cuda-memcheck.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 256  // Size of the array and number of threads

// Macro for checking CUDA errors
#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",         \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Kernel that writes out of bounds on the last thread
__global__ void write_out_of_bounds(int *d_ptr, int size) {
    int idx = threadIdx.x;
    if (idx < size) {
        // Intentionally write to d_ptr[idx + 1]
        // For idx == size - 1, idx + 1 == size which is out of bounds
        d_ptr[idx + 1] = idx;
    }
}

int main(void) {
    int *d_ptr = NULL;
    size_t bytes = N * sizeof(int);

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void **)&d_ptr, bytes));

    // Initialize device memory to zero
    CHECK_CUDA(cudaMemset(d_ptr, 0, bytes));

    // Launch kernel with N threads in a single block
    write_out_of_bounds<<<1, N>>>(d_ptr, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy data back to host for inspection
    int h_ptr[N];
    CHECK_CUDA(cudaMemcpy(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost));

    // Print first few and last few elements to see the writes
    printf("First 5 elements:\n");
    for (int i = 0; i < 5; ++i) {
        printf("h_ptr[%d] = %d\n", i, h_ptr[i]);
    }

    printf("\nLast 5 elements:\n");
    for (int i = N - 5; i < N; ++i) {
        printf("h_ptr[%d] = %d\n", i, h_ptr[i]);
    }

    // Free device memory
    CHECK_CUDA(cudaFree(d_ptr));

    return 0;
}
