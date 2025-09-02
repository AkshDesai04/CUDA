/*
Demonstrate its use by checking a kernel that fails due to an out-of-bounds access.

Thought process:
1. The goal is to write a small CUDA program that intentionally triggers an out-of-bounds memory access inside a kernel.
2. We allocate a small device array, launch a kernel with a thread index that writes past the end of the array.
3. After launching, we use cudaGetLastError() and cudaDeviceSynchronize() to detect the fault.
4. We print out the error string so the user sees that the kernel failed because of out-of-bounds access.
5. The program includes minimal error-checking helpers and cleans up resources.
6. The code is written in plain C with CUDA extensions, compiled as a .cu file.

This program should compile with `nvcc` and, when run, print an error message indicating the out-of-bounds access.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Helper macro for error checking */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",     \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

/* Kernel that intentionally writes out-of-bounds */
__global__ void out_of_bounds_kernel(int *d_arr, int N)
{
    // Compute global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Intentionally access idx+1, causing last thread to go out-of-bounds
    if (idx < N) {
        d_arr[idx + 1] = idx;  // idx+1 can be N when idx == N-1
    }
}

int main(void)
{
    const int N = 10;                    // Size of the array
    const size_t bytes = N * sizeof(int);

    int *d_arr = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_arr, bytes));

    // Initialize device array to zeros
    CUDA_CHECK(cudaMemset(d_arr, 0, bytes));

    // Launch kernel with N threads (one per element)
    int threadsPerBlock = 32;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    out_of_bounds_kernel<<<blocks, threadsPerBlock>>>(d_arr, N);

    // Capture any launch errors
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(launchErr));
    }

    // Synchronize to propagate errors from kernel execution
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(syncErr));
    } else {
        printf("Kernel executed successfully (unexpected).\n");
    }

    // Attempt to copy results back to host (optional)
    int h_arr[N];
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, bytes, cudaMemcpyDeviceToHost));

    // Print host array contents (may contain garbage for the out-of-bounds element)
    printf("Host array after kernel:\n");
    for (int i = 0; i < N; ++i) {
        printf("h_arr[%d] = %d\n", i, h_arr[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_arr));

    return 0;
}
