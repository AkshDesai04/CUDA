/*
Remove the `if (index < N)` check and run the program for a non-multiple size with `cuda-memcheck`. Observe the out-of-bounds access errors.

Thinking process:
1. The goal is to demonstrate that removing the boundary guard in a CUDA kernel causes out-of-bounds memory writes.
2. To trigger such a scenario, we pick a kernel that writes to an array using the global thread index.
3. We purposely skip the usual check `if (index < N)` which prevents threads from accessing past the allocated memory.
4. We choose a problem size `N` that is not a multiple of the block size. For example, `N = 500` and `blockSize = 256`. This means the grid will launch `ceil(500/256) = 2` blocks, giving 512 threads total, but only 500 elements are allocated.
5. The last 12 threads (indices 500–511) will write beyond the allocated array bounds, leading to out-of-bounds errors.
6. The host code allocates device memory for `N` integers, launches the kernel, synchronizes, and copies back the data. We include minimal error checking.
7. When run under `cuda-memcheck`, the tool will report "Invalid write of size 4" or similar for those threads.
8. The code below is a complete `.cu` file that can be compiled with `nvcc` and executed under `cuda-memcheck`.

Note: This program intentionally triggers an out-of-bounds write; it should not be used in production code.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 500
#define BLOCK_SIZE 256

// Error checking macro
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

// Kernel without boundary check
__global__ void writeKernel(int *d_arr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Intentionally omitted: if (idx < N)
    d_arr[idx] = idx; // Out-of-bounds write for idx >= N
}

int main() {
    int *d_arr;
    size_t size = N * sizeof(int);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_arr, size));

    // Determine grid dimensions
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch kernel
    writeKernel<<<gridSize, BLOCK_SIZE>>>(d_arr);

    // Synchronize and check for launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Allocate host memory to copy results
    int h_arr[N];
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost));

    // Print first 10 values
    printf("First 10 values:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Free device memory
    CUDA_CHECK(cudaFree(d_arr));

    return 0;
}
