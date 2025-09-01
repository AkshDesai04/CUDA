```cpp
/* 
Aim of the program:
Read the desired number of threads from the command line and dynamically calculate the grid/block dim3 variables at runtime.

Thinking process:
1. The program takes one command line argument: the total number of threads (N) to launch.
2. For a CUDA kernel launch we need block and grid dimensions. A common strategy is to fix a maximum block size (e.g., 256 threads) and compute the number of blocks as ceil(N / blockSize).
3. To avoid launching blocks that are larger than needed, we clamp blockSize to min(N, 256).
4. Grid dimension is then calculated as (N + blockSize - 1) / blockSize, ensuring enough blocks to cover all threads.
5. The kernel will be simple: each thread writes its global index into a device array if the index is < N.
6. After the kernel execution, we copy the array back to host memory and print a few sample values (first 10 and last 10) to verify correctness.
7. Error checking macros are used to catch CUDA errors.
8. The code is written in pure C++ CUDA (file extension .cu), with all necessary includes and a `__global__` kernel.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error checking macro
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",    \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),         \
                    cudaGetErrorString(err));                              \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Simple kernel that writes its global thread index to an array
__global__ void write_indices(int *d_arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_arr[idx] = idx;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <number_of_threads>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Parse the desired number of threads from command line
    char *endptr = NULL;
    long long int N_long = strtoll(argv[1], &endptr, 10);
    if (endptr == argv[1] || N_long <= 0) {
        fprintf(stderr, "Invalid number of threads: %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    size_t N = (size_t)N_long;

    // Determine block size and grid size
    const size_t MAX_BLOCK_SIZE = 256;
    size_t blockSize = (N < MAX_BLOCK_SIZE) ? N : MAX_BLOCK_SIZE;
    size_t gridSize = (N + blockSize - 1) / blockSize;

    // Allocate device memory
    int *d_arr = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_arr, N * sizeof(int)));

    // Launch kernel
    dim3 blockDim(blockSize);
    dim3 gridDim(gridSize);
    write_indices<<<gridDim, blockDim>>>(d_arr, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    int *h_arr = (int*)malloc(N * sizeof(int));
    if (!h_arr) {
        fprintf(stderr, "Host memory allocation failed\n");
        CHECK_CUDA(cudaFree(d_arr));
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print sample results
    printf("Total threads launched: %zu\n", N);
    printf("Grid size: %zu, Block size: %zu\n", gridSize, blockSize);
    printf("First 10 elements:\n");
    for (size_t i = 0; i < (N < 10 ? N : 10); ++i) {
        printf("h_arr[%zu] = %d\n", i, h_arr[i]);
    }
    if (N > 10) {
        printf("Last 10 elements:\n");
        for (size_t i = N - 10; i < N; ++i) {
            printf("h_arr[%zu] = %d\n", i, h_arr[i]);
        }
    }

    // Clean up
    free(h_arr);
    CHECK_CUDA(cudaFree(d_arr));

    return EXIT_SUCCESS;
}
```