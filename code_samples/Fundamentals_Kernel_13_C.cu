```cpp
/*
Aim of the program: Write a kernel that performs the operation in-place. It takes a single pointer `a` and computes `a[i] = a[i] + 10;`.

Thinking:
1. The kernel should accept a pointer to device memory `int *a` and operate on each element independently.
2. For in-place addition, each thread will read the current value at its index, add 10, and write back.
3. We need to determine the total number of threads launched. We'll calculate the number of blocks from the total number of elements `N` and a chosen block size `BLOCK_SIZE`.
4. Host code will allocate an array, initialize it, copy it to the device, launch the kernel, copy back, and optionally print the results for verification.
5. Error checking macros will help catch CUDA API errors.
6. The program will be self‑contained and compile with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Kernel that adds 10 to each element in-place
__global__ void addTen(int *a, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        a[idx] += 10;
    }
}

int main(void)
{
    const int N = 10;                    // Number of elements
    const int BLOCK_SIZE = 256;          // Threads per block

    int h_a[N];                          // Host array
    for (int i = 0; i < N; ++i)
        h_a[i] = i;                      // Initialize with 0..N-1

    int *d_a = NULL;
    size_t size = N * sizeof(int);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_a, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    addTen<<<gridSize, BLOCK_SIZE>>>(d_a, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost));

    // Print results
    printf("Result after adding 10 to each element:\n");
    for (int i = 0; i < N; ++i)
        printf("%d ", h_a[i]);
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_a));

    return 0;
}
```