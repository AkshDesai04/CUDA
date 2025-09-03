```cpp
/*
Aim of the program: Is the boundary check computationally expensive? Why or why not?

This program demonstrates a simple CUDA kernel that performs a boundary check
before accessing an element of an array. The boundary check is a very cheap
branch that compares an index with the array size. In the context of CUDA
warps, if all threads in a warp perform the same branch (either all pass
or all fail the boundary test), there is no divergence and the branch
execution costs only the cost of the comparison itself, which is negligible.
When some threads pass and others fail, the warp is split into two subwarps,
but since the branch is simple and the payload is small, the overhead is
minimal compared to the cost of memory accesses. Therefore, the boundary
check is not computationally expensive, especially for full warps with
no divergence.

The code below sets up a simple array, launches a kernel that performs a
boundary check, and prints whether each element was processed or skipped.
The main focus is to illustrate how the boundary check is implemented
in a typical CUDA kernel.

Thought process:
1. Create a small array on host and device.
2. Launch a kernel with one thread per element.
3. In the kernel, compute global thread index.
4. Perform a boundary check: if (idx < N) { ... } else { skip }.
5. Store the result (e.g., set to -1 if out of bounds).
6. Copy back to host and print results.
This will show that the kernel handles out-of-range indices gracefully
without incurring significant cost.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 10

// CUDA kernel that performs a boundary check before writing to an array
__global__ void boundary_check_kernel(int *d_array, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Very cheap boundary check
    if (idx < size) {
        // Perform some simple operation, e.g., square the index
        d_array[idx] = idx * idx;
    } else {
        // Mark out-of-bounds threads with a sentinel value
        d_array[idx] = -1;
    }
}

int main(void)
{
    int h_array[N * 2];          // Host array (double the size to force out-of-bounds)
    int *d_array = nullptr;
    size_t bytes = sizeof(int) * N * 2;

    // Allocate device memory
    cudaMalloc((void**)&d_array, bytes);

    // Initialize host array with zeros
    for (int i = 0; i < N * 2; ++i) {
        h_array[i] = 0;
    }

    // Launch kernel with enough threads to cover the array
    int threadsPerBlock = 256;
    int blocks = (N * 2 + threadsPerBlock - 1) / threadsPerBlock;
    boundary_check_kernel<<<blocks, threadsPerBlock>>>(d_array, N);

    // Copy results back to host
    cudaMemcpy(h_array, d_array, bytes, cudaMemcpyDeviceToHost);

    // Print results
    printf("Results after boundary check kernel execution:\n");
    for (int i = 0; i < N * 2; ++i) {
        printf("Index %2d: %d\n", i, h_array[i]);
    }

    // Clean up
    cudaFree(d_array);
    return 0;
}
```