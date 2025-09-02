/*
Can tools like Valgrind detect CUDA device memory leaks? (Not directly, but `cuda-memcheck --leak-check` can).

Thinking:
- We need to demonstrate that Valgrind cannot detect CUDA device memory leaks.
- CUDA device memory is separate from host memory; Valgrind runs on host and cannot see device allocations.
- We can write a simple CUDA C program that allocates device memory with cudaMalloc, never frees it.
- Launch a kernel that writes to this memory.
- When run under valgrind, it will not report the leak.
- When run under cuda-memcheck with --leak-check, it will detect the leaked allocation.
- The program will allocate an array of ints, set them to zero in a kernel, and exit without freeing.
- The code will compile with nvcc and run normally.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void initArray(int *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] = 0;
    }
}

int main() {
    int n = 10;
    int *d_arr;
    cudaError_t err;

    err = cudaMalloc((void**)&d_arr, n * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    int threadsPerBlock = 32;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    initArray<<<blocks, threadsPerBlock>>>(d_arr, n);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        // Intentionally not freeing to leak memory
        return 1;
    }

    printf("Kernel launched. Intentionally leaking device memory.\n");
    // Note: d_arr is not freed to create a device memory leak.
    return 0;
}
