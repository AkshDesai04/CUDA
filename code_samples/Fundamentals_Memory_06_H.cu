/*
In a loop, allocate memory and overwrite the pointer variable in the next iteration. Explain why this is a leak.

Thinking: In CUDA C, allocating device memory with cudaMalloc returns a pointer to the allocated space on the GPU. If we keep a pointer variable (e.g., d_ptr) that is reused in a loop to hold the address of newly allocated memory, each new cudaMalloc overwrites d_ptr with a new address. The previously allocated block becomes unreachable: we no longer have a pointer to it, so we cannot call cudaFree on it. The GPU driver will hold the allocation until the context ends, resulting in a memory leak on the device. To avoid the leak, we should free the previous allocation before reusing the pointer, or keep a collection of pointers and free them all at the end. This program demonstrates the leak by allocating memory in a loop without freeing earlier allocations and then freeing only the last allocation.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that does nothing
__global__ void dummyKernel(int *d_array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_array[idx] = idx;
    }
}

int main() {
    const int N_ITER = 10;          // Number of allocations in the loop
    const int ELEMS_PER_ALLOC = 100; // Number of ints per allocation
    const size_t SIZE = ELEMS_PER_ALLOC * sizeof(int);

    int *d_ptr = NULL;

    printf("Starting allocation loop (with intentional leak)...\n");

    for (int i = 0; i < N_ITER; ++i) {
        // Allocate device memory
        cudaError_t err = cudaMalloc((void**)&d_ptr, SIZE);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed at iteration %d: %s\n", i, cudaGetErrorString(err));
            return EXIT_FAILURE;
        }

        // Optional: Initialize memory
        err = cudaMemset(d_ptr, 0, SIZE);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMemset failed at iteration %d: %s\n", i, cudaGetErrorString(err));
            return EXIT_FAILURE;
        }

        // Launch a dummy kernel to use the memory
        int threadsPerBlock = 256;
        int blocksPerGrid = (ELEMS_PER_ALLOC + threadsPerBlock - 1) / threadsPerBlock;
        dummyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_ptr, ELEMS_PER_ALLOC);

        // Wait for kernel to finish
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed at iteration %d: %s\n", i, cudaGetErrorString(err));
            return EXIT_FAILURE;
        }

        // IMPORTANT: We intentionally *do not* free d_ptr here.
        // The pointer variable will be overwritten on the next loop iteration,
        // leaving the previous allocation unreachable (a leak).
    }

    // Only the last allocation is freed; earlier allocations are leaked.
    cudaError_t err = cudaFree(d_ptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Finished. Last allocation freed. Earlier allocations leaked.\n");

    return EXIT_SUCCESS;
}
