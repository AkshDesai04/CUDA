/*
Aim: Does the leaked memory get freed when the application exits? Use `nvidia-smi` to confirm.

Thinking:
The goal is to write a simple CUDA program that intentionally leaks GPU memory by allocating memory with cudaMalloc and not freeing it before exiting. By running the program and then observing the GPU memory usage with `nvidia-smi`, we can confirm whether the leaked memory is reclaimed after the process terminates. CUDA's runtime typically releases all device memory on process exit, but this can be verified.

We will:
1. Allocate a moderate amount of GPU memory (e.g., 100 MB).
2. Launch a simple kernel that writes to that memory.
3. Intentionally omit `cudaFree` and `cudaDeviceReset`.
4. Exit the program normally.
5. The user can run `nvidia-smi` before and after running this program to see if memory usage goes back to the baseline.

The program will also print the allocated size and a message indicating that the memory was not freed.

Note: Since we cannot run `nvidia-smi` from within the CUDA code, the user will have to run it externally.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Simple kernel that writes the thread index into the array
__global__ void initArray(int *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] = idx;
    }
}

int main(void) {
    // Allocate 25 million integers (~100 MB)
    const int numInts = 25 * 1000 * 1000; // 25 million
    int *d_ptr = NULL;
    size_t size = numInts * sizeof(int);

    printf("Allocating %zu bytes (~%.2f MB) of GPU memory.\n",
           size, size / (1024.0 * 1024.0));

    CHECK_CUDA(cudaMalloc((void **)&d_ptr, size));

    // Launch kernel to initialize the memory
    int threadsPerBlock = 256;
    int blocksPerGrid = (numInts + threadsPerBlock - 1) / threadsPerBlock;
    initArray<<<blocksPerGrid, threadsPerBlock>>>(d_ptr, numInts);
    CHECK_CUDA(cudaGetLastError());

    // Synchronize to ensure kernel completes
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("Kernel finished. Intentionally not freeing GPU memory.\n");
    printf("Exiting program. Check GPU memory usage with `nvidia-smi` before and after.\n");

    // Note: We do not call cudaFree(d_ptr) or cudaDeviceReset().
    // The CUDA runtime will automatically clean up device memory when the process exits.

    return 0;
}
