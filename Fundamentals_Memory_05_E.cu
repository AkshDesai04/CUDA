/*
Use `nvidia-smi` to watch the GPU memory usage while your program runs. Step through with a debugger to see the memory increase after `cudaMalloc` and decrease after `cudaFree`.

My thinking:
- The goal is to create a minimal CUDA program that clearly demonstrates GPU memory allocation and deallocation.
- I will allocate a reasonably sized array on the device (e.g., 100 MB) using `cudaMalloc`.
- To force the GPU to use the memory, I'll launch a simple kernel that writes zeros into the array.
- After the kernel execution, I will free the memory with `cudaFree`.
- I will add console prints and optional `cudaMemGetInfo` calls to show memory state before allocation, after allocation, after kernel, and after free.
- The program will include basic error checking for each CUDA API call.
- The entire program will be self‑contained in a single `.cu` file, ready to compile with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel to initialize the array to zero
__global__ void initArray(float *arr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        arr[idx] = 0.0f;
}

// Helper macro for error checking
#define CHECK_CUDA(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

int main()
{
    size_t numElements = 25 * 1024 * 1024; // 25 million elements (~100 MB for float)
    size_t bytes = numElements * sizeof(float);
    float *d_arr = NULL;

    // Get initial memory info
    size_t freeMem, totalMem;
    CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));
    printf("Before allocation: Free memory = %zu MB, Total memory = %zu MB\n",
           freeMem / (1024 * 1024), totalMem / (1024 * 1024));

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void **)&d_arr, bytes));
    printf("After cudaMalloc: allocated %zu MB\n", bytes / (1024 * 1024));

    // Launch kernel to initialize array
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    initArray<<<blocksPerGrid, threadsPerBlock>>>(d_arr, numElements);
    CHECK_CUDA(cudaGetLastError()); // Check for launch errors
    CHECK_CUDA(cudaDeviceSynchronize()); // Ensure kernel completion

    // Get memory info after kernel
    CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));
    printf("After kernel execution: Free memory = %zu MB\n",
           freeMem / (1024 * 1024));

    // Free device memory
    CHECK_CUDA(cudaFree(d_arr));
    printf("After cudaFree: memory should be released\n");

    // Final memory info
    CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));
    printf("Final state: Free memory = %zu MB\n", freeMem / (1024 * 1024));

    return 0;
}
