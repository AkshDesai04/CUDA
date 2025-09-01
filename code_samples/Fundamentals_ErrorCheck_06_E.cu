```cuda
/* 
 * Aim of the program: Demonstrate its use by checking a kernel that fails due to an out-of-bounds access.
 *
 * My thinking:
 * 1. The goal is to show how to detect and report a runtime error caused by an out-of-bounds memory access
 *    in a CUDA kernel. We intentionally launch a kernel that writes beyond the bounds of an array.
 * 2. We will allocate a small array on the device, launch more threads than the array size,
 *    and let each thread write to array[idx] where idx can exceed the array bounds.
 * 3. After the kernel launch we will check for errors using cudaGetLastError() and cudaDeviceSynchronize().
 *    cudaGetLastError() retrieves errors that are generated during the launch or prior asynchronous operations,
 *    while cudaDeviceSynchronize() forces completion of the kernel and catches errors that happen during execution.
 * 4. We will print the error code and message so that the user can see that the out-of-bounds access was detected.
 * 5. The code is a minimal self-contained .cu file that compiles with nvcc and runs on a CUDA-capable GPU.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Simple kernel that writes to an array without bounds checking
__global__ void outOfBoundsKernel(int *d_array, int N)
{
    // Compute a linear thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Intentionally write to d_array[idx] even if idx >= N
    d_array[idx] = idx;
}

// Helper macro for error checking
#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

int main()
{
    const int N = 10;          // Size of the array
    const int threadsPerBlock = 4;
    const int blocks = (2 * N + threadsPerBlock - 1) / threadsPerBlock; // Intentionally double the number of threads

    // Allocate host memory (not used for data, just for allocation)
    int *h_array = (int *)malloc(N * sizeof(int));

    // Allocate device memory
    int *d_array = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_array, N * sizeof(int)));

    // Launch kernel with more threads than array size to cause out-of-bounds access
    printf("Launching kernel with %d blocks and %d threads per block (total %d threads)\n",
           blocks, threadsPerBlock, blocks * threadsPerBlock);
    outOfBoundsKernel<<<blocks, threadsPerBlock>>>(d_array, N);

    // Check for errors that occur during launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Kernel launch succeeded (no launch error).\n");
    }

    // Synchronize to catch errors that occur during kernel execution
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Kernel execution succeeded (no runtime error).\n");
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_array));
    free(h_array);

    return 0;
}
```