/*
Aim: Write a kernel that uses `__syncthreads()` incorrectly, for example, inside a conditional branch (`if (threadIdx.x < 16) __syncthreads();`). Compile and run this. Explain why this leads to a deadlock and will cause the kernel to hang or error out.

Thinking:
- `__syncthreads()` is a barrier that requires **all** threads in a block to execute it before any can proceed beyond the barrier. 
- If a conditional branch allows only a subset of threads to hit the barrier, the remaining threads will never execute it. Those threads that did execute the barrier will block forever waiting for the others that never reach it, causing a deadlock.
- In the example, we create a block with 32 threads. Threads with `threadIdx.x < 16` (i.e., 0–15) execute `__syncthreads()`, while threads 16–31 skip it. Thus the first 16 threads will wait at the barrier for the last 16 threads that never call it, leading to a hang.
- The kernel will not return, so `cudaDeviceSynchronize()` will block indefinitely. If the GPU driver detects a deadlock it may eventually report an error, but typically the program just hangs.
- The code below demonstrates this behaviour. Running it will show that the program prints "Kernel launched" but never prints "Kernel finished" or the contents of the array, indicating the deadlock.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Kernel that misuses __syncthreads()
__global__ void badSyncKernel(int *data) {
    // Only the first 16 threads will hit the barrier
    if (threadIdx.x < 16) {
        __syncthreads();   // <-- Incorrect usage
    }
    // All threads continue (but those that missed the barrier are not synchronized)
    // Write the thread index into the array
    data[threadIdx.x] = threadIdx.x;
}

int main(void) {
    const int threadsPerBlock = 32;
    const int blocks = 1;
    const int arraySize = threadsPerBlock * blocks;

    int *d_array = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&d_array, arraySize * sizeof(int)));

    printf("Kernel launched\n");
    badSyncKernel<<<blocks, threadsPerBlock>>>(d_array);

    // Synchronize to detect errors
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        printf("cudaDeviceSynchronize returned error: %s\n",
               cudaGetErrorString(syncErr));
    } else {
        printf("Kernel finished\n");
    }

    // Attempt to copy back the data (will not be reached if deadlocked)
    int *h_array = (int *)malloc(arraySize * sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_array, d_array, arraySize * sizeof(int),
                          cudaMemcpyDeviceToHost));

    // Print the results
    printf("Array contents:\n");
    for (int i = 0; i < arraySize; ++i) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    // Clean up
    free(h_array);
    cudaFree(d_array);
    return 0;
}
