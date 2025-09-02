/*
Aim: Fix the race condition from the previous problem by using an atomic operation: atomicAdd(&output[0], 1). Verify that the result is now correct.

Thought process:
- The earlier version of the program incremented a single counter in parallel threads without synchronization, causing a race condition and an incorrect final value.
- To eliminate the race condition, we will use the CUDA built‑in atomicAdd function. atomicAdd(&output[0], 1) guarantees that each increment is performed atomically, so all increments will be accounted for.
- The kernel will simply call atomicAdd once per thread. The number of launched threads determines the expected final value.
- We will allocate a single integer on the device, initialize it to 0, launch a large grid of threads, copy the result back to the host, and print it.
- The expected result should equal the total number of threads launched. This will confirm that the race condition has been fixed.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                    \
    do {                                                    \
        cudaError_t err = call;                             \
        if (err != cudaSuccess) {                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)

__global__ void incrementKernel(int *output)
{
    // Each thread performs an atomic add on the first element
    atomicAdd(output, 1);
}

int main(void)
{
    const int threadsPerBlock = 256;
    const int numBlocks = 256; // Total threads = 65536
    const int totalThreads = threadsPerBlock * numBlocks;

    int *d_output = NULL;
    int h_output = 0;

    // Allocate device memory for a single integer
    CHECK_CUDA(cudaMalloc((void **)&d_output, sizeof(int)));

    // Initialize device memory to zero
    CHECK_CUDA(cudaMemset(d_output, 0, sizeof(int)));

    // Launch kernel
    incrementKernel<<<numBlocks, threadsPerBlock>>>(d_output);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));

    // Print result
    printf("Result after atomicAdd: %d (expected %d)\n", h_output, totalThreads);

    // Clean up
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
