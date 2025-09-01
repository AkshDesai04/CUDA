```cpp
/*
Aim: Write the global ID to an output array at the corresponding index: `output[id] = id;`. Verify the result on the host.

Thinking:
The goal is to demonstrate a simple CUDA kernel that writes each thread’s global identifier into a device array at the same index.  
We’ll:

1. Define a kernel that calculates the global ID (`blockIdx.x * blockDim.x + threadIdx.x`) and stores it in `output[id]`.  
2. On the host, allocate a device array large enough for N elements, and launch the kernel with a grid/blocks configuration that covers all indices.  
3. After kernel execution, copy the result back to the host and iterate over the array to verify that `output[i] == i` for every element.  
4. Use CUDA error checking for robustness and print a simple success or failure message.  
5. Keep the example minimal and self‑contained in a single .cu file.

This satisfies the requirement to write the global ID into the array and verify it on the host.
*/

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024                 // Size of the output array
#define BLOCK_SIZE 256         // Threads per block

// CUDA kernel: each thread writes its global ID to output[id]
__global__ void writeGlobalID(int *output) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        output[id] = id;
    }
}

// Macro for checking CUDA errors following a CUDA API call
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",      \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),            \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

int main(void) {
    int *d_output = NULL;
    int *h_output = (int *)malloc(N * sizeof(int));
    if (h_output == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void **)&d_output, N * sizeof(int)));

    // Determine grid size
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch kernel
    writeGlobalID<<<gridSize, BLOCK_SIZE>>>(d_output);
    CHECK_CUDA(cudaGetLastError());   // Check kernel launch errors
    CHECK_CUDA(cudaDeviceSynchronize()); // Wait for kernel to finish

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify results
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_output[i] != i) {
            fprintf(stderr, "Verification failed at index %d: got %d, expected %d\n",
                    i, h_output[i], i);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Verification passed: all values are correct.\n");
    }

    // Clean up
    free(h_output);
    CHECK_CUDA(cudaFree(d_output));

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
```