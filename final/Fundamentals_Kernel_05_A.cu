/*
Aim: Change the launch configuration to 8 blocks of 32 threads and verify the global IDs are still unique and correct.

Thinking:
1. We need a CUDA kernel that writes each thread's global ID into an array at the index equal to that ID.
2. The global ID is computed as `blockIdx.x * blockDim.x + threadIdx.x`.  
   With 8 blocks and 32 threads per block, this will produce IDs from 0 to 255 inclusive.
3. On the host, allocate an array of 256 integers on both host and device.
4. Launch the kernel with `<<<8, 32>>>`.  
   After execution, copy the device array back to the host.
5. Verify that every entry in the array equals its index.  
   If all match, print a success message; otherwise report mismatches.
6. Include simple error checking for CUDA API calls to aid debugging.
7. The code is self‑contained, uses only the CUDA runtime API and standard C headers, and prints the result to stdout.
*/

#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel that writes each thread's global ID into an array
__global__ void writeGlobalIds(int *arr, int totalThreads) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < totalThreads) {
        arr[gid] = gid;
    }
}

int main(void) {
    const int blocks = 8;
    const int threadsPerBlock = 32;
    const int totalThreads = blocks * threadsPerBlock; // 256

    // Allocate host memory
    int *h_array = (int *)malloc(totalThreads * sizeof(int));
    if (!h_array) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    int *d_array = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_array, totalThreads * sizeof(int)));

    // Launch kernel
    writeGlobalIds<<<blocks, threadsPerBlock>>>(d_array, totalThreads);
    CUDA_CHECK(cudaGetLastError());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_array, d_array, totalThreads * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify that all IDs are unique and correct
    int errors = 0;
    for (int i = 0; i < totalThreads; ++i) {
        if (h_array[i] != i) {
            fprintf(stderr, "Mismatch at index %d: expected %d, got %d\n", i, i, h_array[i]);
            errors++;
        }
    }

    if (errors == 0) {
        printf("Success: All %d global IDs are unique and correct.\n", totalThreads);
    } else {
        printf("Error: %d mismatches found.\n", errors);
    }

    // Clean up
    free(h_array);
    CUDA_CHECK(cudaFree(d_array));

    return (errors == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
