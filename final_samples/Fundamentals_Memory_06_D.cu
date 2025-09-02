/*
Demonstrate a different kind of leak: allocate with `cudaMallocHost` (pinned memory) and forget to free with `cudaFreeHost`.

Thought process:
- The goal is to illustrate a memory leak that occurs when we allocate pinned host memory using cudaMallocHost but never deallocate it with cudaFreeHost. 
- Pinned memory is special in that it is page-locked and typically more expensive to allocate; forgetting to free it can lead to exhaustion of the system's virtual memory or CUDA driver resources. 
- To show this, we create a small CUDA program that:
  1. Allocates a block of pinned host memory.
  2. Fills it with some data.
  3. Allocates a device buffer and copies the data to the device.
  4. Prints confirmation that the copy succeeded.
  5. Deliberately omits the cudaFreeHost call, leaving the allocated pinned memory in a leaked state.
- The program includes basic error checking so that if the allocation or copy fails, we get a meaningful message. 
- After running this program, one could inspect the process memory usage or use CUDA tools (like cuda-memcheck or nvprof) to confirm the presence of the leak. 
- The code is self‑contained and can be compiled with nvcc: `nvcc pinned_leak.cu -o pinned_leak`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CHECK_CUDA(call)                                 \
    do {                                                 \
        cudaError_t err = call;                          \
        if (err != cudaSuccess) {                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__,                  \
                    cudaGetErrorString(err));            \
            return EXIT_FAILURE;                         \
        }                                                \
    } while (0)

int main(void)
{
    const size_t size = 1 << 20; // 1 MiB
    float *pinnedHostPtr = NULL;
    float *devicePtr = NULL;

    // Allocate pinned host memory
    CHECK_CUDA(cudaMallocHost((void **)&pinnedHostPtr, size * sizeof(float)));
    printf("Pinned host memory allocated at %p, size %zu bytes.\n", (void *)pinnedHostPtr, size * sizeof(float));

    // Initialize host memory
    for (size_t i = 0; i < size; ++i) {
        pinnedHostPtr[i] = (float)i;
    }

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void **)&devicePtr, size * sizeof(float)));

    // Copy data from pinned host to device
    CHECK_CUDA(cudaMemcpy(devicePtr, pinnedHostPtr, size * sizeof(float), cudaMemcpyHostToDevice));
    printf("Copied data from pinned host to device.\n");

    // Normally we would free the pinned memory here:
    // cudaFreeHost(pinnedHostPtr);
    // But we intentionally omit it to create a memory leak.

    // Clean up device memory
    CHECK_CUDA(cudaFree(devicePtr));
    printf("Device memory freed.\n");

    // Final synchronization to ensure all operations complete before exit
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("Program exiting without freeing pinned host memory (leak intentional).\n");
    return EXIT_SUCCESS;
}
