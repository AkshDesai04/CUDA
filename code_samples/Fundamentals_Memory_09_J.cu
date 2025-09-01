```cpp
/* 
Aim of the program: Create a table in your output showing Pageable HtoD, Pinned HtoD, Pageable DtoH, and Pinned DtoH bandwidths.

Thinking process:
1. Need to benchmark four memory copy scenarios on CUDA:
   - Pageable host to device (HtoD)
   - Pinned host to device (HtoD)
   - Pageable device to host (DtoH)
   - Pinned device to host (DtoH)
2. Use a reasonably large transfer size (64 MiB) to get measurable bandwidth.
3. Allocate:
   - Pageable host memory with malloc.
   - Pinned host memory with cudaMallocHost.
   - Device memory with cudaMalloc.
4. Use cudaEvent_t for precise timing (cudaEventRecord + cudaEventElapsedTime).
5. Compute bandwidth as (bytes * 1e-9) / (ms / 1000) to get GB/s.
6. Print a clean ASCII table to stdout, matching the requested transfer types.
7. Include basic error checking for CUDA calls to ensure correct execution.
8. Keep the code self‑contained: single .cu file, compile with nvcc.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

int main(void) {
    const size_t transferSize = 64ULL * 1024ULL * 1024ULL; // 64 MiB
    const size_t numElements = transferSize / sizeof(float);

    // Allocate pageable host memory
    float *hostPageable = (float*)malloc(transferSize);
    if (!hostPageable) {
        fprintf(stderr, "Failed to allocate pageable host memory\n");
        return EXIT_FAILURE;
    }

    // Allocate pinned host memory
    float *hostPinned;
    CHECK_CUDA(cudaMallocHost((void**)&hostPinned, transferSize));

    // Allocate device memory
    float *deviceMem;
    CHECK_CUDA(cudaMalloc((void**)&deviceMem, transferSize));

    // Initialize host buffers with dummy data
    for (size_t i = 0; i < numElements; ++i) {
        hostPageable[i] = static_cast<float>(i);
        hostPinned[i]   = static_cast<float>(i);
    }

    // Events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float msPageableHtoD = 0.0f, msPinnedHtoD = 0.0f;
    float msPageableDtoH = 0.0f, msPinnedDtoH = 0.0f;

    // Pageable HtoD
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(deviceMem, hostPageable, transferSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&msPageableHtoD, start, stop));

    // Pinned HtoD
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(deviceMem, hostPinned, transferSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&msPinnedHtoD, start, stop));

    // Pageable DtoH
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(hostPageable, deviceMem, transferSize, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&msPageableDtoH, start, stop));

    // Pinned DtoH
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(hostPinned, deviceMem, transferSize, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&msPinnedDtoH, start, stop));

    // Compute bandwidths in GB/s
    double bwPageableHtoD = (transferSize * 1e-9) / (msPageableHtoD / 1000.0);
    double bwPinnedHtoD   = (transferSize * 1e-9) / (msPinnedHtoD   / 1000.0);
    double bwPageableDtoH = (transferSize * 1e-9) / (msPageableDtoH / 1000.0);
    double bwPinnedDtoH   = (transferSize * 1e-9) / (msPinnedDtoH   / 1000.0);

    // Print table
    printf("+----------------+------------------+\n");
    printf("| Transfer Type  | Bandwidth (GB/s) |\n");
    printf("+----------------+------------------+\n");
    printf("| Pageable HtoD  | %10.2f |\n", bwPageableHtoD);
    printf("| Pinned HtoD    | %10.2f |\n", bwPinnedHtoD);
    printf("| Pageable DtoH  | %10.2f |\n", bwPageableDtoH);
    printf("| Pinned DtoH    | %10.2f |\n", bwPinnedDtoH);
    printf("+----------------+------------------+\n");

    // Clean up
    free(hostPageable);
    CHECK_CUDA(cudaFreeHost(hostPinned));
    CHECK_CUDA(cudaFree(deviceMem));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return EXIT_SUCCESS;
}
```