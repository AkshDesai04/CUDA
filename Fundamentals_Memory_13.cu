```c
/*
Aim: Demonstrate a `cudaErrorMemoryAllocation` error by attempting to allocate an impossibly large amount of memory on the device (e.g., 100 GB). Catch and report the error gracefully.

Thinking:
1. We want to trigger `cudaErrorMemoryAllocation` by asking for more memory than the GPU can provide. 100 GB is far beyond typical device memory.
2. We'll query the device memory using `cudaMemGetInfo` to display the available memory before the allocation attempt.
3. We'll use `cudaMalloc` to request 100 GB of device memory and check the returned `cudaError_t` code.
4. If the allocation fails, we'll print the error string using `cudaGetErrorString` and exit gracefully.
5. For clarity and reusability, we wrap error checking in a simple macro `checkCudaErrors`.
6. The program is written in C (using CUDA runtime API) and compiles as a .cu file. No external libraries or frameworks are required.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* Helper macro for error checking */
#define checkCudaErrors(call)                                    \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

int main(void)
{
    /* Print device information */
    int deviceCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return EXIT_FAILURE;
    }

    int dev = 0;
    checkCudaErrors(cudaSetDevice(dev));

    /* Get free and total memory */
    size_t freeMem, totalMem;
    checkCudaErrors(cudaMemGetInfo(&freeMem, &totalMem));

    printf("Device %d: Total memory = %.2f GB, Free memory = %.2f GB\n",
           dev, (double)totalMem / (1024.0 * 1024.0 * 1024.0),
           (double)freeMem  / (1024.0 * 1024.0 * 1024.0));

    /* Attempt to allocate 100 GB */
    const size_t hugeSize = 100ULL * 1024ULL * 1024ULL * 1024ULL; // 100 GiB
    void *d_ptr = NULL;
    printf("Attempting to allocate %.2f GB on device...\n",
           (double)hugeSize / (1024.0 * 1024.0 * 1024.0));

    cudaError_t status = cudaMalloc(&d_ptr, hugeSize);
    if (status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(status));
        if (status == cudaErrorMemoryAllocation) {
            fprintf(stderr, "Error: Not enough device memory available.\n");
        } else {
            fprintf(stderr, "Error: An unexpected CUDA error occurred.\n");
        }
        return EXIT_FAILURE;
    }

    /* If allocation somehow succeeded (unlikely) */
    printf("Allocation succeeded (unexpected). Cleaning up.\n");
    checkCudaErrors(cudaFree(d_ptr));

    return EXIT_SUCCESS;
}
```