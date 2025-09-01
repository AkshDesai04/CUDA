/*
Aim: Query and check if the device supports Page-locked Memory Mapped On The GPU (`pageableMemoryAccess`).

My approach:
1. Include the necessary headers: <stdio.h> for printing and <cuda_runtime.h> for CUDA runtime API.
2. Use cudaGetDeviceCount() to find how many CUDA capable devices are present.
3. Loop over each device index, retrieve its properties via cudaGetDeviceProperties().
4. The cudaDeviceProp struct contains a boolean field called pageableMemoryAccess. 
   - If pageableMemoryAccess == 1, the device can directly access pageable memory (host memory) without explicit mapping.
   - If 0, the device cannot.
5. Print the device number, name, and whether pageableMemoryAccess is supported.
6. Perform basic error checking on each CUDA API call to ensure robust execution.
7. Keep the code simple and self‑contained, suitable for compiling with nvcc as a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Helper macro for error checking */
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            return EXIT_FAILURE;                                          \
        }                                                                  \
    } while (0)

int main(void)
{
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Found %d CUDA-capable device(s).\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Page-locked Memory Mapped On The GPU (pageableMemoryAccess): %s\n",
               prop.pageableMemoryAccess ? "Supported" : "Not Supported");

        /* Optional: print related properties for more context */
        printf("  pageableMemoryAccessUsesHostPageTables: %s\n",
               prop.pageableMemoryAccessUsesHostPageTables ? "Yes" : "No");
        printf("  RequiresMappedArray: %s\n",
               prop.requiresMappedArray ? "Yes" : "No");
        printf("\n");
    }

    return EXIT_SUCCESS;
}
