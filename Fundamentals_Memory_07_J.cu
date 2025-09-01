/*
Query the device property for `unifiedAddressing`. If true, explain what this means for host and device pointers.
- The program will query the CUDA device for the unifiedAddressing property using cudaGetDeviceProperties.
- If the property is set to 1 (true), it indicates that the device uses a unified memory address space.
- In a unified memory model, host and device pointers refer to the same address space, meaning:
  * The same pointer value can be used on the host and device without any translation.
  * Memory allocated with cudaMalloc can be accessed directly by the host (if pinned or via cudaHostAlloc) and vice versa.
  * Explicit cudaMemcpy calls are not required for data that resides in unified memory.
- If unifiedAddressing is false, the host and device have separate address spaces, requiring explicit memory copies between them.
- The program will print whether unified addressing is enabled and, if so, provide the explanation above.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Device: %s\n", prop.name);
    printf("Unified addressing: %s\n", prop.unifiedAddressing ? "Yes" : "No");

    if (prop.unifiedAddressing) {
        printf("\nExplanation:\n");
        printf("  Unified addressing is enabled: the host and device share a single address space.\n");
        printf("  - A single pointer value can be used on both the host and the device.\n");
        printf("  - Memory allocated with cudaMalloc can be accessed directly by the host\n");
        printf("    (if it is pinned or allocated via cudaHostAlloc) and vice versa.\n");
        printf("  - Explicit cudaMemcpy calls are not required for data that resides in unified memory.\n");
        printf("  - This simplifies programming but may have performance implications depending on the hardware.\n");
    }

    return 0;
}
