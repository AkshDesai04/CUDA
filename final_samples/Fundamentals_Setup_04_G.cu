```cuda
/*
Aim: Query and print the maximum number of blocks that can reside on a single multiprocessor.

Thinking:
- The CUDA Runtime API offers a way to retrieve device attributes.
- The attribute `cudaDevAttrMaxBlocksPerMultiProcessor` tells us the maximum number of resident blocks per multiprocessor.
- Alternatively, the `cudaDeviceProp` structure contains a field `maxBlocksPerMultiProcessor`, but querying via `cudaDeviceGetAttribute` is more explicit and works across all CUDA versions that expose this attribute.
- We'll query this attribute for the default device (device 0) and print the value.
- The program is intentionally simple: include `<stdio.h>` and `<cuda_runtime.h>`, get the attribute, handle errors, and output the integer.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int device = 0;  // default device
    int maxBlocks = 0;
    cudaError_t err = cudaDeviceGetAttribute(&maxBlocks,
                                            cudaDevAttrMaxBlocksPerMultiProcessor,
                                            device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error retrieving device attribute: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Maximum number of blocks per multiprocessor on device %d: %d\n", device, maxBlocks);
    return 0;
}
```