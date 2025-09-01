```cuda
/*
Aim of the program: Query and print the maximum number of blocks that can reside on a single multiprocessor.

Thinking process:
1. The CUDA runtime API provides device properties via cudaDeviceProp.
2. One of the fields is `maxBlocksPerMultiprocessor` (in newer CUDA versions),
   which indicates the maximum number of resident blocks per SM.
3. The program will:
   - Determine the number of CUDA devices.
   - For each device, retrieve its properties.
   - Print the `maxBlocksPerMultiprocessor` value.
4. Error handling will be added to catch any failures in API calls.
5. The code is self‑contained and compiles as a .cu file with nvcc.

Note: If your CUDA toolkit does not expose `maxBlocksPerMultiprocessor`,
   the program will compile but may report a default or zero value.
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n",
                    dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: \"%s\"\n", dev, prop.name);
        printf("  Multiprocessor count            : %d\n", prop.multiProcessorCount);
        printf("  Max threads per block           : %d\n", prop.maxThreadsPerBlock);
        printf("  Max blocks per multiprocessor   : %d\n", prop.maxBlocksPerMultiprocessor);
        printf("\n");
    }

    return EXIT_SUCCESS;
}
```