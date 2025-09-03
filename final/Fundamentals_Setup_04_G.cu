```cpp
/*
Query and print the maximum number of blocks that can reside on a single multiprocessor.

The goal of this program is to determine, for the current CUDA device, how many
active blocks a single streaming multiprocessor (SM) can host simultaneously.
CUDA provides the helper function `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
which, given a kernel, block size, and dynamic shared memory usage, returns the
maximum number of resident blocks per SM that the runtime can schedule.
To use this function we need:

1. A simple kernel (it can do nothing; the occupancy depends only on launch
   configuration, not on kernel body).
2. The block size that will be used for the launch.  A natural choice is the
   maximum number of threads per block supported by the device, which we obtain
   from `cudaDeviceProp::maxThreadsPerBlock`.  Using the maximum block size
   typically yields the highest possible resident block count, although the
   exact value also depends on other resource limits such as shared memory
   and registers.
3. An error‑checking macro to keep the code concise and robust.

The program performs the following steps:
- Retrieves the current device and its properties.
- Defines a dummy kernel.
- Calls `cudaOccupancyMaxActiveBlocksPerMultiprocessor` to compute the
  occupancy for the chosen block size (0 dynamic shared memory).
- Prints the device name, the selected block size, and the resulting
  maximum number of resident blocks per SM.

This output directly answers the query: it shows the maximum number of
blocks that can simultaneously reside on a single multiprocessor for the
current device configuration.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Simple error‑checking macro
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            return EXIT_FAILURE;                                            \
        }                                                                    \
    } while (0)

// Dummy kernel – body is irrelevant for occupancy calculation
__global__ void dummyKernel()
{
    // No operation
}

int main(void)
{
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("Device %d: %s\n", device, prop.name);

    // Use the maximum threads per block supported by the device
    int blockSize = prop.maxThreadsPerBlock;

    int maxBlocksPerSM = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        dummyKernel,
        blockSize,
        0  // dynamic shared memory per block
    );

    if (err != cudaSuccess) {
        fprintf(stderr, "cudaOccupancyMaxActiveBlocksPerMultiprocessor error: %s\n",
                cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Using block size: %d threads\n", blockSize);
    printf("Maximum number of resident blocks per SM: %d\n", maxBlocksPerSM);

    return EXIT_SUCCESS;
}
```