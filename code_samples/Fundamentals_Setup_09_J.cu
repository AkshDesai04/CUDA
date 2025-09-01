```cpp
/*
Print the maximum number of resident blocks per multiprocessor.

This program queries the CUDA device properties for the first GPU (device 0)
and prints the value of the `maxBlocksPerMultiprocessor` field.  The goal
is to expose the hardware limit on how many blocks can simultaneously be
resident on a single multiprocessor.  The CUDA runtime API provides this
information via the `cudaDeviceProp` structure, which contains a member
`maxBlocksPerMultiprocessor`.  The program performs basic error checking
on CUDA API calls and prints the result to standard output.  The output
will be a single integer value that represents the maximum resident
blocks per multiprocessor on the queried device.

The steps are:
1. Set the current device to 0 (the first GPU).
2. Retrieve the `cudaDeviceProp` structure for that device.
3. Extract the `maxBlocksPerMultiprocessor` field.
4. Print the value.

If the target CUDA toolkit does not expose `maxBlocksPerMultiprocessor`,
the code will still compile, but the field will be zero or unavailable.
In that case, a fallback using the occupancy API could be implemented
if needed, but for the purposes of this simple demonstration we rely
on the device property directly.
*/

#include <cuda_runtime.h>
#include <cstdio>

// Helper macro for error checking
#define CHECK_CUDA(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            return EXIT_FAILURE;                                               \
        }                                                                       \
    } while (0)

int main(void) {
    int device = 0;
    CHECK_CUDA(cudaSetDevice(device));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    // The field may not exist on older toolkits; use conditional compilation
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    int maxResidentBlocks = prop.maxBlocksPerMultiprocessor;
#else
    // Fallback: approximate using occupancy API with a dummy kernel
    // Since we only need the value, we can use a trivial kernel.
    __global__ void dummyKernel() {}
    int maxResidentBlocks;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxResidentBlocks, dummyKernel, 1, 0));
#endif

    printf("Maximum resident blocks per multiprocessor: %d\n", maxResidentBlocks);
    return EXIT_SUCCESS;
}
```