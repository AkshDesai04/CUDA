/*
Aim: Query and print the maximum number of registers available per block.

Thinking:
1. The CUDA runtime API provides a struct `cudaDeviceProp` which contains
   information about each GPU device. One of its members is `maxRegsPerBlock`,
   representing the maximum number of 32-bit registers that a single block
   may use.
2. We need to iterate over all CUDA-capable devices on the system, fetch
   their properties with `cudaGetDeviceProperties`, and output the value
   of `maxRegsPerBlock` for each device.
3. A small helper macro (`CHECK_CUDA`) will make error handling cleaner.
4. The program will include the necessary headers, perform the queries,
   print the results to `stdout`, and exit with 0 on success or a non-zero
   value on failure.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            return EXIT_FAILURE;                                            \
        }                                                                    \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return EXIT_SUCCESS;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        printf("Device %d (%s):\n", dev, prop.name);
        printf("  Maximum registers per block: %d\n", prop.maxRegsPerBlock);
    }

    return EXIT_SUCCESS;
}
