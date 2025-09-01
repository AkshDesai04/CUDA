/*
 * Aim of the program: Query and print the maximum number of blocks that can reside on a single multiprocessor.
 *
 * Thinking process:
 * 1. Identify the CUDA runtime API attribute that provides the maximum number of resident blocks per SM.
 *    - CUDA provides cudaDevAttrMaxBlocksPerMultiprocessor (driver API: CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR).
 * 2. Use cudaDeviceGetAttribute to query this attribute for device 0 (or any specified device).
 * 3. Handle possible errors from the API call.
 * 4. Print the retrieved value to standard output.
 * 5. Keep the program minimal and self-contained: include necessary headers, define an error checking macro, and implement main().
 *
 * Note: The value returned is a theoretical maximum based on the GPU architecture and is not necessarily the actual limit in a specific launch configuration.
 */

#include <iostream>
#include <cuda_runtime.h>

inline void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main()
{
    int device = 0; // Use the first CUDA device
    cudaError_t err = cudaSetDevice(device);
    checkCudaError(err, "Failed to set CUDA device");

    int maxBlocksPerSM = 0;
    err = cudaDeviceGetAttribute(&maxBlocksPerSM,
                                 cudaDevAttrMaxBlocksPerMultiprocessor,
                                 device);
    checkCudaError(err, "Failed to get device attribute cudaDevAttrMaxBlocksPerMultiprocessor");

    std::cout << "Maximum number of resident blocks per multiprocessor on device " << device
              << ": " << maxBlocksPerSM << std::endl;

    return 0;
}
