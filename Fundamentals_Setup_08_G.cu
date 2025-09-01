```cuda
/*
Aim: Check if the device is a multi-GPU board (`isMultiGpuBoard`).

Thinking:
The CUDA Runtime API exposes an attribute called
`cudaDevAttrMultiGpuBoard` via `cudaDeviceGetAttribute`.  
When this attribute is queried for a device, it returns 1 if the device
belongs to a multi‑GPU board and 0 otherwise.  We can use this attribute
to determine whether each device on the system is part of such a board.

The program below:
1. Uses `cudaGetDeviceCount` to obtain the number of CUDA capable devices.
2. Iterates over each device index.
3. Calls `cudaDeviceGetAttribute` with `cudaDevAttrMultiGpuBoard` to query
   the multi-GPU status.
4. Prints the result for each device.
5. Includes simple error checking for CUDA calls.

This code is self‑contained and can be compiled with `nvcc`:
    nvcc -o check_mgpu_board check_mgpu_board.cu
and then executed:
    ./check_mgpu_board
*/

#include <cuda_runtime.h>
#include <iostream>

// Helper macro for error checking
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " - " << cudaGetErrorString(err) << std::endl;       \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

int main() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found." << std::endl;
        return 0;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s)." << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        int isMultiGpuBoard = 0;
        // Query the attribute
        CUDA_CHECK(cudaDeviceGetAttribute(&isMultiGpuBoard,
                                          cudaDevAttrMultiGpuBoard,
                                          dev));

        std::cout << "Device " << dev << ": ";
        if (isMultiGpuBoard) {
            std::cout << "Is part of a multi-GPU board." << std::endl;
        } else {
            std::cout << "Is NOT part of a multi-GPU board." << std::endl;
        }
    }

    return 0;
}
```