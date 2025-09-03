/*
Aim: Query and print the maximum z-dimension of a thread block.

Thinking:
- CUDA runtime provides `cudaDeviceProp` with `maxThreadsDim[3]` for the maximum block size in each dimension.
- Use `cudaGetDeviceCount` to find available GPUs, then query each device with `cudaGetDeviceProperties`.
- Print the z-dimension (`maxThreadsDim[2]`) for each device.
- Include basic error handling via a macro.
- The program is simple and uses standard I/O for output.
*/

#include <iostream>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CHECK_CUDA(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error in file '" << __FILE__                \
                      << "' in line " << __LINE__ << ": "                  \
                      << cudaGetErrorString(err) << std::endl;             \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)

int main() {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found." << std::endl;
        return EXIT_FAILURE;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        std::cout << "Device " << dev << " (" << prop.name << "): "
                  << "Maximum z-dimension of a thread block = "
                  << prop.maxThreadsDim[2] << std::endl;
    }

    return EXIT_SUCCESS;
}
