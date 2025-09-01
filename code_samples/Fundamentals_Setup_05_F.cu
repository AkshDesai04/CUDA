/*
Aim of the program: Write a program that prints all key resource limits for a block: max threads, max shared memory, and max registers.
Thinking: The CUDA Runtime API provides cudaDeviceGetAttribute to query various device limits. For a block we are interested in:
- cudaDevAttrMaxThreadsPerBlock  – maximum number of threads per block.
- cudaDevAttrMaxSharedMemoryPerBlock – maximum amount of shared memory (in bytes) per block.
- cudaDevAttrMaxRegistersPerBlock – maximum number of registers per block.
We will query these attributes for device 0, perform basic error checking, and print the results. To make the output clearer we also fetch the device name using cudaGetDeviceProperties. The program is written in standard C++ with CUDA runtime calls and should compile with `nvcc -arch=sm_XX`. The error checking macro `CHECK_CUDA` simplifies handling CUDA errors.
*/

#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                      << " code=" << static_cast<int>(err)                 \
                      << " \"" << cudaGetErrorString(err) << "\"\n";        \
            return EXIT_FAILURE;                                            \
        }                                                                   \
    } while (0)

int main() {
    int device = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    int maxThreadsPerBlock = 0;
    int maxSharedMemoryPerBlock = 0;
    int maxRegistersPerBlock = 0;

    CHECK_CUDA(cudaDeviceGetAttribute(&maxThreadsPerBlock,
                                      cudaDevAttrMaxThreadsPerBlock,
                                      device));
    CHECK_CUDA(cudaDeviceGetAttribute(&maxSharedMemoryPerBlock,
                                      cudaDevAttrMaxSharedMemoryPerBlock,
                                      device));
    CHECK_CUDA(cudaDeviceGetAttribute(&maxRegistersPerBlock,
                                      cudaDevAttrMaxRegistersPerBlock,
                                      device));

    std::cout << "CUDA Device: " << prop.name << "\n";
    std::cout << "Max threads per block:          " << maxThreadsPerBlock << "\n";
    std::cout << "Max shared memory per block:    " << maxSharedMemoryPerBlock << " bytes\n";
    std::cout << "Max registers per block:        " << maxRegistersPerBlock << "\n";

    return EXIT_SUCCESS;
}
