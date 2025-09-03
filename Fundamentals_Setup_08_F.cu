/*
Query and print the number of asynchronous engines the device has (`asyncEngineCount`).

--- Thinking ---
The task is to write a small CUDA program that queries each GPU device on the system and prints the value of the `asyncEngineCount` field from the device properties. The `asyncEngineCount` field indicates the number of asynchronous engines (often related to concurrent execution capability) that the GPU hardware supports.

Key steps:
1. Include the necessary headers: `<cuda_runtime.h>` for CUDA runtime API and `<iostream>` for printing.
2. Define a small helper macro `CUDA_CHECK` to wrap CUDA API calls and report errors.
3. In `main`, first call `cudaGetDeviceCount` to determine how many devices are present.
4. Iterate over each device index, call `cudaGetDeviceProperties` to fill a `cudaDeviceProp` struct.
5. Print the device number, name, and the `asyncEngineCount` value.
6. The program should return `EXIT_SUCCESS` on success, otherwise exit with an error code.

We'll write the code in C++ (valid in a .cu file). No external dependencies beyond the CUDA runtime and standard library. The program is self-contained and can be compiled with `nvcc`. */

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define CUDA_CHECK(call)                                                   \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                      << " code=" << static_cast<int>(err)                 \
                      << " (" << cudaGetErrorString(err) << "): "          \
                      << #call << std::endl;                               \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)

int main()
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found." << std::endl;
        return EXIT_SUCCESS;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        std::cout << "Device " << dev << ": " << prop.name << std::endl;
        std::cout << "  asyncEngineCount = " << prop.asyncEngineCount << std::endl;
        std::cout << std::endl;
    }

    return EXIT_SUCCESS;
}
