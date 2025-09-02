```cuda
/* 
Aim: Write a program that prints a summary of the execution hierarchy limits: Max Grid [X, Y, Z] and Max Block [X, Y, Z].

Thinking:
The goal is to query the CUDA device properties for the maximum grid and block dimensions and display them to the user. 
CUDA provides a convenient structure, cudaDeviceProp, that contains fields such as maxGridSize[3] for the maximum
grid dimensions in X, Y, and Z, and maxThreadsDim[3] for the maximum block dimensions. Additionally, maxThreadsPerBlock
gives the total number of threads allowed in a single block. 
To access these properties, we use cudaGetDeviceProperties, which requires a device ID. For simplicity, we will
use device 0 (the first CUDA device). We should also check for errors after CUDA API calls. The program will be a
stand‑alone .cu file that compiles with nvcc. It will print the max grid and block sizes in a clear, human‑readable format.
*/

#include <iostream>
#include <cuda_runtime.h>

// Macro for CUDA error checking
#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess) {                                     \
            std::cerr << "CUDA error in file '" << __FILE__          \
                      << "' in line " << __LINE__ << ": "             \
                      << cudaGetErrorString(err) << std::endl;        \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

int main() {
    // Get the number of CUDA devices
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return EXIT_FAILURE;
    }

    // Use the first device (device 0)
    int device = 0;
    cudaDeviceProp deviceProp;
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, device));

    // Print max grid dimensions
    std::cout << "Max Grid [X, Y, Z]: ["
              << deviceProp.maxGridSize[0] << ", "
              << deviceProp.maxGridSize[1] << ", "
              << deviceProp.maxGridSize[2] << "]" << std::endl;

    // Print max block dimensions
    std::cout << "Max Block [X, Y, Z]: ["
              << deviceProp.maxThreadsDim[0] << ", "
              << deviceProp.maxThreadsDim[1] << ", "
              << deviceProp.maxThreadsDim[2] << "]" << std::endl;

    // Optionally, print total threads per block
    std::cout << "Max Threads Per Block: " << deviceProp.maxThreadsPerBlock << std::endl;

    return EXIT_SUCCESS;
}
```