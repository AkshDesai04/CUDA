/*
Query and print the maximum number of threads allowed in a single thread block for the primary CUDA device. This value, `maxThreadsPerBlock`, is a key constraint when designing kernels.

Thinking process:
1. The aim is to retrieve `maxThreadsPerBlock` from the primary CUDA device (device 0) using CUDA Runtime API.
2. Use `cudaGetDeviceProperties` which fills a `cudaDeviceProp` structure with device attributes, including `maxThreadsPerBlock`.
3. Perform error checking on CUDA calls to handle possible failures gracefully.
4. Print the retrieved value to standard output.
5. Keep the program simple: include necessary headers, implement `main`, and return 0 on success.
6. Ensure the code is valid CUDA C++ and can be compiled with `nvcc` producing a .cu file.
7. No additional functionality or dependencies required.
*/

#include <cuda_runtime.h>
#include <iostream>

int main() {
    int device = 0;  // primary CUDA device
    cudaDeviceProp prop;

    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Maximum threads per block for device " << device
              << ": " << prop.maxThreadsPerBlock << std::endl;

    return 0;
}
