/*
Aim: Query and print the number of asynchronous engines the device has (`asyncEngineCount`).

Thoughts:
- The CUDA runtime API provides a structure `cudaDeviceProp` that contains a member `asyncEngineCount`.
- To obtain this value we need to query the properties of a CUDA device.
- We'll first determine how many CUDA-capable devices are present with `cudaGetDeviceCount`.
- For simplicity and because the prompt refers to "the device", we'll query the first device (index 0). 
- We must check for errors after each CUDA API call and output a helpful message if something goes wrong.
- The program will print the device index, its name, and the value of `asyncEngineCount` to standard output.
- This code is self-contained and can be compiled with `nvcc` (e.g., `nvcc async_engine_count.cu -o async_engine_count`).
*/

#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return EXIT_FAILURE;
    }

    int dev = 0; // query the first device
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, dev);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaGetDeviceProperties failed for device " << dev
                  << ": " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Device " << dev << " (" << prop.name << ") has "
              << prop.asyncEngineCount << " asynchronous engine(s)." << std::endl;

    return EXIT_SUCCESS;
}
