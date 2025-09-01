```cpp
/*
Aim of the program: Print the amount of shared memory available per multiprocessor.

Thinking:
The goal is to query the CUDA device properties to determine how much shared memory each multiprocessor has. The CUDA runtime provides a struct, cudaDeviceProp, which contains the field sharedMemoryPerMultiprocessor. To obtain this, we need to:
1. Identify the active GPU device. If the program runs on a system with multiple GPUs, we can use cudaGetDevice to retrieve the current device index.
2. Allocate a cudaDeviceProp instance and call cudaGetDeviceProperties to fill it with information about the chosen device.
3. Access the sharedMemoryPerMultiprocessor field and print it in a human-readable format.
4. Handle any potential errors by checking the return codes of CUDA API calls and printing appropriate error messages.

This program is intentionally simple and uses only CUDA runtime functions. No device code or kernels are required. We also add minimal error handling for robustness.
*/

#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device count: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found." << std::endl;
        return 1;
    }

    int device = 0;
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get current device: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Device " << device << ": " << prop.name << std::endl;
    std::cout << "Shared memory per multiprocessor: " 
              << prop.sharedMemoryPerMultiprocessor << " bytes" << std::endl;

    return 0;
}
```