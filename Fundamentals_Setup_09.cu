/*
Write a program to check and print whether the primary GPU supports running multiple kernels concurrently. The property you are looking for is `concurrentKernels` in `cudaDeviceProp`.

I thought about how to structure the program: the goal is simply to query the device properties for the primary GPU (device 0) and examine the `concurrentKernels` flag. If that flag is set to 1, the GPU can run multiple kernels concurrently; otherwise it cannot. I want to keep the code minimal, but also robust: I will check for errors from the CUDA runtime API calls, and print clear messages indicating the status. I will also print the device name and the flag value for completeness.

The program will:

1. Use `cudaGetDeviceCount` to make sure at least one GPU is present.
2. Use `cudaGetDeviceProperties` to fetch the properties of device 0.
3. Inspect `prop.concurrentKernels`.
4. Print a message describing whether concurrent kernels are supported.
5. Handle any CUDA errors gracefully.

I will also include a small helper macro `CUDA_CHECK` to simplify error handling. The program can be compiled with `nvcc` and run on any system with CUDA drivers installed.

*/

#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                      << " - " << cudaGetErrorString(err) << std::endl;     \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found." << std::endl;
        return EXIT_FAILURE;
    }

    int device = 0; // primary GPU
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "Device " << device << ": " << prop.name << std::endl;
    std::cout << "  concurrentKernels: " << prop.concurrentKernels << std::endl;

    if (prop.concurrentKernels) {
        std::cout << "This GPU supports running multiple kernels concurrently." << std::endl;
    } else {
        std::cout << "This GPU does NOT support running multiple kernels concurrently." << std::endl;
    }

    return EXIT_SUCCESS;
}
