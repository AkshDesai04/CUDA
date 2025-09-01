/*
Write a loop that iterates through all available devices and calls `cudaSetDevice` followed by `cudaDeviceReset` for each one.

Thinking process:
1. Use the CUDA Runtime API to first query the number of devices with cudaGetDeviceCount().
2. Iterate over each device index from 0 to count-1.
3. For each device:
   a. Call cudaSetDevice(index) to make the device current.
   b. Call cudaDeviceReset() to reset the device.
   c. Check the return status of both calls; if any call fails, print an error message and continue with the next device.
4. After resetting all devices, print a completion message.
5. The program is a simple command‑line utility written in C++ (since CUDA’s headers integrate naturally with C++), but it can be compiled as a CUDA source (.cu) file.
6. No kernel launches or device memory allocations are required; the focus is solely on device selection and reset.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <string>

// Helper macro for checking CUDA errors
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            std::cerr << "CUDA error at " << __FILE__ << ":"        \
                      << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        }                                                           \
    } while (0)

int main() {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found." << std::endl;
        return 0;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s)." << std::endl;

    for (int device = 0; device < deviceCount; ++device) {
        std::cout << "Setting device " << device << "..." << std::endl;
        CHECK_CUDA(cudaSetDevice(device));

        std::cout << "Resetting device " << device << "..." << std::endl;
        CHECK_CUDA(cudaDeviceReset());

        std::cout << "Device " << device << " reset successfully." << std::endl;
    }

    std::cout << "All devices have been reset." << std::endl;
    return 0;
}
