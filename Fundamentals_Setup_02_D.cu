/*
Aim: Iterate through all devices and print the maximum number of threads per block for each.
In order to achieve this, the program first queries the number of CUDA-capable devices
present on the system using cudaGetDeviceCount. It then iterates over each device
by index. For each device, it retrieves the device properties via cudaGetDeviceProperties,
which contains the field maxThreadsPerBlock that represents the maximum number of threads
that can be launched in a single block on that device. The program prints the device
index, name, and this maximum thread count. It also includes basic error handling
to catch and report CUDA API failures, and a fallback message when no CUDA devices
are detected. The implementation uses standard C++ I/O for output and is
intended to be compiled with nvcc as a .cu file.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Error retrieving device count: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found." << std::endl;
        return 0;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s)." << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            std::cerr << "Error retrieving properties for device " << dev
                      << ": " << cudaGetErrorString(err) << std::endl;
            continue;
        }

        std::cout << "Device " << dev << ": " << prop.name << std::endl;
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    }

    return 0;
}
