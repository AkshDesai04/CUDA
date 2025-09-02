/*
Aim: Query and print the GPU's core clock rate in GHz.

Thought process:
- The CUDA Runtime API provides device properties via cudaGetDeviceProperties.
- The property `clockRate` is expressed in kilohertz (kHz).
- To convert this to gigahertz (GHz), divide by 1,000,000.
- The program will first query the number of CUDA-capable devices with cudaGetDeviceCount.
- For each device, retrieve its properties and calculate the clock rate in GHz.
- Output the results to standard output.
- Basic error checking is performed after each CUDA API call.
- This code uses only standard C++ libraries and the CUDA Runtime API, so it can be compiled with nvcc.
*/

#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Error getting device count: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found." << std::endl;
        return EXIT_SUCCESS;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            std::cerr << "Error getting properties for device " << dev << ": " << cudaGetErrorString(err) << std::endl;
            continue;
        }

        double clockGHz = static_cast<double>(prop.clockRate) / 1e6; // prop.clockRate is in kHz
        std::cout << "Device " << dev << " (" << prop.name << "): Core clock rate = "
                  << clockGHz << " GHz" << std::endl;
    }

    return EXIT_SUCCESS;
}
