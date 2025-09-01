/*
 Aim: Write a program that prints a summary of the execution hierarchy limits: Max Grid [X, Y, Z] and Max Block [X, Y, Z].

 Thinking:
 1. Use CUDA runtime API to query device properties. The structure `cudaDeviceProp` contains the limits we need: `maxGridDimX`, `maxGridDimY`, `maxGridDimZ` for the grid dimensions and `maxBlockDimX`, `maxBlockDimY`, `maxBlockDimZ` for the block dimensions.
 2. The program will:
    - Query the number of CUDA devices.
    - Select the first device (device 0) for simplicity.
    - Retrieve its properties with `cudaGetDeviceProperties`.
    - Print the device name and the requested limits in a readable format.
 3. Include necessary headers: `<cuda_runtime.h>` for the CUDA API and `<iostream>` for console output.
 4. Perform minimal error checking: ensure a device exists and that the property query succeeds; otherwise, report the CUDA error string and exit with a non‑zero status.
 5. The code is written as a standard CUDA source file (`.cu`) so it can be compiled with `nvcc`. No device kernels are required; the program runs entirely on the host.
*/

#include <iostream>
#include <cuda_runtime.h>

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable device found." << std::endl;
        return 1;
    }

    int device = 0; // Use the first device
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Device " << device << " (" << prop.name << "):" << std::endl;
    std::cout << "Max Grid  [X, Y, Z]: [" << prop.maxGridDimX
              << ", " << prop.maxGridDimY
              << ", " << prop.maxGridDimZ << "]" << std::endl;
    std::cout << "Max Block [X, Y, Z]: [" << prop.maxBlockDimX
              << ", " << prop.maxBlockDimY
              << ", " << prop.maxBlockDimZ << "]" << std::endl;

    return 0;
}
