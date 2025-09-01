/*
Write a program to determine and print the 'warp size' for the default CUDA device. The warp size is a fundamental property that dictates how threads are scheduled. Find this value in the device properties struct.

My approach:
1. Include necessary headers: <iostream> for standard I/O, <cuda_runtime.h> for CUDA runtime API.
2. In main, query the number of CUDA devices using cudaGetDeviceCount. Handle case where no device is found.
3. Use device ID 0 as the default device. (Could be made configurable but not required.)
4. Retrieve device properties via cudaGetDeviceProperties, passing device ID 0 and a struct of type cudaDeviceProp.
5. Print the warpSize field from the properties struct.
6. Include basic error checking for each CUDA call, printing the CUDA error string if an error occurs.
7. Return 0 on success.
The program is straightforward and uses only standard C++ and CUDA runtime API, suitable for compilation with nvcc.

*/

#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device count: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return EXIT_FAILURE;
    }

    int device = 0; // default device
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Default CUDA device (" << device << "): " << prop.name << std::endl;
    std::cout << "Warp size: " << prop.warpSize << std::endl;

    return EXIT_SUCCESS;
}
