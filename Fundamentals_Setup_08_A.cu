/*
Write a program that gets the driver version and prints its major and minor versions separately. (e.g., Driver version 11.4 -> Major: 11, Minor: 4).

Thinking:
- The CUDA driver API provides the function `cudaDriverGetVersion` which returns the driver version as an integer.
- The integer encoding follows the pattern `major * 1000 + minor * 10 + subminor`.
  For example, driver version 11.4 is encoded as 11040.
- To extract the major version, we can divide by 1000.
- To extract the minor version, we first take the remainder after dividing by 1000, then divide by 10.
  The subminor part is the remainder after dividing by 10, which we ignore for this task.
- We'll include proper error handling: check the return value of `cudaDriverGetVersion` and print a helpful message if it fails.
- We'll use the C++ standard library for output (`iostream`), but the code is fully compilable as a CUDA C/C++ program.
- The program will be compiled with `nvcc` and will not require any additional CUDA libraries beyond the runtime/driver headers.

*/

#include <cuda.h>
#include <iostream>

int main()
{
    int driverVersion = 0;
    cudaError_t err = cudaDriverGetVersion(&driverVersion);

    if (err != cudaSuccess) {
        std::cerr << "Failed to get CUDA driver version: "
                  << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Extract major and minor from the integer encoding
    int major = driverVersion / 1000;
    int minor = (driverVersion % 1000) / 10;

    std::cout << "Driver version " << major << "." << minor << std::endl;
    std::cout << "Major: " << major << ", Minor: " << minor << std::endl;

    return 0;
}
