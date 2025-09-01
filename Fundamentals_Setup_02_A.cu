```cpp
/* 
Aim: Iterate through all available CUDA devices and print their total global memory in gigabytes (GiB) for each.

Thinking:
1. We need to query the number of CUDA-capable devices using cudaGetDeviceCount.
2. For each device index from 0 to count-1, retrieve device properties via cudaGetDeviceProperties.
3. The total global memory is stored in the 'totalGlobalMem' field (in bytes).
4. Convert bytes to GiB by dividing by (1024^3). We'll use a double for precision.
5. Print the result using std::cout with fixed-point notation and two decimal places.
6. Include necessary headers: <iostream>, <iomanip> for formatting, and <cuda_runtime.h> for CUDA API.
7. We'll wrap the code in a main function that performs the loop and prints each device's memory.
8. Error handling: if cudaGetDeviceCount returns an error, we'll print it and exit.
9. The program will be a simple console application compiled with nvcc.
*/

#include <iostream>
#include <iomanip>
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

        double memGiB = static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);
        std::cout << "Device " << dev << " (" << prop.name << "): "
                  << std::fixed << std::setprecision(2)
                  << memGiB << " GiB" << std::endl;
    }

    return EXIT_SUCCESS;
}
```