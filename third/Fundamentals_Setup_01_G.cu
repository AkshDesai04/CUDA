```cpp
/*
Query and print the total number of registers available per thread block on the primary device.

I need to write a CUDA C program that retrieves the properties of the first GPU device (device 0) and prints the number of registers that can be used by a single thread block. CUDA provides the struct cudaDeviceProp which includes the field regsPerBlock. According to the CUDA documentation, regsPerBlock represents the number of registers per block in units of 32. The actual number of 32-bit registers is therefore regsPerBlock * 32. I will output both the raw value and the multiplied value for clarity. The program will be very simple: include <cuda_runtime.h> and <iostream>, call cudaGetDeviceProperties, handle any error, then print the desired information. This code should compile with nvcc and run on a system that has a CUDA-capable GPU.
*/

#include <cuda_runtime.h>
#include <iostream>

int main() {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);  // primary device
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Total number of registers available per thread block (raw units of 32): "
              << prop.regsPerBlock << std::endl;

    // Convert to actual number of 32-bit registers
    int totalRegisters = prop.regsPerBlock * 32;
    std::cout << "Total number of registers available per thread block (actual count): "
              << totalRegisters << std::endl;

    return 0;
}
```