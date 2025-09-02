```cuda
/*
Write a program that prints all key resource limits for a block: max threads, max shared memory, and max registers.
Thinking:
1. We need to query CUDA device attributes that give the maximum values for a block.
2. The relevant attributes are:
   - cudaDevAttrMaxThreadsPerBlock (maximum number of threads that can be launched in a single block)
   - cudaDevAttrMaxSharedMemoryPerBlock (maximum amount of shared memory per block, in bytes)
   - cudaDevAttrMaxRegistersPerBlock (maximum number of registers per block)
3. Use cudaGetDevice and cudaDeviceGetAttribute to obtain these values.
4. Print them to stdout. Handle error checking for each CUDA call.
5. The program will be a simple command-line utility that compiles with nvcc.
6. Include necessary headers: <cuda_runtime.h>, <iostream>, <string>.
7. Ensure proper error handling: if any cudaGet* call fails, print an error message and exit with non-zero status.
8. Use a single function main that performs all logic.
9. No additional functionality or support files needed.
10. The output will be human-readable, e.g.:
    Max threads per block: 1024
    Max shared memory per block: 49152 bytes
    Max registers per block: 65536
*/

#include <cuda_runtime.h>
#include <iostream>
#include <string>

int main() {
    int device = 0; // Use the first CUDA device
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get CUDA device: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    int maxThreadsPerBlock;
    err = cudaDeviceGetAttribute(&maxThreadsPerBlock,
                                 cudaDevAttrMaxThreadsPerBlock,
                                 device);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get Max Threads Per Block: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    int maxSharedMemoryPerBlock;
    err = cudaDeviceGetAttribute(&maxSharedMemoryPerBlock,
                                 cudaDevAttrMaxSharedMemoryPerBlock,
                                 device);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get Max Shared Memory Per Block: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    int maxRegistersPerBlock;
    err = cudaDeviceGetAttribute(&maxRegistersPerBlock,
                                 cudaDevAttrMaxRegistersPerBlock,
                                 device);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get Max Registers Per Block: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Max threads per block: " << maxThreadsPerBlock << std::endl;
    std::cout << "Max shared memory per block: " << maxSharedMemoryPerBlock << " bytes" << std::endl;
    std::cout << "Max registers per block: " << maxRegistersPerBlock << std::endl;

    return 0;
}
```