```cpp
/*
Aim of the program: Write a program that queries the available global memory and prints a warning if it is less than a certain threshold (e.g., 4096 MB).

Thinking process:
1. We need a CUDA application that runs on the GPU and uses the CUDA Runtime API.
2. The program should query device properties to know total memory, and also query free memory.
3. Use `cudaMemGetInfo` to get the amount of free and total global memory.
4. Compare the free memory against a threshold value (4096 MB). If free memory is less, print a warning.
5. Add basic error checking after each CUDA API call for robustness.
6. Use standard C++ I/O (`<iostream>`) to print information.
7. The code should be a single .cu file that can be compiled with `nvcc`.
8. The top of the file will contain a multiline comment with the aim and this reasoning, as requested.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define CHECK_CUDA(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__  \
                      << " - " << cudaGetErrorString(err) << std::endl;   \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

int main() {
    int device = 0;  // Default to device 0
    CHECK_CUDA(cudaSetDevice(device));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    std::cout << "Device [" << device << "]: " << prop.name << std::endl;

    size_t freeMem, totalMem;
    CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));

    std::cout << "Total global memory: " << (totalMem / (1024ULL * 1024ULL)) << " MB" << std::endl;
    std::cout << "Free global memory: " << (freeMem / (1024ULL * 1024ULL)) << " MB" << std::endl;

    const size_t thresholdMB = 4096;  // 4 GB threshold
    const size_t thresholdBytes = thresholdMB * 1024ULL * 1024ULL;

    if (freeMem < thresholdBytes) {
        std::cerr << "WARNING: Available global memory (" << (freeMem / (1024ULL * 1024ULL))
                  << " MB) is less than the threshold (" << thresholdMB << " MB)." << std::endl;
    } else {
        std::cout << "Available global memory meets the threshold." << std::endl;
    }

    return 0;
}
```