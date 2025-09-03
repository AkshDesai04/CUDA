```cpp
/* 
Create a std::vector of device pointers. In a loop, allocate memory and push_back the pointer into the vector. Never free them.

Thought process:
- The program needs to compile with nvcc as a .cu file.
- We will use the standard C++ std::vector to store device memory pointers.
- Each pointer will be obtained with cudaMalloc, which returns a void*.
- The loop will allocate a fixed amount of device memory (e.g., 256 bytes) for each iteration.
- We will push each allocated pointer into the vector.
- As instructed, we will not free any of the allocated memory before program exit.
- Error checking for cudaMalloc will be performed; if allocation fails, the program will print an error and exit.
- The program will also print the addresses of the allocated pointers for demonstration purposes.
- Since the memory is not freed, the program will intentionally leak device memory, matching the requirement.

Key decisions:
- Use `std::vector<void*>` because device pointers are of type void*.
- The loop count and allocation size are hard-coded for simplicity, but could be made configurable.
- The program prints a simple message for each allocation to confirm success.
- No kernel launches or other CUDA operations are performed; only memory allocation.

Potential pitfalls:
- Not freeing memory is intentional but would normally lead to a leak.
- The program should handle cudaMalloc failure gracefully.
- The vector is host-side; it holds device addresses, so no issues with device-side vector.

The final code below follows these guidelines and can be compiled with `nvcc -o program program.cu`.
*/

#include <vector>
#include <iostream>
#include <cuda_runtime.h>

int main() {
    const int numAllocations = 10;     // Number of device pointers to allocate
    const size_t allocationSize = 256; // Size of each allocation in bytes

    std::vector<void*> devicePtrs;
    devicePtrs.reserve(numAllocations); // Reserve space to avoid reallocations

    for (int i = 0; i < numAllocations; ++i) {
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, allocationSize);
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed at iteration " << i
                      << " with error: " << cudaGetErrorString(err) << std::endl;
            return EXIT_FAILURE;
        }
        devicePtrs.push_back(ptr);
        std::cout << "Allocated " << allocationSize << " bytes at device address "
                  << ptr << " (index " << i << ")" << std::endl;
    }

    // The program intentionally does not free the allocated device memory.
    std::cout << "All allocations completed. Exiting without freeing memory." << std::endl;
    return EXIT_SUCCESS;
}
```