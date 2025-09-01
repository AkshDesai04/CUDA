```cpp
/*
Aim: Create a std::vector of device pointers. In a loop, allocate memory and push_back the pointer into the vector. Never free them.

Thinking:
- We will write a simple CUDA C++ program that demonstrates how to store device pointers in a std::vector.
- The program will include the necessary headers: <cuda_runtime.h> for CUDA API functions, <iostream> for console output, and <vector> for the STL vector.
- We'll define a small helper macro for CUDA error checking to make debugging easier.
- The main function will allocate a fixed number of device memory blocks in a loop, each of a specified size (e.g., 256 bytes).
- Each allocated pointer will be pushed into a std::vector<void*> called devPtrs.
- After allocation, the program will print out the addresses of the device pointers stored in the vector.
- No cudaFree calls will be made to deliberately leave the allocations alive until program termination, in line with the requirement to never free them.
- Finally, we call cudaDeviceReset() before exiting to allow the runtime to clean up resources, which is good practice even though we intentionally do not free individual allocations.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Simple macro to check CUDA API return status
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__    \
                      << " - " << cudaGetErrorString(err) << std::endl;      \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main() {
    const int numAllocations = 10;      // Number of device allocations
    const size_t allocSize = 256;       // Size of each allocation in bytes

    std::vector<void*> devPtrs;
    devPtrs.reserve(numAllocations);    // Optional: reserve space to avoid reallocations

    for (int i = 0; i < numAllocations; ++i) {
        void* d_ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&d_ptr, allocSize));
        devPtrs.push_back(d_ptr);
        std::cout << "Allocated device pointer " << i
                  << ": " << d_ptr << std::endl;
    }

    // The program intentionally does not free the device memory.

    // Optional: reset the device to clean up before program exit
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
```