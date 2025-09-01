```cpp
/*
Create a `std::vector` of device pointers. In a loop, allocate memory and `push_back` the pointer into the vector. Never free them.

The program demonstrates how to maintain a collection of device pointers using a C++ `std::vector`. It allocates a fixed number of device memory blocks in a loop, stores each returned pointer in the vector, and prints the host-visible pointer addresses. The code intentionally does not free the device memory, illustrating a memory leak scenario which may be useful for testing GPU memory usage or for educational purposes. In a production setting, the device memory would need to be freed, but for this simple demonstration we omit that step as per the requirement.
*/

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

int main()
{
    const int numAllocations = 5;   // Number of device allocations
    const size_t allocationSize = 256 * sizeof(float); // Size of each allocation

    std::vector<float*> devPtrs; // Vector to hold device pointers

    for (int i = 0; i < numAllocations; ++i) {
        float* d_ptr = nullptr;
        cudaError_t err = cudaMalloc(&d_ptr, allocationSize);
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed at iteration " << i
                      << " with error: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        devPtrs.push_back(d_ptr);
        std::cout << "Allocated device memory " << i
                  << " at address: " << static_cast<void*>(d_ptr) << std::endl;
    }

    // Intentionally not freeing the allocated device memory

    std::cout << "Number of device pointers stored: " << devPtrs.size() << std::endl;

    // Wait for all GPU work to finish before exiting
    cudaDeviceSynchronize();

    return 0;
}
```