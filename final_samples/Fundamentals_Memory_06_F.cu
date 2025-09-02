```cuda
/* 
Aim: Create a `std::vector` of device pointers. In a loop, allocate memory and `push_back` the pointer into the vector. Never free them.

Thinking:
The goal is to demonstrate the creation of a C++ `std::vector` that holds pointers to memory allocated on the CUDA device.
- Include the necessary headers: `<vector>` for the container, `<iostream>` for output, and `<cuda_runtime.h>` for CUDA API calls.
- Define the number of allocations (`N`) and the size of each allocation (`sizePerAllocation`).
- Create a `std::vector<void*>` to store the device pointers.
- In a loop from 0 to `N-1`:
    - Call `cudaMalloc` to allocate a chunk of device memory.
    - Check for errors after each allocation.
    - Push the returned device pointer into the vector.
- After the loop, print the addresses stored in the vector to confirm allocation.
- Do NOT call `cudaFree` on any of the pointers, as per the requirement to never free them.
- The program uses the host side only; no device kernels are launched.
- Compile with `nvcc -std=c++11 -o vector_of_device_ptrs vector_of_device_ptrs.cu`.

*/

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

int main()
{
    const int N = 5;                     // Number of device allocations
    const size_t sizePerAllocation = 256; // Bytes per allocation

    // Vector to hold device pointers
    std::vector<void*> devPtrVec;

    // Allocate device memory in a loop and push pointers into the vector
    for (int i = 0; i < N; ++i)
    {
        void* d_ptr = nullptr;
        cudaError_t err = cudaMalloc(&d_ptr, sizePerAllocation);
        if (err != cudaSuccess)
        {
            std::cerr << "cudaMalloc failed at iteration " << i
                      << " with error: " << cudaGetErrorString(err) << std::endl;
            return EXIT_FAILURE;
        }

        devPtrVec.push_back(d_ptr);
        std::cout << "Allocated device memory at address: " << d_ptr
                  << " (size: " << sizePerAllocation << " bytes)" << std::endl;
    }

    // Note: Device memory is intentionally not freed to follow the instruction.
    // The program ends here; device memory will be reclaimed when the process exits.

    return EXIT_SUCCESS;
}
```