```c
/* 
Aim of the program:
Write a program that allocates a large amount of memory on the device (e.g., 512MB) and then immediately frees it using `cudaFree`. Use robust error checking for all API calls to ensure the allocation and deallocation were successful.

My thinking:
1. Include the CUDA runtime header and iostream for console output.
2. Define a macro `CHECK_CUDA` that wraps CUDA runtime API calls and prints an error message with the function name, error string, file, line, and exits if the call fails.
3. In `main`, specify the desired allocation size in bytes: 512 * 1024 * 1024.
4. Declare a device pointer of type `int*` (or `char*`) and call `cudaMalloc` with the size. Use the macro for error checking.
5. Optionally, print a success message indicating the allocation succeeded.
6. Immediately call `cudaFree` on the pointer, again using the macro for error checking.
7. Print a success message indicating the deallocation succeeded.
8. Return 0 to indicate success.
*/

#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error in function '" << #call << "' at " \
                      << __FILE__ << ":" << __LINE__ << " : "           \
                      << cudaGetErrorString(err) << std::endl;          \
            std::exit(EXIT_FAILURE);                                    \
        }                                                               \
    } while (0)

int main()
{
    const size_t allocationSize = 512ULL * 1024 * 1024; // 512 MB
    int* devPtr = nullptr;

    std::cout << "Attempting to allocate " << allocationSize
              << " bytes on the device." << std::endl;

    // Allocate memory on the device
    CHECK_CUDA(cudaMalloc(&devPtr, allocationSize));

    std::cout << "Allocation succeeded. Pointer: " << devPtr << std::endl;

    // Free the allocated memory immediately
    CHECK_CUDA(cudaFree(devPtr));

    std::cout << "Memory successfully freed." << std::endl;

    return 0;
}
```