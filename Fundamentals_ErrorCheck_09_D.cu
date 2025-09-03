/*
Aim of the program: Demonstrate that the exception is caught correctly by triggering an error (e.g., failed cudaMalloc).

Thinking process:
1. CUDA C++ does not have built‑in exception handling for CUDA API calls, but we can wrap the calls in a helper that throws a std::runtime_error when an error occurs.
2. The program will attempt to allocate an impractically large amount of GPU memory (e.g., size_t(-1)), which should trigger an error such as cudaErrorMemoryAllocation.
3. The allocation call is wrapped in a try/catch block. When the exception is thrown, we catch it and print a descriptive message, thus proving that the exception mechanism works.
4. The code is written in a single .cu file, uses standard CUDA headers, and is fully self‑contained.
5. No external dependencies or extra files are required; the program compiles with nvcc and runs on any CUDA‑capable device.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

// Helper function to check CUDA API return codes and throw an exception if an error occurs.
inline void checkCuda(cudaError_t err, const char* operation)
{
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(err));
    }
}

int main()
{
    std::cout << "Attempting to allocate an excessively large amount of GPU memory to trigger an error.\n";

    try
    {
        // Intentionally request an impossible amount of memory (e.g., maximum value of size_t).
        size_t largeSize = static_cast<size_t>(-1); // maximum possible size
        void* d_ptr = nullptr;

        // This cudaMalloc call should fail on any realistic GPU.
        checkCuda(cudaMalloc(&d_ptr, largeSize), "cudaMalloc");

        // If the allocation unexpectedly succeeds, free the memory.
        cudaFree(d_ptr);
        std::cout << "Unexpectedly succeeded in allocating large memory.\n";
    }
    catch (const std::exception& e)
    {
        // The exception is caught here, demonstrating correct exception handling.
        std::cout << "Caught exception as expected: " << e.what() << '\n';
    }

    std::cout << "Program completed.\n";
    return 0;
}
