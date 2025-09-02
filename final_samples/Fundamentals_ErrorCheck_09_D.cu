/*
Demonstrate that the exception is caught correctly by triggering an error (e.g., failed `cudaMalloc`).

Thought process:
1. The goal is to show that a runtime error (such as an out‑of‑memory situation) can be caught using C++ exception handling in a CUDA program.
2. Since the CUDA C API (`cudaMalloc`, `cudaMemcpy`, etc.) does not throw C++ exceptions by itself, we will manually check the return value of `cudaMalloc` and, on failure, throw a `std::runtime_error` containing the CUDA error string.
3. In `main()` we wrap the call to the memory allocation function in a `try`/`catch` block. When the exception is thrown, the `catch` block will intercept it and print a clear message, thereby demonstrating that the exception was caught correctly.
4. To guarantee a failure, we will request an astronomically large allocation (e.g., 2^62 bytes) which will almost certainly exceed the GPU's memory capacity. The program will also perform a small, successful allocation first to illustrate normal operation and ensure that the failure occurs only when the oversized allocation is attempted.
5. The code is written in a single `.cu` file, includes the necessary headers, and uses C++ features (exception handling, `std::runtime_error`, `std::cout`/`std::cerr`).
6. The program can be compiled with `nvcc` and run on a system with an NVIDIA GPU.
*/

#include <iostream>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

/* Helper function to convert cudaError_t to a human‑readable string */
std::string cudaErrorToString(cudaError_t err)
{
    return std::string(cudaGetErrorString(err));
}

/* Attempt to allocate a large amount of device memory.
   On failure, throw a std::runtime_error containing the CUDA error message. */
void allocateHugeMemory()
{
    const size_t hugeSize = static_cast<size_t>(1) << 62; // 4 EiB, far beyond typical GPU memory
    void* d_ptr = nullptr;

    cudaError_t err = cudaMalloc(&d_ptr, hugeSize);
    if (err != cudaSuccess)
    {
        // Free any partially allocated memory (cudaMalloc may return a valid pointer on failure)
        if (d_ptr != nullptr)
        {
            cudaFree(d_ptr);
        }
        throw std::runtime_error("cudaMalloc failed: " + cudaErrorToString(err));
    }

    // If allocation succeeded (unlikely), free it immediately
    cudaFree(d_ptr);
}

/* Allocate a small block of memory that should succeed.
   This demonstrates normal operation and ensures that the failure
   we trigger later is intentional. */
void allocateSmallMemory()
{
    const size_t smallSize = 256 * 1024; // 256 KB
    void* d_ptr = nullptr;

    cudaError_t err = cudaMalloc(&d_ptr, smallSize);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Small cudaMalloc failed: " + cudaErrorToString(err));
    }

    cudaFree(d_ptr);
}

int main()
{
    std::cout << "Starting CUDA memory allocation test.\n";

    try
    {
        // First perform a normal allocation to confirm everything is working
        allocateSmallMemory();
        std::cout << "Small memory allocation succeeded.\n";

        // Now trigger a deliberate failure
        std::cout << "Attempting huge memory allocation (should fail)...\n";
        allocateHugeMemory();

        // If we get here, something unexpected happened
        std::cerr << "Unexpectedly succeeded in huge allocation!\n";
        return EXIT_FAILURE;
    }
    catch (const std::exception& e)
    {
        // Catch the exception thrown by allocateHugeMemory and print the message
        std::cerr << "Caught exception: " << e.what() << '\n';
        std::cerr << "Exception handling worked correctly.\n";
        return EXIT_SUCCESS;
    }

    // This line should never be reached
    return EXIT_FAILURE;
}
