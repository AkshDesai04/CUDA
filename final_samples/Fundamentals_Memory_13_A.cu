/*
Use `cudaMemGetInfo` to find out how much free memory is available, then try to allocate slightly more than that amount to trigger the error.

Thought process:
1. Include necessary headers: <cuda_runtime.h>, <cstdio>, <cstdlib>, <iostream>.
2. In main, call cudaMemGetInfo(&freeMem, &totalMem). Check the returned cudaError_t; if it fails, print the error and exit.
3. Print freeMem and totalMem values to give context.
4. Compute an allocation size that is slightly larger than freeMem (e.g., freeMem + 1MB). Ensure the addition does not overflow size_t.
5. Call cudaMalloc with the computed size. Capture the error code.
6. If cudaMalloc returns cudaSuccess, free the allocated memory with cudaFree; otherwise, print the error code and human‑readable message using cudaGetErrorString.
7. Exit with appropriate status.

This program demonstrates how to query available GPU memory and intentionally provoke an allocation failure, which is useful for testing error handling paths in CUDA applications.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>

int main() {
    size_t freeMem, totalMem;
    cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemGetInfo failed: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "GPU total memory: " << totalMem << " bytes" << std::endl;
    std::cout << "GPU free memory : " << freeMem << " bytes" << std::endl;

    // Allocate slightly more than free memory (add 1 MB)
    const size_t oneMB = 1 * 1024 * 1024;
    size_t allocSize = freeMem + oneMB;

    // Protect against overflow (unlikely but good practice)
    if (allocSize < freeMem) {
        std::cerr << "Integer overflow when computing allocation size." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Attempting to allocate " << allocSize << " bytes ("
              << (double)allocSize / (1024 * 1024) << " MB)" << std::endl;

    void* devPtr = nullptr;
    err = cudaMalloc(&devPtr, allocSize);

    if (err == cudaSuccess) {
        std::cout << "Allocation succeeded unexpectedly. Freeing memory." << std::endl;
        cudaFree(devPtr);
        return EXIT_SUCCESS;
    } else {
        std::cerr << "cudaMalloc failed as expected: " << cudaGetErrorString(err)
                  << " (error code " << err << ")" << std::endl;
        return EXIT_FAILURE;
    }
}
