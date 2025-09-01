```cpp
/*
 * Aim of the program:
 * Use `cudaDeviceSynchronize` and proper error checking. An out-of-bounds access often results in `cudaErrorIllegalAddress`. Catch and report this error.
 *
 * My entire thinking:
 * I want to write a small CUDA C program that demonstrates how to detect an illegal memory access (out-of-bounds write) on the device. The key steps are:
 * 1. Allocate a small array on the GPU (e.g., 5 integers).
 * 2. Launch a kernel with more threads than the array can safely handle. Each thread will write to the array at an index based on its threadIdx.x.
 * 3. Because the kernel will write past the end of the array for threads with index >= 5, this should trigger a `cudaErrorIllegalAddress` when the device synchronizes.
 * 4. Use `cudaDeviceSynchronize()` to force the host to wait for the kernel and surface any asynchronous errors.
 * 5. Wrap CUDA API calls with an error-checking macro so that any failure (including the illegal address) is caught and reported with a readable message.
 * 6. After the kernel launch, check the return code of the kernel launch (if any), then call `cudaDeviceSynchronize()` and check its return code. If an error occurs, print it with `cudaGetErrorString`.
 * 7. The program will clean up resources before exiting.
 * 8. No extra files are needed; all code is contained in a single .cu file. The program is written in standard C++ (but compatible with C), uses `nvcc` for compilation, and follows best practices for error handling in CUDA.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

// Macro for checking CUDA errors following a kernel launch or API call
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " code=" << static_cast<int>(err)                     \
                      << " \"" << cudaGetErrorString(err) << "\"\n";           \
            exit(EXIT_FAILURE);                                               \
        }                                                                      \
    } while (0)

// Kernel that intentionally writes out-of-bounds
__global__ void outOfBoundsKernel(int *arr, int size) {
    int idx = threadIdx.x; // Each thread writes to its own index
    // Intentionally write to arr[idx] even if idx >= size
    arr[idx] = idx;
}

int main() {
    const int arraySize = 5;
    const int numThreads = 10; // More than arraySize to force out-of-bounds
    int *d_array = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_array, arraySize * sizeof(int)));

    // Launch kernel with more threads than arraySize
    outOfBoundsKernel<<<1, numThreads>>>(d_array, arraySize);

    // Check for kernel launch errors
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                  << cudaGetErrorString(launchErr) << std::endl;
        cudaFree(d_array);
        return EXIT_FAILURE;
    }

    // Synchronize to catch any runtime errors (like illegal address)
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        std::cerr << "Error after kernel execution: "
                  << cudaGetErrorString(syncErr) << std::endl;
        cudaFree(d_array);
        return EXIT_FAILURE;
    }

    // If we reach here, no errors were detected
    std::cout << "Kernel executed successfully (no illegal address detected)." << std::endl;

    // Clean up
    CUDA_CHECK(cudaFree(d_array));
    return EXIT_SUCCESS;
}
```