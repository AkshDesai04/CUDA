```cpp
/*
Aim: Experiment with kernel launch failure by launching a kernel with more threads per block
than the device maximum (e.g., 2048). Use proper error checking after the launch to catch
and report the cudaErrorInvalidConfiguration error.

Thought process:
1. Determine the device's maximum threads per block using cudaGetDeviceProperties.
2. Define a trivial kernel that does nothing.
3. Intentionally set blockDim.x to a value larger than the reported maximum (here 2048).
4. Launch the kernel and immediately call cudaGetLastError() to capture any launch
   configuration errors. The expected error is cudaErrorInvalidConfiguration.
5. Print the error string to confirm that the failure was detected.
6. For completeness, also call cudaDeviceSynchronize and check its return value, although
   the launch error is usually caught before synchronization.
7. Ensure the program compiles as a standalone .cu file and uses only standard CUDA
   APIs and C++ I/O for reporting.
*/

#include <iostream>
#include <cuda_runtime.h>

// A trivial kernel that does nothing.
__global__ void dummyKernel() {
    // Intentionally empty.
}

int main() {
    // Retrieve device properties to know the maximum threads per block.
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Device 0 maxThreadsPerBlock: " << prop.maxThreadsPerBlock << std::endl;

    // Intentionally set block size greater than maxThreadsPerBlock to trigger failure.
    const int oversizedThreadsPerBlock = 2048; // Example exceeding typical max (1024)
    dim3 blockSize(oversizedThreadsPerBlock);
    dim3 gridSize(1);

    // Launch the kernel.
    dummyKernel<<<gridSize, blockSize>>>();

    // Check for kernel launch errors.
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Kernel launch failed with error: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "Kernel launched successfully (unexpected)." << std::endl;
    }

    // Optional: synchronize to see if any runtime error occurs.
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "cudaDeviceSynchronize returned error: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "cudaDeviceSynchronize completed successfully." << std::endl;
    }

    return 0;
}
```