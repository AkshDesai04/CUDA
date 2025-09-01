```cpp
/* 
Aim of the program (word for word):
"Experiment with kernel launch failure. Try to launch a kernel with more threads per block than the device maximum (e.g., 2048). Use proper error checking after the launch to catch and report the `cudaErrorInvalidConfiguration` error."

My thoughts while writing this program:

1. The user wants to see a kernel launch failure due to exceeding the maximum number of threads per block.  
2. The maximum threads per block is device-dependent, so we query it with `cudaGetDeviceProperties`.  
3. We'll attempt to launch a kernel with 2048 threads per block, which should exceed the limit on almost all current GPUs (typically 1024).  
4. After the kernel launch we must check for errors using `cudaGetLastError()` and report the specific `cudaErrorInvalidConfiguration` error.  
5. A small, no-op kernel is sufficient for this experiment; it simply runs on the GPU and does nothing.  
6. The code will print the device name and max threads per block for clarity, then attempt the launch, and finally print the error status.  
7. We will include a small helper macro `CHECK_CUDA` for error checking after API calls.  
8. All includes and main are standard CUDA C++.

The program will be compiled with `nvcc`. It demonstrates that launching a kernel with too many threads per block triggers an immediate launch failure and can be caught with proper error handling. 
*/

#include <cuda_runtime.h>
#include <iostream>
#include <string>

// Helper macro for CUDA runtime error checking
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                      << " - " << cudaGetErrorString(err) << std::endl;     \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// A trivial kernel that does nothing
__global__ void dummyKernel() {
    // No operation; just to occupy the thread
}

int main() {
    int device_id = 0;
    CHECK_CUDA(cudaSetDevice(device_id));

    // Query device properties
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));

    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Maximum threads per block: " << prop.maxThreadsPerBlock << std::endl;

    // Define an attempt to launch with 2048 threads per block
    const int threads_per_block = 2048;
    const int blocks = 1; // just one block for simplicity

    std::cout << "Attempting to launch kernel with " << threads_per_block
              << " threads per block (" << blocks << " blocks)." << std::endl;

    // Launch the kernel
    dummyKernel<<<blocks, threads_per_block>>>();

    // Check for launch errors
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(launch_err)
                  << " (error code " << launch_err << ")" << std::endl;
    } else {
        // Wait for GPU to finish
        CHECK_CUDA(cudaDeviceSynchronize());
        std::cout << "Kernel launched successfully." << std::endl;
    }

    return 0;
}
```