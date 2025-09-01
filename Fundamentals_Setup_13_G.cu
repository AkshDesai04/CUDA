```cpp
/* 
   Query and print whether a kernel execution timeout is enabled on the device. 
   ---------------------------------------------------------------------------
   Aim: The program should determine for each CUDA-capable device whether the 
   kernel execution timeout feature is enabled. This feature, typically present 
   on Windows with integrated GPUs, limits the maximum execution time of a 
   kernel to prevent the GPU from hanging the OS. The CUDA runtime provides 
   a device attribute (cudaDevAttrKernelExecTimeout) that indicates this 
   setting. We will query this attribute for all devices present on the system 
   and print the result.

   Thinking process:
   1. Include necessary headers: <cuda_runtime.h> for CUDA APIs and 
      <iostream> for console output.
   2. Define a helper macro or function to check CUDA error codes and print
      meaningful messages.
   3. In main():
      a. Get the number of CUDA devices via cudaGetDeviceCount().
      b. For each device index:
         - Retrieve the kernel execution timeout attribute using 
           cudaDeviceGetAttribute().
         - Interpret the returned integer: 1 means enabled, 0 means disabled.
         - Print the device index and the status in a human-readable form.
      c. Handle any errors from API calls gracefully.
   4. Compile with nvcc and run; the output should be something like:
         Device 0: Kernel Execution Timeout Enabled: Yes
      or
         Device 0: Kernel Execution Timeout Enabled: No
*/

#include <cuda_runtime.h>
#include <iostream>

// Helper macro to check CUDA API return codes
#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA error at " << __FILE__         \
                      << ":" << __LINE__ << " - "            \
                      << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE);                         \
        }                                                     \
    } while (0)

int main() {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found." << std::endl;
        return EXIT_SUCCESS;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        int timeoutEnabled = 0;
        // cudaDevAttrKernelExecTimeout indicates if a kernel exec timeout
        // is enabled (1) or disabled (0) for the device.
        CHECK_CUDA(cudaDeviceGetAttribute(&timeoutEnabled,
                                          cudaDevAttrKernelExecTimeout,
                                          dev));

        std::cout << "Device " << dev << ": Kernel Execution Timeout Enabled: "
                  << (timeoutEnabled ? "Yes" : "No") << std::endl;
    }

    return EXIT_SUCCESS;
}
```