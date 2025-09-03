```cpp
/*
Write a program that queries and prints the total amount of global memory available on device 0 in megabytes (MB). You will need to get the `totalGlobalMem` property from `cudaDeviceProp` and convert it from bytes to MB.

Thinking:
- The program is a minimal CUDA C++ program that queries the properties of the first CUDA device (device 0).
- We need to include <cuda_runtime.h> and <stdio.h> for CUDA runtime API and standard I/O.
- Use `cudaGetDeviceProperties(&prop, 0)` to obtain a `cudaDeviceProp` structure.
- The `totalGlobalMem` field contains the device's global memory size in bytes. Convert this to megabytes by dividing by 1024*1024.
- Print the result in a human-readable form (e.g., "Device 0 global memory: X.XX MB").
- Include basic error checking after the CUDA API call to handle cases where the device does not exist or another error occurs.
- The program will compile with nvcc and run on a machine with an appropriate CUDA installation.
- No additional CUDA kernels are needed; this is purely host code.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                    \
    do {                                                    \
        cudaError_t err = call;                             \
        if (err != cudaSuccess) {                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",    \
                    __FILE__, __LINE__,                     \
                    cudaGetErrorString(err));                \
            return EXIT_FAILURE;                            \
        }                                                   \
    } while (0)

int main(void) {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    double memMB = (double)prop.totalGlobalMem / (1024.0 * 1024.0);

    printf("Device 0 global memory: %.2f MB\n", memMB);

    return EXIT_SUCCESS;
}
```