```cpp
/*
Write a program that prints the CUDA driver version and the CUDA runtime version detected on your system. Use `cudaDriverGetVersion` and `cudaRuntimeGetVersion` and explain in a comment what the difference between the two is.

My thinking:
- The program is a simple C++ (or C) CUDA host code that calls the two CUDA Runtime API functions: cudaDriverGetVersion and cudaRuntimeGetVersion.
- We need to include <cuda_runtime.h> for runtime API and <cuda.h> for driver API. Both provide the functions.
- The program should print the integer versions (e.g., 11020 for CUDA 11.2). These integers encode major*1000 + minor*10 + patch.
- The comment explaining the difference: Driver API is lower level, deals with GPU device drivers; runtime API is higher level, abstracts driver for user code. Driver API can run on systems without a full CUDA installation; runtime needs the runtime library. Usually, the driver version is >= runtime version.
- We'll check the return status of each call and print an error if they fail.
- We'll use printf from <cstdio> or std::cout. For simplicity, use printf.
- We'll convert the integer version to human-readable form by printing the integer value.
- We'll compile with nvcc. The code is self-contained.
*/

#include <cstdio>
#include <cuda_runtime.h>
#include <cuda.h>

int main(void) {
    int driverVersion = 0;
    int runtimeVersion = 0;

    // Get CUDA driver version
    cudaError_t driverErr = cudaDriverGetVersion(&driverVersion);
    if (driverErr != cudaSuccess) {
        fprintf(stderr, "cudaDriverGetVersion failed: %s\n", cudaGetErrorString(driverErr));
        return 1;
    }

    // Get CUDA runtime version
    cudaError_t runtimeErr = cudaRuntimeGetVersion(&runtimeVersion);
    if (runtimeErr != cudaSuccess) {
        fprintf(stderr, "cudaRuntimeGetVersion failed: %s\n", cudaGetErrorString(runtimeErr));
        return 1;
    }

    printf("CUDA Driver Version: %d\n", driverVersion);
    printf("CUDA Runtime Version: %d\n", runtimeVersion);

    return 0;
}
```