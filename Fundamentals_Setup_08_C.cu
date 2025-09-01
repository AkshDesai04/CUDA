```cuda
/*
Aim: Write a program that compares the driver and runtime versions and prints a warning if the driver version is older than the runtime version, as this is an invalid configuration.

Thinking:
- The CUDA driver API provides cudaDriverGetVersion which returns the version of the installed driver.
- The CUDA runtime API provides cudaRuntimeGetVersion which returns the version of the CUDA runtime linked with the program.
- Both functions return an int with the major*1000 + minor*10 + patch format (e.g., 11020 for CUDA 11.2.0).
- To compare, simply retrieve both versions and compare the integers.
- If driver < runtime, print a warning message indicating the mismatch and that the configuration is invalid.
- We'll also print the versions for debugging purposes.
- Error handling: check return status of the API calls and use cudaGetErrorString for readable error messages.
- The program is a simple console application; no CUDA kernels are launched.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

int main(void) {
    int driver_version = 0;
    int runtime_version = 0;
    cudaError_t err;

    // Get driver version
    err = cudaDriverGetVersion(&driver_version);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error retrieving CUDA driver version: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Get runtime version
    err = cudaRuntimeGetVersion(&runtime_version);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error retrieving CUDA runtime version: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("CUDA Driver Version : %d\n", driver_version);
    printf("CUDA Runtime Version: %d\n", runtime_version);

    if (driver_version < runtime_version) {
        fprintf(stderr, "WARNING: CUDA driver version (%d) is older than the runtime version (%d).\n"
                        "This is an invalid configuration and may lead to undefined behavior.\n",
                driver_version, runtime_version);
    }

    return 0;
}
```