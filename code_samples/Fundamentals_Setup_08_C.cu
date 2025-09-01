/*
Aim: Write a program that compares the driver and runtime versions and prints a warning if the driver version is older than the runtime version, as this is an invalid configuration.
Thinking:
1. Use the CUDA Runtime API functions `cudaRuntimeGetVersion()` and `cudaDriverGetVersion()` to obtain the runtime and driver version numbers, respectively. These functions return the version as an integer of the form `MAJOR * 1000 + MINOR * 10 + PATCH`. For example, CUDA 11.2.0 returns `11020`.
2. The functions return a `cudaError_t` which we should check to ensure the call succeeded. If an error occurs, we should print an error message and exit.
3. Compare the two integer version values. If the driver version is strictly less than the runtime version, that indicates that the installed driver is older than the runtime library, which is an invalid configuration. In this case, we print a warning message indicating the mismatch.
4. If the driver version is greater than or equal to the runtime version, we can print a confirmation that the configuration is valid.
5. The program is written in C/C++ and compiled with `nvcc`. It includes necessary headers (`<cuda_runtime.h>` and `<cstdio>`) and uses `printf` for output. The main function returns an `int` status code.

The code below follows this plan.
*/

#include <cuda_runtime.h>
#include <cstdio>

int main() {
    int runtimeVersion = 0;
    int driverVersion = 0;

    // Get runtime version
    cudaError_t rtErr = cudaRuntimeGetVersion(&runtimeVersion);
    if (rtErr != cudaSuccess) {
        fprintf(stderr, "Failed to get CUDA runtime version: %s\n", cudaGetErrorString(rtErr));
        return 1;
    }

    // Get driver version
    cudaError_t drvErr = cudaDriverGetVersion(&driverVersion);
    if (drvErr != cudaSuccess) {
        fprintf(stderr, "Failed to get CUDA driver version: %s\n", cudaGetErrorString(drvErr));
        return 1;
    }

    // Print obtained versions
    printf("CUDA Runtime Version: %d\n", runtimeVersion);
    printf("CUDA Driver Version : %d\n", driverVersion);

    // Compare versions
    if (driverVersion < runtimeVersion) {
        fprintf(stderr,
                "WARNING: CUDA driver version (%d) is older than runtime version (%d). "
                "This is an invalid configuration and may lead to runtime errors.\n",
                driverVersion, runtimeVersion);
        return 1;
    } else {
        printf("Driver version is compatible with runtime.\n");
    }

    return 0;
}
