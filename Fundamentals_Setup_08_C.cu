/*
Aim of the program:
Write a program that compares the driver and runtime versions and prints a warning if the driver version is older than the runtime version, as this is an invalid configuration.

Thinking:
- The program needs to obtain the driver and runtime CUDA version numbers.
- The CUDA runtime API provides cudaRuntimeGetVersion() which returns an int like 11020 for CUDA 11.2.
- The CUDA driver API provides cudaDriverGetVersion() which similarly returns an int.
- Both functions return a cudaError_t indicating success or failure, so we should check for errors.
- After retrieving both versions, print them to the console.
- Compare the two integers: if driver_version < runtime_version, we print a warning using fprintf to stderr.
- Use standard C libraries (stdio.h). Include <cuda_runtime.h> for runtime functions and <cuda.h> for driver functions.
- Return 0 on success, or a non-zero error code if retrieving the versions fails.
- Compile with nvcc (e.g., nvcc -o check_cuda_versions check_cuda_versions.cu).
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

static void printVersion(int version)
{
    // The version is a 4-digit integer: e.g., 11020 -> 11.20
    int major = version / 1000;
    int minor = (version % 1000) / 10;
    int patch = version % 10;
    if (patch == 0)
        printf("%d.%d", major, minor);
    else
        printf("%d.%d.%d", major, minor, patch);
}

int main(void)
{
    int driver_version = 0;
    int runtime_version = 0;
    cudaError_t err;

    /* Get driver version */
    err = cudaDriverGetVersion(&driver_version);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get CUDA driver version: %s\n", cudaGetErrorString(err));
        return 1;
    }

    /* Get runtime version */
    err = cudaRuntimeGetVersion(&runtime_version);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get CUDA runtime version: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("CUDA Driver Version : ");
    printVersion(driver_version);
    printf("\n");

    printf("CUDA Runtime Version: ");
    printVersion(runtime_version);
    printf("\n");

    if (driver_version < runtime_version)
    {
        fprintf(stderr,
                "WARNING: CUDA driver version is older than runtime version. "
                "This is an invalid configuration.\n");
    }

    return 0;
}
