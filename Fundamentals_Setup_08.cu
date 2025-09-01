/*
Aim: Write a program that prints the CUDA driver version and the CUDA runtime version detected on your system. 
Use `cudaDriverGetVersion` and `cudaRuntimeGetVersion` and explain in a comment what the difference between the two is.

Thinking:
1. The program should be a single .cu file that can be compiled with `nvcc`.
2. We need to include both the driver API header <cuda.h> and the runtime API header <cuda_runtime.h>.
3. Call `cudaDriverGetVersion(&driverVersion)` and check the returned cudaError_t for success.
4. Call `cudaRuntimeGetVersion(&runtimeVersion)` similarly.
5. Print both versions in a human‑readable form. The version numbers are returned as an integer of the form major*1000 + minor*10 + patch, so we convert to major.minor.patch format.
6. Provide a helpful comment explaining that the driver API refers to the installed NVIDIA driver (the low‑level API that talks directly to the GPU hardware), while the runtime API is a higher‑level abstraction that is linked with the application and may be newer than the driver. The runtime API is built on top of the driver API, so the driver version can be older or newer than the runtime version.
7. Include error checking and exit with non‑zero status on failure.
8. The program will be self‑contained and only output the two version strings.

Difference between the two:
- `cudaDriverGetVersion` queries the version of the NVIDIA CUDA driver installed on the system. This is the low‑level component that interacts directly with the GPU hardware.
- `cudaRuntimeGetVersion` queries the version of the CUDA Runtime API that the program is linked against. The runtime API sits on top of the driver API and may be newer or older than the driver; it provides a simplified interface for kernel launches, memory management, etc. The runtime is essentially a thin wrapper around the driver API.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

/* Helper function to convert CUDA version integer to human‑readable string */
static void printCudaVersion(const char *label, int version)
{
    int major = version / 1000;
    int minor = (version % 1000) / 10;
    int patch = version % 10;
    printf("%s: %d.%d.%d\n", label, major, minor, patch);
}

int main(void)
{
    int driverVersion = 0;
    cudaError_t driverStatus = cudaDriverGetVersion(&driverVersion);
    if (driverStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDriverGetVersion failed: %s\n", cudaGetErrorString(driverStatus));
        return 1;
    }

    int runtimeVersion = 0;
    cudaError_t runtimeStatus = cudaRuntimeGetVersion(&runtimeVersion);
    if (runtimeStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaRuntimeGetVersion failed: %s\n", cudaGetErrorString(runtimeStatus));
        return 1;
    }

    printf("CUDA Driver Version:\n");
    printCudaVersion("Driver", driverVersion);

    printf("\nCUDA Runtime Version:\n");
    printCudaVersion("Runtime", runtimeVersion);

    return 0;
}
