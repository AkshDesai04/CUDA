/*
 * Aim of the program:
 * Query the device property `kernelExecTimeoutEnabled` and print whether a kernel execution timeout is enabled.
 *
 * Thought process:
 * 1. The CUDA Runtime API provides `cudaGetDeviceProperties` which fills a `cudaDeviceProp` struct.
 * 2. The struct contains a field `kernelExecTimeoutEnabled` (int) that indicates whether the device enforces a kernel execution time limit.
 * 3. To use it, we need to include <cuda_runtime.h>, call `cudaGetDeviceProperties` for the current device (which we can obtain via `cudaGetDevice`).
 * 4. Handle any error codes returned by CUDA API calls, printing descriptive messages if an error occurs.
 * 5. After retrieving the property, print a human‑readable message: "Kernel execution timeout is enabled" or "Kernel execution timeout is not enabled".
 * 6. Since this is a simple console program, `main` returns 0 on success or a non‑zero exit code on failure.
 *
 * Additional considerations:
 * - The program should compile with `nvcc` and run on any CUDA capable device.
 * - We'll use `fprintf(stderr, ...)` for error messages and `printf` for the normal output.
 * - No device memory allocation or kernel launch is required; we only query device properties.
 */

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get current CUDA device: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (prop.kernelExecTimeoutEnabled)
    {
        printf("Kernel execution timeout is enabled.\n");
    }
    else
    {
        printf("Kernel execution timeout is not enabled.\n");
    }

    return 0;
}
