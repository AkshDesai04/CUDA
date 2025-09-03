/*
On Linux, the OS may have a watchdog timer that kills the kernel after a few seconds, causing `cudaDeviceSynchronize` to return an error (`cudaErrorLaunchTimeout`). Try to catch and report this error.

Thinking:
The goal is to demonstrate how to detect and report the cudaErrorLaunchTimeout that can occur when the Linux watchdog kills a long‑running CUDA kernel. We will create a deliberately slow (or infinite) kernel that will trigger the watchdog. After launching the kernel we call cudaDeviceSynchronize() to wait for completion and capture any errors. If the error code is cudaErrorLaunchTimeout we print a specific message; otherwise we print the generic CUDA error string. The program is simple, portable, and uses only standard CUDA runtime API calls. It should compile with nvcc and run on any device that supports the watchdog mechanism.*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Kernel that never terminates (or takes an extremely long time) */
__global__ void infiniteKernel(void)
{
    // A tight loop that never exits to trigger the watchdog timer.
    while (true)
    {
        // No-op, just keep spinning.
    }
}

int main(void)
{
    cudaError_t err;

    /* Reset the device to ensure a clean start */
    err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        printf("cudaDeviceReset failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    /* Launch the kernel that will time out */
    infiniteKernel<<<1, 1>>>();

    /* Synchronize and check for errors */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        if (err == cudaErrorLaunchTimeout)
        {
            printf("Caught cudaErrorLaunchTimeout: kernel launch timed out due to watchdog.\n");
        }
        else
        {
            printf("Kernel error: %s\n", cudaGetErrorString(err));
        }
    }
    else
    {
        printf("Kernel finished successfully (this should not happen with infiniteKernel).\n");
    }

    return 0;
}
