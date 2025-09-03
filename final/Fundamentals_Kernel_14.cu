```cpp
/*
Experiment with kernel launch failure. Try to launch a kernel with more threads per block than the device maximum (e.g., 2048). Use proper error checking after the launch to catch and report the cudaErrorInvalidConfiguration error.

Thinking:

The goal of this program is to intentionally trigger a kernel launch failure by requesting more threads per block than the GPU supports. Most CUDA-capable GPUs have a limit of 1024 threads per block, but by launching 2048 threads we guarantee that the configuration is invalid. The kernel itself does nothing interesting; it just computes an index, which is sufficient to be a valid CUDA kernel.

To detect the failure, we perform the following steps:

1. Define a trivial kernel `dummyKernel` that is safe to launch.
2. In `main`, specify `threadsPerBlock = 2048` and `blocksPerGrid = 1`. This configuration exceeds the maximum threads per block for almost all devices.
3. Launch the kernel with the invalid configuration: `dummyKernel<<<blocksPerGrid, threadsPerBlock>>>()`.
4. Immediately after the launch, call `cudaGetLastError()` to capture any error that occurred during the launch phase. For a configuration error, this will return `cudaErrorInvalidConfiguration`.
5. Call `cudaDeviceSynchronize()` to capture any errors that occur during kernel execution, although in this case the launch error should already have been caught.
6. Convert the error codes to human-readable strings with `cudaGetErrorString()` and print them to the console.

This program demonstrates how to handle and report kernel launch errors gracefully, which is essential for robust CUDA applications. It also serves as an educational example of the importance of respecting hardware limits when configuring kernels.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dummyKernel()
{
    // Simple kernel that does nothing significant
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Prevent compiler optimization by using the index in a dummy operation
    if (idx == 0) { /* do nothing */ }
}

int main()
{
    // Intentionally use more threads per block than most GPUs support
    const int threadsPerBlock = 2048; // Exceeds typical max (1024)
    const int blocksPerGrid   = 1;

    printf("Launching kernel with %d threads per block on %d block(s).\n",
           threadsPerBlock, blocksPerGrid);

    // Launch kernel
    dummyKernel<<<blocksPerGrid, threadsPerBlock>>>();

    // Check for launch errors
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess)
    {
        printf("Kernel launch failed with error: %s (%d)\n",
               cudaGetErrorString(launchErr), launchErr);
    }
    else
    {
        printf("Kernel launched successfully (unexpected).\n");
    }

    // Synchronize to catch any execution errors
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess)
    {
        printf("Kernel execution failed with error: %s (%d)\n",
               cudaGetErrorString(syncErr), syncErr);
    }
    else
    {
        printf("Kernel executed successfully (unexpected).\n");
    }

    // Reset the device before exiting
    cudaDeviceReset();

    return 0;
}
```