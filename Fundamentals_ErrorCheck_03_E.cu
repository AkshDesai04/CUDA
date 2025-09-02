/*
Aim of the program: Why is it good practice to check for errors both immediately after launch and at synchronization? (Catch config errors early, catch execution errors later).
Thinking:  
In CUDA programming, many errors are divided into two categories: configuration errors and execution errors.  
* Configuration errors are detected at the moment a kernel launch request is processed by the driver.  
  Examples include invalid grid/block dimensions, exceeding the maximum number of threads per block, or specifying too many blocks for the device.  
  These errors are caught immediately after the launch call by querying `cudaGetLastError()`.  
* Execution errors occur while the kernel is running on the device, such as accessing invalid memory, division by zero, or triggering an assertion.  
  These errors are not known until the device has finished executing the kernel, so they are usually caught by synchronizing the device (`cudaDeviceSynchronize()`) and then checking the last error again.  

By checking for errors immediately after launch we can catch configuration problems before the kernel starts executing, saving time and providing clearer diagnostics.  
By checking after synchronization we can detect problems that arise during kernel execution, ensuring that any runtime failures are not silently ignored.  
The following program demonstrates both checks by intentionally provoking a configuration error (too many threads per block) and an execution error (writing to a null device pointer).  It prints the appropriate error messages after each stage, illustrating why both checks are essential for robust CUDA code.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void configKernel()
{
    // No-op kernel; used only to trigger configuration error
}

__global__ void execKernel(int *ptr)
{
    // Intentionally write to a null pointer to cause an execution error
    int idx = threadIdx.x;
    ptr[idx] = idx;
}

int main()
{
    cudaError_t err;

    /* ----------------- 1. Configuration error test ----------------- */
    printf("Launching kernel with configuration error (excessive threads per block)...\n");
    configKernel<<<1, 1000000>>>();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error immediately after launch: %s\n", cudaGetErrorString(err));
    } else {
        printf("No error after launch (unexpected for config error).\n");
    }

    /* ----------------- 2. Execution error test ----------------- */
    // Intentionally set device pointer to null (unallocated memory)
    int *d_ptr = nullptr;

    printf("\nLaunching kernel that writes to an unallocated device pointer...\n");
    execKernel<<<1, 256>>>(d_ptr);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error immediately after launch: %s\n", cudaGetErrorString(err));
    } else {
        printf("No error after launch (expected; execution error will be caught later).\n");
    }

    /* Synchronize to catch execution error */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error after synchronization: %s\n", cudaGetErrorString(err));
    } else {
        printf("No error after synchronization (unexpected for execution error).\n");
    }

    return 0;
}
