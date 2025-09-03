```cpp
/*
 * Aim of the program:
 * Try to pass the address of a host variable (`int x; int* p = &x;`) to a kernel.
 * Dereference it. Observe the error.
 *
 * Thinking:
 * In CUDA, when a kernel is launched, all arguments are copied from host to
 * device memory before the kernel runs. If we pass a host pointer (an address
 * in host virtual memory) to a kernel, the kernel receives that address as a
 * device pointer. On the device, that address does not refer to valid
 * memory, so dereferencing it results in an invalid memory access.
 *
 * The program below demonstrates this by:
 *   1. Declaring an integer on the host (`x`) and a host pointer to it (`p`).
 *   2. Launching a kernel that receives this pointer and tries to read
 *      the integer value by dereferencing it.
 *   3. Using `cudaDeviceSynchronize()` and `cudaGetLastError()` to capture
 *      and print the error that occurs because the device tries to access
 *      a host address.
 *
 * The expected output is an error message indicating an invalid device
 * pointer or invalid memory access, demonstrating that passing a host
 * address to a kernel and dereferencing it is not allowed.
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that attempts to dereference the passed pointer
__global__ void derefKernel(int *p)
{
    // Try to read the integer value from the pointer
    int val = *p; // This should cause an invalid memory access
    // Print the value (won't be executed if the above line fails)
    printf("Value read from device pointer: %d\n", val);
}

int main()
{
    // Host variable
    int x = 42;
    // Host pointer to the variable
    int *p = &x;

    // Launch kernel with the host pointer
    derefKernel<<<1, 1>>>(p);

    // Synchronize to catch any errors that occurred during kernel execution
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess)
    {
        printf("cudaDeviceSynchronize error: %s\n", cudaGetErrorString(syncErr));
    }

    // Check for launch errors
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess)
    {
        printf("Kernel launch error: %s\n", cudaGetErrorString(launchErr));
    }
    else
    {
        printf("Kernel launched successfully (unexpected)\n");
    }

    return 0;
}
```