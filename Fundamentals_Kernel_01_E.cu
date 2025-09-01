/*
Add `cudaDeviceSynchronize()` after the launch and print 'Kernel synchronized!' after it.

Thoughts:
- The program is a minimal CUDA C program that launches a simple kernel.
- After the kernel launch, we need to call cudaDeviceSynchronize() to block the host until the kernel completes.
- After synchronization, we print a message to confirm.
- Include error checking for kernel launch and synchronization.
- The kernel can just print a message from the device for demonstration.
- The host prints "Kernel synchronized!" after synchronization.
- This demonstrates correct use of cudaDeviceSynchronize and console output.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void simpleKernel()
{
    // Each thread prints a simple message (only one thread in this case)
    printf("Hello from the GPU kernel!\n");
}

int main(void)
{
    // Launch kernel with 1 block and 1 thread
    simpleKernel<<<1, 1>>>();

    // Check for any errors launching the kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch simpleKernel kernel (error code %s)!\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Synchronize device to wait for kernel completion
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %s!\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Print confirmation after synchronization
    printf("Kernel synchronized!\n");

    // Optional: Reset device (good practice)
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
