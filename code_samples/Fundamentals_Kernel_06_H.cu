```cuda
/*
 * Allocate memory, but forget to copy the pointer's value back from the device.
 * Print the host variable. What does it show and why?
 *
 * Aim of the program: Demonstrate that if we allocate memory on the device,
 * write to it, and then do NOT copy the result back to the host, the host
 * variable remains unchanged (its original value). Printing the host
 * variable will therefore show its initial value (e.g., 0) and not the
 * value written on the device.
 *
 * Reasoning:
 *   1. The host variable 'hostVal' is initialized to 0.
 *   2. Device memory 'devPtr' is allocated with cudaMalloc.
 *   3. A simple kernel writes the value 42 into the device memory.
 *   4. We intentionally omit the cudaMemcpy that would bring the data
 *      back from the device to 'hostVal'.
 *   5. Printing 'hostVal' shows 0, because the host memory was never
 *      updated. The device memory and host memory are separate,
 *      and without an explicit copy the host never sees the value
 *      written on the device.
 *
 * This program thus illustrates the necessity of performing
 * cudaMemcpy (or other appropriate device-to-host transfer) to
 * synchronize data between the device and the host.
 */

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void write_value(int *dptr, int val)
{
    *dptr = val;
}

int main()
{
    int hostVal = 0;               // Host variable initialized to 0
    int *devPtr = NULL;            // Device pointer

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&devPtr, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch kernel to write 42 to device memory
    write_value<<<1, 1>>>(devPtr, 42);
    cudaDeviceSynchronize();

    // Intentionally omitted: cudaMemcpy(&hostVal, devPtr, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the host variable
    printf("hostVal = %d\n", hostVal);  // Expected to print 0

    // Clean up
    cudaFree(devPtr);
    return 0;
}
```