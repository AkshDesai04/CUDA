```cuda
/*
 * Aim of the program:
 * Print the total constant memory on the device in bytes.
 *
 * Thinking:
 * 1. We want to query the device's total constant memory, which is part of the
 *    cudaDeviceProp structure returned by cudaGetDeviceProperties.
 * 2. The program will:
 *    a. Obtain the current device ID using cudaGetDevice.
 *    b. Retrieve the properties of that device with cudaGetDeviceProperties.
 *    c. Extract the field 'totalConstMem' from the struct, which gives the
 *       amount of constant memory in bytes.
 *    d. Print the value to the standard output.
 * 3. Error handling: All CUDA API calls return cudaError_t; we check for
 *    errors and print messages using cudaGetErrorString when necessary.
 * 4. The program is self-contained and can be compiled with nvcc:
 *    nvcc -o const_mem total_const_mem.cu
 *    Then run ./const_mem to see the output.
 */

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int device_id = 0;
    cudaError_t err;

    // Get the current device ID (default is 0 if none set)
    err = cudaGetDevice(&device_id);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get current device: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Retrieve device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get device properties for device %d: %s\n",
                device_id, cudaGetErrorString(err));
        return 1;
    }

    // Print total constant memory in bytes
    printf("Total constant memory on device %d: %zu bytes\n", device_id, prop.totalConstMem);

    return 0;
}
```