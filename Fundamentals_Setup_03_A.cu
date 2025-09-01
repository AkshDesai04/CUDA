/*
Query and print the total amount of constant memory available on device 0 in kilobytes (KB).

Thinking:
- The goal is to retrieve the constant memory size for CUDA device 0.
- CUDA provides device properties via cudaGetDeviceProperties, which includes totalConstMem in bytes.
- We'll query device 0, convert bytes to kilobytes by dividing by 1024.0.
- The program will simply print this value.
- Error checking for cudaGetDeviceProperties is added for robustness.
- The output will be a single line showing the constant memory in KB with two decimal places.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    cudaError_t err;
    int device = 0;

    // Query device properties for device 0
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties for device %d: %s\n", device, cudaGetErrorString(err));
        return 1;
    }

    // Convert constant memory size from bytes to kilobytes
    double kb = prop.totalConstMem / 1024.0;

    // Print the result
    printf("Total constant memory on device %d: %.2f KB\n", device, kb);

    return 0;
}
