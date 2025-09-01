/*
Aim: Query and print the total amount of constant memory available on device 0 in kilobytes (KB).

Thinking process:
1. The CUDA runtime API provides a `cudaDeviceProp` structure that holds various properties of a device, including the amount of constant memory available (`totalConstMem` in bytes).
2. To retrieve this information for device 0, use `cudaGetDeviceProperties(&prop, 0)`.
3. Convert the byte value to kilobytes by dividing by 1024.
4. Print the result using `printf`, ensuring the size type is printed correctly with `%zu`.
5. Include basic error checking: if `cudaGetDeviceProperties` fails, print an error message and exit.
6. Include necessary headers: `<stdio.h>` for I/O and `<cuda_runtime.h>` for CUDA runtime functions.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties for device %d: %s\n",
                device, cudaGetErrorString(err));
        return -1;
    }

    size_t totalConstMemBytes = prop.totalConstMem;
    size_t totalConstMemKB = totalConstMemBytes / 1024;  // Convert to kilobytes

    printf("Total constant memory on device %d: %zu KB\n", device, totalConstMemKB);

    return 0;
}
