```cpp
/*
Write a program to determine and print the 'warp size' for the default CUDA device. The warp size is a fundamental property that dictates how threads are scheduled. Find this value in the device properties struct.

Thinking:
1. The program must query the properties of the default CUDA device (device 0) using the CUDA Runtime API.
2. The relevant property is `warpSize` in the `cudaDeviceProp` struct.
3. Include necessary headers: <cuda_runtime.h> for CUDA runtime API and <stdio.h> for I/O.
4. Use `cudaGetDeviceProperties()` to fill a `cudaDeviceProp` structure.
5. Handle any errors from the CUDA call by printing an error message and exiting with a non-zero status.
6. Print the device name and warp size to stdout.
7. Keep the program simple and self-contained so it can be compiled directly with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0;  // Default CUDA device
    cudaDeviceProp prop;

    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Device %d: %s\n", device, prop.name);
    printf("Warp size: %d\n", prop.warpSize);

    return 0;
}
```