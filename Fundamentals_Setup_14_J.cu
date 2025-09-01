/*
Query and print the maximum pitch in bytes for a memory allocation.

Thinking process:
1. The program should be written in CUDA C and compiled with nvcc.  
2. To obtain the maximum pitch a device supports for memory allocation, we use the CUDA runtime API function
   `cudaDeviceGetAttribute`. The relevant attribute is `cudaDevAttrMaxPitch`.  
3. The function signature is `cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device)`.  
4. The attribute value is returned as an `int` representing the number of bytes.  
5. We will check the return value of the function for errors and print an error message if it fails.  
6. The program will set the device to device 0 by default (the first CUDA device).  
7. The output will be the maximum pitch in bytes for that device.  
8. The code will include the necessary headers `<stdio.h>` and `<cuda_runtime.h>`.  
9. A helper macro for checking CUDA errors will be omitted for brevity; the error handling is done inline.  
10. Finally, we output a single integer value preceded by a descriptive message.

The program is simple and self-contained; compiling it with `nvcc` should produce an executable that prints the desired value.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0;
    int maxPitch = 0;
    cudaError_t err;

    // Optional: set the current device (defaults to 0, but be explicit)
    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Query the maximum pitch supported by the device
    err = cudaDeviceGetAttribute(&maxPitch, cudaDevAttrMaxPitch, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetAttribute failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Maximum pitch on device %d is %d bytes.\n", device, maxPitch);
    return 0;
}
