/*
Aim of the program: Query and print the maximum pitch in bytes for a memory allocation.

Thought process:
1. The CUDA Runtime API provides device properties via cudaGetDeviceProperties().
   The struct cudaDeviceProp contains a field 'maxPitch' that indicates the maximum
   pitch (in bytes) allowed for a memory allocation created with cudaMallocPitch.
2. The program will:
   - Determine the current device (we'll simply use device 0 for simplicity).
   - Call cudaGetDeviceProperties() to obtain the properties of that device.
   - Print the value of maxPitch.
3. Error handling will be minimal: if cudaGetDeviceProperties fails, the program
   will report the error and exit.
4. The output will be a single line stating the maximum pitch in bytes, e.g.:
      Maximum pitch: 2147483648 bytes
5. Since the program is meant to be compiled as a .cu file, we include the
   necessary headers and use the standard C main function.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0;  // Use device 0
    cudaDeviceProp prop;
    cudaError_t err;

    // Get properties of the selected device
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Print the maximum pitch in bytes
    printf("Maximum pitch: %llu bytes\n", (unsigned long long)prop.maxPitch);

    return 0;
}
