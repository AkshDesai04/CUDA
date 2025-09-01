/*
Write a program to determine if the primary GPU is an integrated (sharing memory with the CPU) or a discrete device. Print the result.

Thinking:
- Use the CUDA Runtime API to query device properties.
- The struct cudaDeviceProp contains a field named 'integrated' (an unsigned int).
  If this field is non-zero, the device is integrated and shares memory with the host CPU.
  If it is zero, the device is discrete and has its own dedicated memory.
- First, determine how many CUDA-capable devices are present with cudaGetDeviceCount().
  If no devices are found, we report that and exit gracefully.
- Next, retrieve the properties for the primary device (device index 0) using
  cudaGetDeviceProperties().
- Inspect the 'integrated' flag:
  * If it is set (1), print that the primary GPU is integrated.
  * Otherwise, print that the primary GPU is discrete.
- All CUDA API calls are checked for errors. On error, a descriptive message is printed
  and the program exits with a non-zero status.
- The program uses only the standard C headers and the CUDA runtime header, making it
  straightforward to compile with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0); // primary device is usually index 0
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed for device 0: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (prop.integrated) {
        printf("Primary GPU is integrated (shared memory with CPU).\n");
    } else {
        printf("Primary GPU is discrete.\n");
    }

    return 0;
}
