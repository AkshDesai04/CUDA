```cuda
/*
Aim of the program:
Modify the program to print the L2 cache size in megabytes (MB) with one decimal place.

Thinking:
- We need to query the CUDA device properties to obtain the L2 cache size. The CUDA runtime API provides a `cudaDeviceProp` structure which contains a field `l2CacheSize` that holds the size in bytes.
- The size should be converted from bytes to megabytes. One megabyte is 1024 * 1024 bytes. After conversion we format the output with one decimal place using the format specifier `%.1f`.
- The program will query the first available CUDA device (device 0). If multiple devices exist, we could extend the program to loop over them, but for simplicity we handle a single device.
- We include necessary headers `<stdio.h>` and `<cuda_runtime.h>`. 
- The main function will:
  1. Get the number of CUDA devices.
  2. If none, print an error and exit.
  3. Get the properties of device 0.
  4. Compute L2 cache size in MB.
  5. Print it with one decimal place.
- No additional supporting files or information are required; everything is contained within this single .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device count: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return -1;
    }

    // Query properties of device 0
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device properties: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Convert L2 cache size from bytes to megabytes
    double l2SizeMB = (double)prop.l2CacheSize / (1024.0 * 1024.0);

    printf("L2 Cache Size: %.1f MB\n", l2SizeMB);

    return 0;
}
```