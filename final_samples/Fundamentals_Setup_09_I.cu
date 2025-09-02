```c
/*
Print the amount of shared memory available per multiprocessor.

Thought Process:
- The goal is to query CUDA device properties to obtain the amount of
  shared memory that each multiprocessor (SM) can provide.
- The CUDA Runtime API offers `cudaGetDeviceProperties`, which fills a
  `cudaDeviceProp` structure. The field `sharedMemPerMultiprocessor`
  contains the requested information.
- For a simple standalone program:
  1. Query device count to ensure at least one device is present.
  2. Use device 0 (or any valid device) to get its properties.
  3. Print the value in bytes, and optionally convert to KB/MB for readability.
- Error handling: check return status of `cudaGetDeviceProperties` and
  `cudaGetDeviceCount`; on failure, print a message and exit with
  non-zero code.
- The program will be compiled with `nvcc` as a .cu file.
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
        fprintf(stderr, "No CUDA-capable device found.\n");
        return 1;
    }

    int device = 0; // Use the first device
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Device %d: %s\n", device, prop.name);
    printf("Shared memory per multiprocessor: %lu bytes\n", prop.sharedMemPerMultiprocessor);
    printf("Shared memory per multiprocessor (KB): %lu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("Shared memory per multiprocessor (MB): %.3f MB\n",
           prop.sharedMemPerMultiprocessor / (1024.0 * 1024.0));

    return 0;
}
```