```c
/*
Aim of the program:
Iterate through all devices and print the `concurrentKernels` status for each one.

Thinking process:
1. To obtain the number of CUDA-capable devices, use cudaGetDeviceCount().
2. For each device index from 0 to count-1:
   a. Call cudaGetDeviceProperties() to retrieve a cudaDeviceProp structure.
   b. Access the 'concurrentKernels' field of that structure, which indicates
      whether the device supports concurrent kernel execution.
   c. Print the device index, name, and the concurrentKernels status (1 for true,
      0 for false).
3. Handle potential errors: if any CUDA API call fails, print the error string
   returned by cudaGetErrorString() and exit with a non-zero status.
4. Use standard C I/O (printf) for output; no special CUDA kernels are needed.
5. Ensure the program compiles with nvcc and runs on systems with NVIDIA GPUs.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    printf("Found %d CUDA device(s).\n", deviceCount);
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error getting properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }
        printf("Device %d: %s\n", dev, prop.name);
        printf("  Concurrent Kernels support: %s\n", prop.concurrentKernels ? "Yes" : "No");
    }

    return 0;
}
```