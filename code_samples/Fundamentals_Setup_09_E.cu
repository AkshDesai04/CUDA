```cpp
/*
Iterate through all devices and print the `concurrentKernels` status for each one.

Thinking:
1. Use cudaGetDeviceCount to determine how many CUDA-capable devices are present.
2. Iterate from device 0 to count-1.
3. For each device, call cudaGetDeviceProperties to obtain a cudaDeviceProp struct.
4. The field `concurrentKernels` indicates whether concurrent kernel execution is supported.
   It is an int: 1 if supported, 0 otherwise.
5. Print the device index and the status in a readable form.
6. Handle any CUDA errors by printing an error message and exiting.
7. Ensure the program can be compiled with nvcc and runs on any system with CUDA installed.
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices detected.\n");
        return 0;
    }

    printf("Detected %d CUDA-capable device(s).\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to get properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: \"%s\"\n", dev, prop.name);
        printf("  concurrentKernels support: %s\n", prop.concurrentKernels ? "YES" : "NO");
    }

    return 0;
}
```