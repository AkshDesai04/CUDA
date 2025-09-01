```cpp
/* 
Aim of the program:
Write a program to check and print whether the primary GPU supports running multiple kernels concurrently. 
The property you are looking for is concurrentKernels in cudaDeviceProp.

Thinking:
1. Use CUDA Runtime API to query device properties.
2. First, get the number of CUDA devices via cudaGetDeviceCount.
3. If at least one device is present, select the primary device (device 0).
4. Call cudaGetDeviceProperties to fill a cudaDeviceProp structure.
5. Inspect the concurrentKernels field of the structure; if it is non-zero, the GPU can run multiple kernels concurrently.
6. Print a user-friendly message indicating support or lack thereof.
7. Include error checking after each CUDA API call for robustness.
8. Compile with nvcc to produce a .cu file that can be executed on a machine with a CUDA-capable GPU.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable device found.\n");
        return 0;
    }

    int device = 0; // Primary device
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n", device, cudaGetErrorString(err));
        return 1;
    }

    printf("Device %d: %s\n", device, prop.name);
    if (prop.concurrentKernels) {
        printf("This GPU supports running multiple kernels concurrently.\n");
    } else {
        printf("This GPU does NOT support running multiple kernels concurrently.\n");
    }

    return 0;
}
```