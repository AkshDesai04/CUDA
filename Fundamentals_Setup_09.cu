```cuda
/*
Write a program to check and print whether the primary GPU supports running multiple kernels concurrently.
The property you are looking for is `concurrentKernels` in `cudaDeviceProp`.

Thought process:
- The goal is to query the device properties of the primary GPU (device 0) and examine the 
  `concurrentKernels` flag to determine if it can run multiple kernels at the same time.
- Use the CUDA runtime API: `cudaGetDeviceProperties` to fill a `cudaDeviceProp` struct.
- Check the `concurrentKernels` integer: a non-zero value indicates support.
- Provide simple error checking for the CUDA API calls.
- Print a clear message indicating whether concurrent kernel execution is supported.
- Keep the program minimal and self‑contained: include <cuda_runtime.h> and <stdio.h>.
- The code can be compiled with `nvcc` and run on any system with CUDA installed.
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
    int device = 0; // Primary GPU
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error retrieving device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Device %d: %s\n", device, prop.name);
    if (prop.concurrentKernels) {
        printf("Concurrent kernel execution is supported.\n");
    } else {
        printf("Concurrent kernel execution is NOT supported.\n");
    }

    return 0;
}
```