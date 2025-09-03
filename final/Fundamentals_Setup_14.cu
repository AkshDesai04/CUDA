/*
Write a program to check if the primary GPU can overlap data copying with kernel execution. This capability is crucial for performance and is indicated by the `asyncEngineCount` property (a value > 0).

Thinking:
1. The goal is to determine whether the primary GPU supports overlapping of memory copy operations with kernel execution.
2. CUDA exposes the `asyncEngineCount` field in the `cudaDeviceProp` structure, which indicates how many asynchronous engines the device has. If this value is greater than zero, the GPU can perform concurrent copy and execution.
3. The program will:
   - Include the necessary CUDA runtime headers.
   - Query the properties of the default device (device 0).
   - Check the `asyncEngineCount` field.
   - Print a clear message indicating whether the GPU supports overlapping.
4. Error handling will be included to catch any failures from CUDA API calls.
5. The program is kept simple and portable, focusing solely on the property check. It can be compiled with `nvcc` and run on systems with CUDA installed.

No additional functionality (like launching a kernel) is needed for this capability check; just inspecting the device property suffices.
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
    cudaError_t err;
    int device = 0; // Primary GPU

    // Get device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Device %d: %s\n", device, prop.name);
    printf("asyncEngineCount: %d\n", prop.asyncEngineCount);

    if (prop.asyncEngineCount > 0) {
        printf("Result: This GPU supports overlapping of data copying with kernel execution.\n");
    } else {
        printf("Result: This GPU does NOT support overlapping of data copying with kernel execution.\n");
    }

    return 0;
}
