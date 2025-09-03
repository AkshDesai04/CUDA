```c
/*
Aim of the program: Try to get device properties for an invalid device index with cudaGetDeviceProperties.
 
Thinking process:
1. I want to demonstrate what happens when cudaGetDeviceProperties is called with a device index that is not valid.
2. CUDA device indices range from 0 to (deviceCount - 1). Any index outside this range is invalid.
3. I'll first query the number of CUDA-capable devices with cudaGetDeviceCount. This is optional, but useful to determine an out-of-range index.
4. I'll construct an invalid index by adding 1 to the device count (or using -1). Both are invalid, but adding 1 ensures it is out of bounds even if count is 0.
5. I'll call cudaGetDeviceProperties with this invalid index. The function returns a cudaError_t value indicating the error.
6. I'll print the returned error code and its human-readable string using cudaGetErrorString to show the failure.
7. The program will compile as a standard .cu file, include <cuda_runtime.h> and <stdio.h>, and contain a simple main() that performs these steps.
8. No additional functionality is needed; the goal is solely to trigger and display the error for an invalid device index.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Choose an invalid device index: deviceCount is out of range (valid indices are 0..deviceCount-1)
    int invalidIndex = deviceCount; // If deviceCount is 0, this is also invalid

    // Alternatively, could use -1 to guarantee invalid index:
    // int invalidIndex = -1;

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, invalidIndex);
    if (err != cudaSuccess) {
        printf("Attempted to get properties for invalid device index %d.\n", invalidIndex);
        printf("cudaGetDeviceProperties returned error: %s (%d)\n", cudaGetErrorString(err), err);
    } else {
        // This branch should not execute for an invalid index, but handle just in case
        printf("Device %d properties retrieved (unexpectedly):\n", invalidIndex);
        printf("  Name: %s\n", prop.name);
        printf("  Total Global Memory: %lu bytes\n", prop.totalGlobalMem);
    }

    return 0;
}
```