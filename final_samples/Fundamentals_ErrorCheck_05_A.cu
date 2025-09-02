/*
First, get the device count. Then, try to set the device to `count` (which is an invalid index, since they are 0-based).

**Thinking Process:**

1. The program’s aim is to demonstrate the behavior of the CUDA runtime when an invalid device index is used.  
2. We will call `cudaGetDeviceCount` to obtain the number of CUDA-capable devices available on the system.  
3. `cudaGetDeviceCount` returns the count as a positive integer (e.g., 1, 2, …).  
4. CUDA device indices are 0‑based, so the valid range for `cudaSetDevice` is `[0, count-1]`.  
5. By intentionally passing the value `count` to `cudaSetDevice`, we provide an index that is *one* larger than the highest valid index.  
6. This should trigger an error from the runtime – typically `cudaErrorInvalidDevice`.  
7. The program will capture this error, print the error code and message, and exit.  
8. This exercise is useful for testing error handling and for understanding the importance of valid device indices.

The code below performs these steps and prints the result of the attempted device set operation.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("Failed to get device count: %s (%d)\n", cudaGetErrorString(err), err);
        return -1;
    }

    printf("Number of CUDA devices found: %d\n", deviceCount);

    // Intentionally use an invalid device index: count (which is one greater than the max valid index)
    int invalidIndex = deviceCount;
    err = cudaSetDevice(invalidIndex);
    if (err != cudaSuccess) {
        printf("Attempt to set device to %d failed: %s (%d)\n", invalidIndex, cudaGetErrorString(err), err);
    } else {
        printf("Successfully set device to %d (unexpected)\n", invalidIndex);
    }

    return 0;
}
