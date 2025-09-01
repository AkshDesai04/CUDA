```cpp
/*
Check and print if the device can overlap a `memcpy` operation with kernel execution (`deviceOverlap`).

Thinking:
1. We need to query the CUDA device attribute `cudaDevAttrDeviceOverlap` which indicates whether the device
   can overlap memory copies with kernel execution.
2. The CUDA Runtime API function `cudaDeviceGetAttribute` retrieves this attribute.
3. First, obtain the current device ID using `cudaGetDevice`.
4. Call `cudaDeviceGetAttribute` passing the address of an int to receive the attribute value,
   the enum `cudaDevAttrDeviceOverlap`, and the device ID.
5. Check for any CUDA errors returned by `cudaDeviceGetAttribute`. If an error occurs, print it
   and exit.
6. If the attribute value is non-zero, print that the device supports overlapping memcpy with
   kernel execution; otherwise print that it does not.
7. Use standard C++/C headers and a simple `main` function. No additional CUDA kernels are needed
   because we are only querying device capability.
8. The program will compile as a CUDA source file (.cu) with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Unable to get current device: %s\n", cudaGetErrorString(err));
        return 1;
    }

    int deviceOverlap = 0;
    err = cudaDeviceGetAttribute(&deviceOverlap, cudaDevAttrDeviceOverlap, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Unable to get device attribute cudaDevAttrDeviceOverlap: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceOverlap) {
        printf("Device %d supports overlapping memcpy with kernel execution.\n", device);
    } else {
        printf("Device %d does NOT support overlapping memcpy with kernel execution.\n", device);
    }

    return 0;
}
```