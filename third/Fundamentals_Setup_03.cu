```cpp
/*
Aim of the program:
Write a program that queries and prints the total amount of global memory available on device 0 in megabytes (MB). 
You will need to get the `totalGlobalMem` property from `cudaDeviceProp` and convert it from bytes to MB.

Thinking:
1. Include necessary headers: <cstdio> for printf and <cuda_runtime.h> for CUDA API.
2. In main, declare a cudaDeviceProp struct and use cudaGetDeviceProperties(&prop, 0) to fill it for device 0.
3. Check for errors from cudaGetDeviceProperties. If an error occurs, print an error message and exit with non-zero status.
4. Convert prop.totalGlobalMem (bytes) to megabytes by dividing by (1024 * 1024). Use a double or float for fractional MB.
5. Print the result using printf, formatting with e.g., "%.2f MB" for two decimal places.
6. Return 0 on success.
*/

#include <cstdio>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    int device = 0;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    double totalMemMB = static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0);
    printf("Total global memory on device %d: %.2f MB\n", device, totalMemMB);

    return 0;
}
```