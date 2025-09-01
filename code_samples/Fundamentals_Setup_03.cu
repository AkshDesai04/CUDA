/*
Aim: Write a program that queries and prints the total amount of global memory available on device 0 in megabytes (MB). You will need to get the `totalGlobalMem` property from `cudaDeviceProp` and convert it from bytes to MB.

Thinking:
1. Include the necessary headers:
   - <cuda_runtime.h> for CUDA runtime API functions and structures.
   - <cstdio> for standard I/O.
2. In `main`, declare a `cudaDeviceProp` structure to hold device properties.
3. Call `cudaGetDeviceProperties(&prop, 0)` to fill the structure for device 0.
4. Check the return status. If an error occurs, print an error message and exit.
5. Retrieve the total global memory from `prop.totalGlobalMem`, which is expressed in bytes.
6. Convert bytes to megabytes by dividing by 1024*1024. Use a floating‑point type to preserve fractional part.
7. Print the result using `printf`, formatting to two decimal places for readability.
8. Return 0 to indicate successful execution.

Edge Cases:
- If the system has no CUDA device, `cudaGetDeviceProperties` will return an error; the program will handle this gracefully.
- The calculation uses floating point to avoid integer division truncation.

*/

#include <cuda_runtime.h>
#include <cstdio>

int main() {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    size_t bytes = prop.totalGlobalMem;
    double megabytes = static_cast<double>(bytes) / (1024.0 * 1024.0);

    printf("Device 0 Total Global Memory: %.2f MB\n", megabytes);

    return 0;
}
