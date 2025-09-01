```cpp
/*
Aim of the program: 
Write a program that queries and prints the total amount of global memory available on device 0 in megabytes (MB). 
You will need to get the totalGlobalMem property from cudaDeviceProp and convert it from bytes to MB.

Thinking process:
1. We need a CUDA C/C++ program that uses the CUDA runtime API.
2. The key function is cudaGetDeviceProperties which fills a cudaDeviceProp struct with information about a GPU.
3. The struct contains a field called totalGlobalMem which is the total amount of global memory in bytes.
4. Convert bytes to megabytes by dividing by (1024 * 1024).
5. Print the result to stdout using printf.
6. Add basic error checking: after calling cudaGetDeviceProperties, check if the return status is cudaSuccess. If not, print an error and exit.
7. Use #include <cstdio> for printf, and #include <cuda_runtime.h> for CUDA runtime functions.
8. The program will be a single .cu file with a main() that performs these steps.
9. No device code is required; this is purely host code.
10. Keep the code minimal, clear, and compilable with nvcc.
*/

#include <cstdio>
#include <cuda_runtime.h>

int main()
{
    // Query properties of device 0
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Convert bytes to megabytes
    const double bytesPerMB = 1024.0 * 1024.0;
    double totalMB = static_cast<double>(prop.totalGlobalMem) / bytesPerMB;

    // Print the result
    printf("Device 0 total global memory: %.2f MB\n", totalMB);

    return 0;
}
```