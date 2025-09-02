/*
Write a CUDA program to query and print the number of CUDA-enabled devices available on your system. The program should use the `cudaGetDeviceCount` API call and print a user-friendly message, such as 'Found X CUDA-capable devices.'

[Thinking]
The task is straightforward: we need a minimal CUDA C program that queries the number of CUDA-capable devices and prints a friendly message. 
1. Include the CUDA runtime header `cuda_runtime.h` for the API function.
2. Use `int main()` as the entry point.
3. Declare an `int deviceCount` and call `cudaGetDeviceCount(&deviceCount)`.
4. Check the returned `cudaError_t` for success; if there's an error, print an informative message and exit with a non‑zero status.
5. If successful, print "Found X CUDA-capable devices." where X is `deviceCount`.
6. Return 0 on success.
This program requires no kernel launch or device code, so it can be compiled with `nvcc` and run on any machine with CUDA installed.

*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Found %d CUDA-capable device%s.\n", deviceCount,
           deviceCount == 1 ? "" : "s");
    return 0;
}
