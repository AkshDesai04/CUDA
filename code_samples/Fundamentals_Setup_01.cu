/*
Aim: Write a CUDA program to query and print the number of CUDA-enabled devices available on your system. The program should use the `cudaGetDeviceCount` API call and print a user-friendly message, such as 'Found X CUDA-capable devices.'

Thinking:
1. Include necessary headers: <cuda_runtime.h> for CUDA API, <cstdio> for printf.
2. In main, declare an integer variable `deviceCount`.
3. Call `cudaGetDeviceCount(&deviceCount)`. This function returns a cudaError_t.
4. Check the return value. If it's not cudaSuccess, use `cudaGetErrorString` to print a descriptive error message and exit.
5. If successful, print the message: "Found X CUDA-capable devices." using printf, replacing X with `deviceCount`.
6. Return 0 from main.
7. The program is straightforward and doesn't launch any kernels. It's a host-only program that queries device information.
8. Ensure the code compiles with nvcc: `nvcc -o device_query device_query.cu`.
9. Include minimal error handling to keep the code clean.
*/

#include <cuda_runtime.h>
#include <cstdio>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaGetDeviceCount failed: %s\n",
                cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Found %d CUDA-capable device%s.\n",
           deviceCount,
           deviceCount == 1 ? "" : "s");

    return EXIT_SUCCESS;
}
