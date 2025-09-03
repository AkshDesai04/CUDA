/*
Aim of the program:
Write a program that prints all key resource limits for a block: max threads, max shared memory, and max registers.

Thinking process:
1. The goal is to query CUDA device properties that describe block-level resource limits.
2. The CUDA runtime API provides `cudaGetDeviceProperties` which fills a `cudaDeviceProp` structure.
3. The relevant fields in `cudaDeviceProp` are:
   - `maxThreadsPerBlock`  → maximum number of threads that can be launched in a single block.
   - `sharedMemPerBlock`   → maximum amount of shared memory (in bytes) that can be allocated per block.
   - `regsPerBlock`        → maximum number of 32‑bit registers that can be used by a single block.
4. We will query properties for the current device (device 0 for simplicity).  
   In a more robust implementation we might detect the active device or iterate over all devices.
5. The program will handle errors returned by CUDA API calls and report them using `cudaGetErrorString`.
6. Finally, the values will be printed to stdout with clear labels.
7. Since this is a small utility, no kernel launch is necessary; the program runs on the host.
8. The code will be compiled with `nvcc` and produce a `.cu` file containing the complete program.

*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0; // default to device 0
    cudaDeviceProp prop;
    cudaError_t err;

    // Get properties of the chosen device
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Print the resource limits
    printf("CUDA Device %d Properties:\n", device);
    printf("  Maximum threads per block   : %d\n", prop.maxThreadsPerBlock);
    printf("  Maximum shared memory per block : %zu bytes\n", prop.sharedMemPerBlock);
    printf("  Maximum registers per block : %d\n", prop.regsPerBlock);

    return 0;
}
