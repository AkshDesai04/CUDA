/*
Write a helper function `void host_to_device_copy(int* h_data, int* d_data, int size)` that encapsulates the `cudaMemcpy` call.

Thoughts:
- The function should take a pointer to host data `h_data`, a pointer to device data `d_data`, and the number of integers `size`.
- It will perform a host-to-device memory copy using `cudaMemcpy`.
- Error checking is important: after the copy, we should check the return status and handle any errors gracefully.
- We'll include the necessary header `<cuda_runtime.h>` to access CUDA runtime API functions.
- The function signature uses `int*` for both host and device pointers; the user will allocate device memory separately using `cudaMalloc`.
- The copy size in bytes is `size * sizeof(int)`.
- We will use `cudaMemcpyHostToDevice` as the direction flag.
- For error handling, we can print an error message to `stderr` and exit the program if the copy fails, ensuring the user is aware of any problems.
- A minimal `main` function is not required, but adding one is optional for demonstration. However, per instruction, only provide the helper function and necessary includes.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void host_to_device_copy(int* h_data, int* d_data, int size) {
    // Perform the host-to-device copy
    cudaError_t err = cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in host_to_device_copy: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
