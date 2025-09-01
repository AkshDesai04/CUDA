/*
Aim of the program:
Write a helper function `void host_to_device_copy(int* h_data, int* d_data, int size)` that encapsulates the `cudaMemcpy` call.

My thinking:
- The function will copy an array of integers from host memory (`h_data`) to device memory (`d_data`) using the CUDA runtime API.
- It accepts the number of elements `size`; the actual byte count for `cudaMemcpy` will be `size * sizeof(int)`.
- For robustness, the function will perform a minimal error check on the result of `cudaMemcpy` and report failures to `stderr`.
- The code includes the necessary CUDA runtime header `<cuda_runtime.h>` and the C I/O header for error printing.
- The helper returns `void` and does not throw exceptions, keeping the API simple for use in a variety of contexts.
*/

#include <cuda_runtime.h>
#include <cstdio>

/**
 * @brief Copy data from host to device.
 *
 * This helper function wraps the cudaMemcpy call for copying an array of
 * integers from host memory to device memory. It performs basic error
 * checking and prints a message if the copy fails.
 *
 * @param h_data Pointer to the source array on the host.
 * @param d_data Pointer to the destination array on the device.
 * @param size   Number of elements (int) to copy.
 */
void host_to_device_copy(int* h_data, int* d_data, int size) {
    size_t bytes = static_cast<size_t>(size) * sizeof(int);
    cudaError_t err = cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying data from host to device: %s\n", cudaGetErrorString(err));
    }
}
