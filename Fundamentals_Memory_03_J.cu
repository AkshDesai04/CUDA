/*
Aim: Write a generic `device_copy` function that takes two `void*` pointers, a size, and calls `cudaMemcpy` with the DtoD kind.

Thinking:
- The function must be usable from C/C++ code and operate on raw pointers, so `void*` is appropriate for both source and destination.
- The transfer is purely on the device, thus `cudaMemcpyDeviceToDevice` is the required `cudaMemcpyKind`.
- Returning the `cudaError_t` from `cudaMemcpy` allows the caller to detect and handle any issues.
- No special error handling or kernel launches are needed; a simple wrapper suffices.
- Include `<cuda_runtime.h>` for the CUDA runtime API.
- The code can be compiled with `nvcc` as a standard `.cu` file.
*/

#include <cuda_runtime.h>

/**
 * @brief Generic device-to-device memory copy.
 *
 * Copies `size` bytes from `src` to `dst` on the GPU.
 *
 * @param dst   Destination device pointer.
 * @param src   Source device pointer.
 * @param size  Number of bytes to copy.
 * @return cudaError_t status of the copy operation.
 */
cudaError_t device_copy(void* dst, const void* src, size_t size)
{
    return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
}

/* Example usage (optional):
int main(void) {
    const size_t n = 1024;
    int *d_a, *d_b;
    cudaMalloc(&d_a, n * sizeof(int));
    cudaMalloc(&d_b, n * sizeof(int));

    // Fill d_a with data, then copy to d_b
    cudaError_t err = device_copy(d_b, d_a, n * sizeof(int));
    if (err != cudaSuccess) {
        printf("Copy failed: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
}
*/
