/*
Write a generic `device_copy` function that takes two `void*` pointers, a size, and calls `cudaMemcpy` with the DtoD kind.

Thoughts:
1. The function should be as generic as possible, so we use `void*` for source and destination pointers and a `size_t` for the number of bytes to copy. 
2. The CUDA API provides `cudaMemcpy` with a `cudaMemcpyKind` parameter; for device-to-device we use `cudaMemcpyDeviceToDevice`. 
3. Returning a `cudaError_t` allows the caller to check for success or failure. 
4. In the example `main`, we allocate two device arrays, copy data from host to one device array, use `device_copy` to copy from that array to the second, and finally copy back to host to verify. 
5. We include basic error checking and cleanup of device memory. 
6. The file is a complete CUDA C file that can be compiled with `nvcc`. 
7. No external files or headers are required beyond the standard CUDA runtime header and stdio. 
8. The function itself is trivial, but the surrounding code demonstrates its usage and verifies correctness. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

/**
 * @brief Generic device-to-device memory copy.
 *
 * Copies `size` bytes from `src` to `dst`, both of which must be device pointers.
 *
 * @param dst Destination device pointer.
 * @param src Source device pointer.
 * @param size Number of bytes to copy.
 * @return cudaError_t CUDA error status.
 */
cudaError_t device_copy(void* dst, const void* src, size_t size) {
    return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
}

int main() {
    const size_t N = 10;
    int h_A[N], h_B[N];

    /* Initialize host array */
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<int>(i);
    }

    /* Allocate device memory */
    int *d_A = nullptr;
    int *d_B = nullptr;
    cudaError_t err;

    err = cudaMalloc(&d_A, N * sizeof(int));
    if (err != cudaSuccess) {
        printf("cudaMalloc d_A failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc(&d_B, N * sizeof(int));
    if (err != cudaSuccess) {
        printf("cudaMalloc d_B failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        return 1;
    }

    /* Copy host data to device */
    err = cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("cudaMemcpy HtoD failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return 1;
    }

    /* Use the generic device_copy function */
    err = device_copy(d_B, d_A, N * sizeof(int));
    if (err != cudaSuccess) {
        printf("device_copy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return 1;
    }

    /* Copy result back to host */
    err = cudaMemcpy(h_B, d_B, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy DtoH failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return 1;
    }

    /* Verify and print result */
    printf("Copied array: ");
    for (size_t i = 0; i < N; ++i) {
        printf("%d ", h_B[i]);
    }
    printf("\n");

    /* Clean up */
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
