/*
Change the data type from `int` to `float` and initialize the host array with floating-point values.
The goal of this exercise is to modify a CUDA C program that originally used integers to now use floating-point numbers. The host (CPU) side will allocate and initialize two arrays of floats, copy them to the device (GPU), perform an element-wise addition on the GPU, copy the result back to the host, and finally print a few elements of the result to verify correctness. 

Key points considered during the rewrite:
1. **Kernel signature and types**: Changed from `int*` to `float*` and adjusted all arithmetic to use floating-point.
2. **Memory allocation**: `cudaMalloc` and `cudaMemcpy` use `float` pointers; the size of the allocation is `N * sizeof(float)` instead of `N * sizeof(int)`.
3. **Host initialization**: Instead of integer values, each element is set to a floating-point number, e.g., `A[i] = 1.0f * i` and `B[i] = 2.0f * i`.
4. **Thread indexing**: Remains the same; we still launch enough blocks to cover `N` elements, with each thread handling one element.
5. **Result verification**: After copying back, we print the first 10 results to check that `C[i] = A[i] + B[i]` holds.
6. **Error handling**: Basic `cudaGetLastError()` checks are added for each CUDA API call.
7. **Clean-up**: All device memory is freed before exiting.

This code demonstrates the straightforward transition from integer to floating-point arithmetic in a simple CUDA kernel while maintaining the same overall program structure and logic.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024          // Size of the arrays
#define THREADS_PER_BLOCK 256

// CUDA kernel for vector addition: C = A + B
__global__ void vectorAdd(const float *A, const float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

int main(void)
{
    float *h_A, *h_B, *h_C;          // Host arrays
    float *d_A, *d_B, *d_C;          // Device arrays
    size_t size = N * sizeof(float);

    // Allocate host memory
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays with floating-point values
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f * i;   // Example: 0.0, 1.0, 2.0, ...
        h_B[i] = 2.0f * i;   // Example: 0.0, 2.0, 4.0, ...
    }

    // Allocate device memory
    cudaError_t err;
    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    err = cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_A);
        return EXIT_FAILURE;
    }
    err = cudaMalloc((void **)&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return EXIT_FAILURE;
    }

    // Copy host memory to device
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Launch kernel
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Copy result back to host
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Verify the result for the first 10 elements
    printf("First 10 results of vector addition (C = A + B):\n");
    for (int i = 0; i < 10; ++i) {
        printf("C[%d] = %f (Expected %f)\n", i, h_C[i], h_A[i] + h_B[i]);
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return EXIT_SUCCESS;
}
