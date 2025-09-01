/*
Aim of the program: Write a `main` function that calls your function, then copies the zeroed array back to the host and verifies that all elements are indeed 0.

Thinking:
- The goal is to create a simple CUDA program that demonstrates copying data to GPU, performing a kernel operation (zeroing an array), copying it back, and validating the result.
- We need a kernel that writes zeros into each element of a device array. The kernel will be launched with enough blocks/threads to cover all elements.
- In `main`, we will:
  1. Allocate a host array and fill it with some non-zero values to ensure the kernel really modifies it.
  2. Allocate device memory with `cudaMalloc`.
  3. Copy the host array to the device with `cudaMemcpy`.
  4. Launch the zeroing kernel.
  5. Copy the device array back to the host.
  6. Verify each element is zero, printing a success message or a failure notice.
- We'll include a simple error-checking macro to catch CUDA errors.
- Choose a moderate array size (e.g., 1<<20 elements) and block size (256 threads per block) for demonstration.
- The program will be self-contained and compile with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",           \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Kernel that sets each element of the array to 0
__global__ void zeroArray(int *d_arr, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        d_arr[idx] = 0;
}

int main(void)
{
    const int N = 1 << 20;          // Array size: 1,048,576 elements
    const int bytes = N * sizeof(int);

    // Allocate and initialize host array with non-zero values
    int *h_arr = (int *)malloc(bytes);
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
        h_arr[i] = i + 1;  // arbitrary non-zero values
    }

    // Allocate device memory
    int *d_arr;
    CHECK_CUDA(cudaMalloc((void **)&d_arr, bytes));

    // Copy host array to device
    CHECK_CUDA(cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice));

    // Launch kernel: one thread per element
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    zeroArray<<<blocks, threadsPerBlock>>>(d_arr, N);
    CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA(cudaDeviceSynchronize()); // Wait for kernel to finish

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_arr, d_arr, bytes, cudaMemcpyDeviceToHost));

    // Verify that all elements are zero
    int allZero = 1;
    for (int i = 0; i < N; ++i) {
        if (h_arr[i] != 0) {
            fprintf(stderr, "Verification failed at index %d: %d\n", i, h_arr[i]);
            allZero = 0;
            break;
        }
    }

    if (allZero) {
        printf("Success: All elements are zero.\n");
    } else {
        printf("Failure: Some elements are not zero.\n");
    }

    // Clean up
    free(h_arr);
    CHECK_CUDA(cudaFree(d_arr));

    return EXIT_SUCCESS;
}
