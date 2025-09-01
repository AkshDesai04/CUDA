/*
Aim: Write a kernel to achieve the same result: `kernel_set_zero(float* arr, int N)` where each thread sets one element to `0.0f`.

Thinking:
The goal is to demonstrate a simple CUDA kernel that zeroes out an array. Each element of the array will be assigned to `0.0f` by a single thread.  
Key points:
- The kernel will be launched with a sufficient number of threads so that each element of the array has at least one thread responsible for it.  
- Each thread computes its global index from block and thread indices and checks that it is within bounds before writing to the array.  
- We’ll provide a small host program that allocates memory on both host and device, initializes the array with non‑zero values, copies it to the device, launches the kernel, copies the results back, and verifies that every element is zero.  
- Basic error checking macros will be used to catch CUDA API errors.  
- For simplicity we will use a 1D launch configuration and choose a block size of 256 threads, which is a common choice. The grid size will be calculated as `(N + blockSize - 1) / blockSize` to cover all elements.  
- The kernel itself is trivial:  
  ```cpp
  __global__ void kernel_set_zero(float* arr, int N) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < N) arr[idx] = 0.0f;
  }
  ```  
- The host code will allocate an array of size N (e.g., 1024 elements), fill it with random non‑zero floats, then invoke the kernel and check the result.

Now, the full CUDA program follows below.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

__global__ void kernel_set_zero(float* arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        arr[idx] = 0.0f;
    }
}

int main() {
    const int N = 1024;                 // Number of elements
    const int blockSize = 256;          // Threads per block
    const int gridSize = (N + blockSize - 1) / blockSize;

    // Allocate host memory
    float *h_arr = (float*)malloc(N * sizeof(float));
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host array with random non-zero values
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; ++i) {
        h_arr[i] = ((float)rand() / RAND_MAX) * 10.0f + 1.0f; // avoid zeros
    }

    // Allocate device memory
    float *d_arr;
    CUDA_CHECK(cudaMalloc((void**)&d_arr, N * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    kernel_set_zero<<<gridSize, blockSize>>>(d_arr, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify that all elements are zero
    int success = 1;
    for (int i = 0; i < N; ++i) {
        if (h_arr[i] != 0.0f) {
            fprintf(stderr, "Element %d is not zero: %f\n", i, h_arr[i]);
            success = 0;
            break;
        }
    }

    if (success) {
        printf("All elements successfully set to zero.\n");
    }

    // Clean up
    free(h_arr);
    CUDA_CHECK(cudaFree(d_arr));

    return EXIT_SUCCESS;
}
