/*
Aim of the program: Write a kernel to achieve the same result: `kernel_set_zero(float* arr, int N)` where each thread sets one element to `0.0f`.

Thinking process:
1. The kernel must be defined with __global__ and take a pointer to float and an integer size.
2. Each thread calculates a global index using blockIdx, blockDim, and threadIdx.
3. If the index is within bounds (idx < N), it writes 0.0f to arr[idx].
4. In host code, allocate device memory, optionally initialize it with some non-zero values for demonstration.
5. Choose a reasonable block size (e.g., 256 threads per block) and compute the number of blocks needed to cover N elements.
6. Launch the kernel, then copy back the result to host memory to verify that all elements are zero.
7. Include simple error checking for CUDA API calls to make debugging easier.
8. Free device and host memory before exiting.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that sets each element of the array to 0.0f
__global__ void kernel_set_zero(float* arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        arr[idx] = 0.0f;
    }
}

// Helper macro for checking CUDA errors
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void) {
    const int N = 1024;          // Number of elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float* h_arr = (float*)malloc(size);
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host array with non-zero values
    for (int i = 0; i < N; ++i) {
        h_arr[i] = (float)i + 1.0f;
    }

    // Allocate device memory
    float* d_arr;
    CUDA_CHECK(cudaMalloc((void**)&d_arr, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice));

    // Define block and grid sizes
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    kernel_set_zero<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost));

    // Verify that all elements are zero
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_arr[i] != 0.0f) {
            fprintf(stderr, "Element %d not zero: %f\n", i, h_arr[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("All elements set to zero successfully.\n");
    } else {
        printf("Verification failed.\n");
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_arr));
    free(h_arr);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
