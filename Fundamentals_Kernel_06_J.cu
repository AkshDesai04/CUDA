```cuda
/*
Add robust error checking around every CUDA API call (`cudaMalloc`, `cudaMemcpy`, `cudaFree`).

This program demonstrates how to add two vectors on the GPU while performing comprehensive error checking
for each CUDA API call. The key points covered in the comment and the code are:
- The aim of the program is exactly as stated in the prompt: to add robust error checking around `cudaMalloc`, `cudaMemcpy`, `cudaFree`.
- A macro `CUDA_CHECK` is defined to encapsulate the error checking logic. It checks the return status of any CUDA runtime API call, prints the error message along with file and line number, and exits the program if an error occurs.
- Every call to `cudaMalloc`, `cudaMemcpy`, `cudaFree`, and kernel launch (`cudaLaunchKernel` or `<<<>>>`) is wrapped by `CUDA_CHECK`. For kernel launches, we also call `cudaGetLastError()` and `cudaDeviceSynchronize()` to catch launch failures.
- The kernel itself (`vectorAdd`) performs element-wise addition of two input arrays.
- In `main()`, we allocate host memory, initialize it, allocate device memory, copy data to device, launch the kernel, copy results back, and free all allocated memory.
- The program prints a success message if all operations complete without errors.
- This structure ensures that any failure in memory allocation, data transfer, or kernel execution is immediately reported and the program terminates, preventing silent failures.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro to check CUDA runtime API errors
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify results (optional)
    bool success = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: host %f, device %f\n", i, expected, h_C[i]);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Vector addition succeeded.\n");
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```