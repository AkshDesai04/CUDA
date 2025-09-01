/*
Aim: Wrap all three `cudaMemcpy` calls (HtoD for A, HtoD for B, DtoH for C).

Thought process:
1. The goal is to illustrate how to perform memory copies between host and device in a CUDA program.
2. We'll implement a simple element-wise vector addition kernel.
3. We'll allocate three arrays on the host: A, B, and C.
4. We'll allocate corresponding device arrays: d_A, d_B, d_C.
5. We'll copy A and B from host to device (HtoD) using cudaMemcpy.
6. We'll launch the kernel to compute C = A + B on the device.
7. We'll copy the result C back from device to host (DtoH) using cudaMemcpy.
8. For robustness, each cudaMemcpy call will be wrapped in a helper function that checks for errors and prints diagnostic information if needed.
9. The wrapper function `checkCudaMemcpy` will accept the source pointer, destination pointer, size, and the kind of copy (cudaMemcpyKind) and then perform the copy, checking the return status.
10. The program will clean up all allocated memory and exit cleanly.

The code below follows this plan, providing a complete .cu file with the required functionality.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel for element-wise vector addition: C = A + B
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Wrapper function to perform cudaMemcpy with error checking
void checkCudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) {
    cudaError_t err = cudaMemcpy(dst, src, count, kind);
    if (err != cudaSuccess) {
        const char *kindStr;
        switch (kind) {
            case cudaMemcpyHostToDevice: kindStr = "HtoD"; break;
            case cudaMemcpyDeviceToHost: kindStr = "DtoH"; break;
            case cudaMemcpyDeviceToDevice: kindStr = "DtoD"; break;
            case cudaMemcpyHostToHost: kindStr = "HtoH"; break;
            default: kindStr = "Unknown";
        }
        fprintf(stderr, "Error during cudaMemcpy (%s): %s\n", kindStr, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void) {
    const int N = 1 << 20;           // Number of elements (e.g., 1M)
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
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
    cudaError_t err;
    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_A failed: %s\n", cudaGetErrorString(err)); return EXIT_FAILURE; }
    err = cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_B failed: %s\n", cudaGetErrorString(err)); return EXIT_FAILURE; }
    err = cudaMalloc((void **)&d_C, size);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_C failed: %s\n", cudaGetErrorString(err)); return EXIT_FAILURE; }

    // Copy host arrays to device (HtoD) using wrapper
    checkCudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    checkCudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Copy result back to host (DtoH) using wrapper
    checkCudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Optional: Verify result
    bool success = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (h_C[i] != expected) {
            fprintf(stderr, "Verification failed at index %d: %f != %f\n", i, h_C[i], expected);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Vector addition successful. First 5 results:\n");
        for (int i = 0; i < 5; ++i) {
            printf("C[%d] = %f\n", i, h_C[i]);
        }
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
