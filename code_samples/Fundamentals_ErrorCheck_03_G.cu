```cuda
/* 
Is `cudaGetLastError()` a good way to check for kernel launch errors? (Yes, it's often used for this).

Thinking:
To illustrate how `cudaGetLastError()` is used to detect kernel launch failures, this program performs a simple vector addition on the GPU. The kernel is launched with a fixed grid and block configuration. Immediately after the launch, we call `cudaGetLastError()` to capture any launch‑time errors such as mis‑configured launch parameters or resource constraints. We also call `cudaDeviceSynchronize()` and again check `cudaGetLastError()` to catch any runtime errors that occur during kernel execution (e.g., invalid memory accesses). The program prints any detected errors and exits. This demonstrates that `cudaGetLastError()` is a reliable and common way to check for kernel launch errors.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple vector addition kernel
__global__ void vecAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize input arrays
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy input data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Synchronize and check for runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Simple verification
    bool ok = true;
    for (int i = 0; i < N; ++i)
    {
        if (h_C[i] != h_A[i] + h_B[i])
        {
            ok = false;
            break;
        }
    }

    if (ok)
        printf("Vector addition succeeded.\n");
    else
        printf("Result verification failed.\n");

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return EXIT_SUCCESS;
}
```