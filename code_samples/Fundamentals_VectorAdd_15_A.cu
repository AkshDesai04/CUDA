/*
 * Aim: Implement a fused kernel that computes SAXPY and also the dot product of two vectors (using atomic adds for the reduction part).
 *
 * Thinking:
 * - The program allocates two float vectors `x` and `y` on the host, copies them to the device, and launches a fused CUDA kernel.
 * - Each thread processes one element: it performs the SAXPY operation `y[i] = a * x[i] + y[i]` and then computes the product `x[i] * y[i]` using the updated `y[i]`.
 * - The dot product accumulation is performed by atomically adding each partial product to a device scalar `dot`.
 * - After kernel execution, the host copies back the updated `y` and the computed dot product for verification.
 * - The dot product is expected to be `a * sum(x_i^2) + sum(x_i * y_initial_i)` if `y` was initially `y_initial`. For the simple test case below where `x[i] = 1` and `y[i] = 2`, `a = 3`, the updated `y[i]` becomes `5` and the dot product should be `5 * N`.
 * - Atomic adds on floating point are supported on modern GPUs (compute capability 2.0+). They are relatively slow but acceptable for demonstration purposes.
 * - Error checking is included for CUDA API calls. The kernel uses a simple linear indexing scheme.
 * - The program prints the computed dot product and verifies it against the expected value.
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N (1 << 20)          // Number of elements in vectors
#define BLOCK_SIZE 256       // Threads per block

// Fused kernel: SAXPY + dot product accumulation
__global__ void saxpy_dot_kernel(const float *x, float *y, float a, float *dot, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // SAXPY: y = a * x + y
        y[idx] = a * x[idx] + y[idx];
        // Compute product for dot product
        float prod = x[idx] * y[idx];
        // Atomic addition to accumulate dot product
        atomicAdd(dot, prod);
    }
}

int main(void)
{
    float *h_x = NULL, *h_y = NULL;
    float *d_x = NULL, *d_y = NULL;
    float *d_dot = NULL;
    float h_dot = 0.0f;
    const float a = 3.0f;

    // Allocate host memory
    h_x = (float*)malloc(N * sizeof(float));
    h_y = (float*)malloc(N * sizeof(float));
    if (h_x == NULL || h_y == NULL) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return -1;
    }

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_x[i] = 1.0f;  // example value
        h_y[i] = 2.0f;  // example value
    }

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_x, N * sizeof(float));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_x failed: %s\n", cudaGetErrorString(err)); return -1; }
    err = cudaMalloc((void**)&d_y, N * sizeof(float));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_y failed: %s\n", cudaGetErrorString(err)); return -1; }
    err = cudaMalloc((void**)&d_dot, sizeof(float));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_dot failed: %s\n", cudaGetErrorString(err)); return -1; }

    // Copy data from host to device
    err = cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy d_x failed: %s\n", cudaGetErrorString(err)); return -1; }
    err = cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy d_y failed: %s\n", cudaGetErrorString(err)); return -1; }

    // Initialize dot product accumulator on device to 0
    err = cudaMemset(d_dot, 0, sizeof(float));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemset d_dot failed: %s\n", cudaGetErrorString(err)); return -1; }

    // Launch kernel
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    saxpy_dot_kernel<<<numBlocks, BLOCK_SIZE>>>(d_x, d_y, a, d_dot, N);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Synchronize to ensure kernel completion
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy results back to host
    err = cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy h_y failed: %s\n", cudaGetErrorString(err)); return -1; }
    err = cudaMemcpy(&h_dot, d_dot, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy h_dot failed: %s\n", cudaGetErrorString(err)); return -1; }

    // Compute expected dot product: after SAXPY, y[i] = a*1 + 2 = 5
    float expected_dot = 5.0f * (float)N;
    printf("Computed dot product: %f\n", h_dot);
    printf("Expected dot product: %f\n", expected_dot);
    printf("Difference: %f\n", fabsf(h_dot - expected_dot));

    // Clean up
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_dot);
    free(h_x);
    free(h_y);

    return 0;
}
