```cuda
/* 
Aim: Compare the performance of the fused kernel (`C=A+B`, `D=A-B`) against two separate kernel launches (one for addition, one for subtraction). The fused kernel should be faster.

Thinking:
1. We will allocate large arrays (e.g., N = 1<<24) of floats on the host, initialize them with some deterministic values (e.g., A[i] = i, B[i] = 2*i).
2. We copy these arrays to device memory.
3. We launch a fused kernel that computes both C = A + B and D = A - B in one pass.
4. We record the execution time using CUDA events.
5. We then launch two separate kernels: one for addition and one for subtraction, each with the same launch configuration, and time them together.
6. After each kernel set, we copy back the results to host and optionally verify correctness.
7. Finally, we print the timings for comparison.
8. We also handle CUDA errors for robustness.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N (1 << 24)          // Size of arrays (16 million elements)
#define BLOCK_SIZE 256       // Threads per block

__global__ void fusedKernel(const float *A, const float *B, float *C, float *D, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float a = A[idx];
        float b = B[idx];
        C[idx] = a + b;
        D[idx] = a - b;
    }
}

__global__ void addKernel(const float *A, const float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void subKernel(const float *A, const float *B, float *D, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        D[idx] = A[idx] - B[idx];
    }
}

void checkCudaError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error after %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    // Host memory allocation
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    float *h_C = (float*)malloc(N * sizeof(float));
    float *h_D = (float*)malloc(N * sizeof(float));
    float *h_C_ref = (float*)malloc(N * sizeof(float));
    float *h_D_ref = (float*)malloc(N * sizeof(float));

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(2 * i);
    }

    // Device memory allocation
    float *d_A, *d_B, *d_C, *d_D;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));
    cudaMalloc((void**)&d_D, N * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError("cudaMemcpy to device");

    // Determine grid dimensions
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ------------------- Fused kernel -------------------
    cudaEventRecord(start);
    fusedKernel<<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, d_C, d_D, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    checkCudaError("fusedKernel");

    float timeFused;
    cudaEventElapsedTime(&timeFused, start, stop);

    // Copy results back
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_D, d_D, N * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError("cudaMemcpy to host after fused");

    // ------------------- Separate kernels -------------------
    // Reset events
    cudaEventRecord(start);
    addKernel<<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    subKernel<<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, d_D, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    checkCudaError("addKernel or subKernel");

    float timeSeparate;
    cudaEventElapsedTime(&timeSeparate, start, stop);

    // Copy results back
    cudaMemcpy(h_C_ref, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_D_ref, d_D, N * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError("cudaMemcpy to host after separate");

    // Verify correctness
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != h_C_ref[i] || h_D[i] != h_D_ref[i]) {
            correct = false;
            printf("Mismatch at index %d: fused (C=%f, D=%f) vs separate (C=%f, D=%f)\n",
                   i, h_C[i], h_D[i], h_C_ref[i], h_D_ref[i]);
            break;
        }
    }
    if (!correct) {
        printf("Result verification failed.\n");
    } else {
        printf("Result verification succeeded.\n");
    }

    // Print timings
    printf("Fused kernel time: %.3f ms\n", timeFused);
    printf("Separate kernels time: %.3f ms\n", timeSeparate);
    if (timeFused < timeSeparate) {
        printf("Fused kernel is faster.\n");
    } else {
        printf("Separate kernels are faster (unexpected).\n");
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    free(h_C_ref);
    free(h_D_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```