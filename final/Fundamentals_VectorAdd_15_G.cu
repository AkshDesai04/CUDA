```c
/*
Aim of the program:
Fuse a copy and a scale operation: `B[i] = A[i] * scalar`.

Thinking:
The goal is to demonstrate how a simple vector scaling operation can be fused into a single CUDA kernel, avoiding intermediate memory copies. In this program, the host initializes an input array A and a scalar value. The array A is copied to device memory. A kernel then performs the fused operation: each element of B is computed as A[i] multiplied by the scalar. Finally, the result is copied back to the host for verification. 
We use a simple error checking macro to catch CUDA API failures, and we measure the kernel execution time using cudaEvent_t. The code is self-contained, includes all necessary headers, and can be compiled with nvcc as a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                      \
    } while (0)

__global__ void scaleKernel(const float *A, float *B, float scalar, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        B[idx] = A[idx] * scalar;
    }
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    const float scalar = 2.5f;

    // Host memory allocation
    float *h_A = (float *)malloc(N * sizeof(float));
    float *h_B = (float *)malloc(N * sizeof(float));
    if (!h_A || !h_B)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input data
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = (float)i;
    }

    // Device memory allocation
    float *d_A = NULL;
    float *d_B = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_A, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_B, N * sizeof(float)));

    // Copy input data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Timing events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    scaleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, scalar, N);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_B, d_B, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify a few elements
    int errors = 0;
    for (int i = 0; i < 10; ++i)
    {
        float expected = h_A[i] * scalar;
        if (fabs(h_B[i] - expected) > 1e-5f)
        {
            fprintf(stderr, "Mismatch at index %d: expected %f, got %f\n",
                    i, expected, h_B[i]);
            errors++;
        }
    }
    if (errors == 0)
    {
        printf("Verification passed.\n");
    }

    printf("Kernel execution time: %f ms\n", milliseconds);

    // Cleanup
    free(h_A);
    free(h_B);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return EXIT_SUCCESS;
}
```