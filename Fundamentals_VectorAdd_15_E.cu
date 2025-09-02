```cpp
/* 
Aim of the program: 
Fuse a multiply and an add: `C[i] = A[i] * scalar1; D[i] = B[i] + scalar2;`

Thinking behind this implementation:
1. The task is to perform two element-wise operations on large arrays using CUDA.
   - Multiply each element of array A by a scalar (scalar1) and store the result in array C.
   - Add a scalar (scalar2) to each element of array B and store the result in array D.
2. To maximize performance, both operations are fused into a single kernel launch.
   - This reduces kernel launch overhead and keeps data in registers or shared memory.
3. The host code will:
   - Allocate host memory for A, B, C, D.
   - Initialize A and B with deterministic values for easy verification.
   - Allocate corresponding device memory and copy A and B to the device.
   - Launch the kernel with enough threads to cover all N elements.
   - Copy the results back to the host and optionally verify correctness.
4. Error handling is done with a helper macro to keep the code clean.
5. Simple timing using CUDA events is included to give an idea of performance.
6. The code is self-contained, written in C++ (CUDA compatible), and compiled with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N (1 << 20)          // 1,048,576 elements
#define THREADS_PER_BLOCK 256

// Macro for checking CUDA errors following a CUDA API call
#define CUDA_CHECK(call)                                                        \
    {                                                                           \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    }

// Kernel that fuses multiplication and addition into a single pass
__global__ void fuseKernel(const float *A, const float *B,
                           float *C, float *D,
                           float scalar1, float scalar2,
                           int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] * scalar1;
        D[idx] = B[idx] + scalar2;
    }
}

int main(void)
{
    // Host pointers
    float *h_A = NULL;
    float *h_B = NULL;
    float *h_C = NULL;
    float *h_D = NULL;

    // Device pointers
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    float *d_D = NULL;

    size_t bytes = N * sizeof(float);

    // Allocate host memory
    h_A = (float *)malloc(bytes);
    h_B = (float *)malloc(bytes);
    h_C = (float *)malloc(bytes);
    h_D = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C || !h_D) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;          // 0, 1, 2, ...
        h_B[i] = (float)(N - i);    // N, N-1, N-2, ...
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_D, bytes));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Scalars
    float scalar1 = 2.0f;
    float scalar2 = 3.0f;

    // Determine grid dimensions
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch kernel
    fuseKernel<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, d_D, scalar1, scalar2, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_D, d_D, bytes, cudaMemcpyDeviceToHost));

    // Calculate elapsed time
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Kernel execution time: %f ms\n", ms);

    // Optional verification: print first 5 elements
    printf("\nFirst 5 elements after fused operation:\n");
    for (int i = 0; i < 5; ++i) {
        printf("Index %d: C = %f (expected %f), D = %f (expected %f)\n",
               i, h_C[i], h_A[i] * scalar1,
               h_D[i], h_B[i] + scalar2);
    }

    // Clean up
    free(h_A); free(h_B); free(h_C); free(h_D);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_D));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return EXIT_SUCCESS;
}
```