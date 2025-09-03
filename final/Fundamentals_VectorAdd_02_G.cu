/*
Aim of the program: Implement vector division: C[i] = A[i] / B[i]. Add a check in the kernel to prevent division by zero.

Thinking process:
1. I need a complete CUDA C (.cu) file that demonstrates vector division while guarding against division by zero.
2. The program will allocate device memory for three vectors A, B, and C of type float, copy data from host to device, run a kernel to compute C[i] = A[i] / B[i] only if B[i] is not zero, otherwise set C[i] to zero.
3. I will provide a small example with a fixed vector size (e.g., 256 elements) and initialize the vectors on the host. 
4. After kernel execution, the result will be copied back to the host and printed for verification.
5. I’ll include basic error checking after CUDA API calls for robustness.
6. The kernel will use the standard block/thread indexing pattern. Each thread computes a single element; we’ll guard the division to avoid crashes from division by zero.
7. The program will be self-contained: includes, macros, main function, kernel, and memory cleanup.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

// Kernel for vector division with zero check
__global__ void vecDivKernel(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (B[idx] != 0.0f) {
            C[idx] = A[idx] / B[idx];
        } else {
            // Prevent division by zero: set result to 0
            C[idx] = 0.0f;
        }
    }
}

int main(void)
{
    const int N = 256;
    const int bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)(i + 1);               // 1, 2, 3, ...
        h_B[i] = (float)(i % 5);               // 0,1,2,3,4 repeating
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecDivKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Print a few results to verify
    printf("Index\tA\tB\tC (A/B)\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d\t%.1f\t%.1f\t%.3f\n", i, h_A[i], h_B[i], h_C[i]);
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
