```cuda
/*
Aim of the program: Change the data type from `int` to `float` and initialize the host array with floating-point values.
Thinking: The original program was a simple CUDA vector addition using integer arrays. 
To meet the new requirement, the data type of all arrays (host and device) has been switched from int to float. 
The host arrays are initialized with explicit floating‑point literals (e.g., 1.0f, 2.0f, etc.). 
The kernel signature and the memory allocation calls have been updated accordingly. 
A helper macro is provided to check CUDA errors after API calls. 
The program demonstrates allocation, copying, kernel launch, and result verification for floating‑point vectors.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro to check for CUDA errors
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    // Size of vectors
    const int N = 10;
    const int SIZE = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(SIZE);
    float *h_B = (float *)malloc(SIZE);
    float *h_C = (float *)malloc(SIZE);

    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays with floating-point values
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i + 1.0f;          // 1.0f, 2.0f, ..., N
        h_B[i] = (float)(N - i);           // N, N-1, ..., 1
    }

    // Allocate device memory
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, SIZE));
    CUDA_CHECK(cudaMalloc((void **)&d_B, SIZE));
    CUDA_CHECK(cudaMalloc((void **)&d_C, SIZE));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, SIZE, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, SIZE, cudaMemcpyDeviceToHost));

    // Verify result
    printf("Result of vector addition (float values):\n");
    for (int i = 0; i < N; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return EXIT_SUCCESS;
}
```