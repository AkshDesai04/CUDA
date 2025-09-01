```cuda
/*
 * Aim of the program:
 * Implement `D[i] = (A[i] * s1) + (B[i] * s2)`.
 *
 * Thinking:
 * 1. The task is to perform a simple element-wise linear combination of two input arrays
 *    A and B using two scalar multipliers s1 and s2. The result is stored in array D.
 *
 * 2. Since the operation is embarrassingly parallel (each element can be computed independently),
 *    it is a perfect fit for CUDA. We will write a kernel that each thread processes one
 *    element of the arrays. A simple grid-stride loop is used to handle arrays larger than
 *    the total number of threads.
 *
 * 3. On the host side we will:
 *    - Define a small example dataset for demonstration purposes.
 *    - Allocate device memory for A, B, and D.
 *    - Copy A and B from host to device.
 *    - Launch the kernel with an appropriate number of blocks and threads per block.
 *    - Copy the result D back to the host.
 *    - Print the first few elements to verify correctness.
 *
 * 4. Error checking will be added after CUDA API calls for robustness.
 *
 * 5. The code uses float arrays but can be adapted to double or int easily by changing types
 *    and kernel signature. The number of elements N can be modified to test performance or
 *    to work with larger datasets. The kernel uses a grid‑stride loop so it works for any
 *    N, not just multiples of block size.
 *
 * 6. The program is written as a single .cu file and can be compiled with:
 *      nvcc -o elementwise elementwise.cu
 *
 * 7. No external libraries are required.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel to compute D[i] = (A[i] * s1) + (B[i] * s2)
__global__ void compute(float *D, const float *A, const float *B, float s1, float s2, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Grid-stride loop to cover all elements
    for (int i = idx; i < N; i += blockDim.x * gridDim.x)
    {
        D[i] = (A[i] * s1) + (B[i] * s2);
    }
}

// Utility macro for checking CUDA errors
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void)
{
    const int N = 1 << 20;      // 1,048,576 elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_D = (float*)malloc(bytes);

    // Initialize input arrays with some values
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = (float)i * 0.1f;          // arbitrary values
        h_B[i] = (float)(N - i) * 0.2f;
    }

    // Scalars
    const float s1 = 2.5f;
    const float s2 = -1.5f;

    // Allocate device memory
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_D = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_D, bytes));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    compute<<<blocksPerGrid, threadsPerBlock>>>(d_D, d_A, d_B, s1, s2, N);
    CUDA_CHECK(cudaGetLastError());  // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_D, d_D, bytes, cudaMemcpyDeviceToHost));

    // Verify by printing first 10 results
    printf("First 10 results of D[i] = (A[i] * %.2f) + (B[i] * %.2f):\n", s1, s2);
    for (int i = 0; i < 10; ++i)
    {
        printf("D[%d] = %f\n", i, h_D[i]);
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_D);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_D));

    return 0;
}
```