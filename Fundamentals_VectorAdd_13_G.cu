/*
Implement `C[i] = (A[i] == B[i]) ? 1 : 0`.

My thoughts:
To solve this problem, I will write a CUDA kernel that iterates over the indices of the input arrays A and B. Each thread will compute its global index, check that it is within bounds, compare the elements A[idx] and B[idx], and write 1 to C[idx] if they are equal, otherwise 0.

The host code will:
1. Define a small example size (e.g., N = 10) for demonstration.
2. Allocate host arrays A, B, and C.
3. Fill A and B with sample data.
4. Allocate device memory for the three arrays.
5. Copy A and B from host to device.
6. Launch the kernel with an appropriate grid and block configuration.
7. Copy the result C back to the host.
8. Verify and print the result.
9. Free all allocated memory.

I will include basic error checking for CUDA API calls to make debugging easier. The program will be a single .cu file that can be compiled with nvcc. No external dependencies beyond the standard CUDA runtime are required. The code is self‑contained and demonstrates the requested element‑wise equality comparison using CUDA.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel to compare two integer arrays elementwise
__global__ void compareArrays(const int *A, const int *B, int *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = (A[idx] == B[idx]) ? 1 : 0;
    }
}

// Utility macro for CUDA error checking
#define CUDA_CHECK(call)                                            \
    {                                                               \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",        \
                    __FUNCTION__, __FILE__, __LINE__,               \
                    cudaGetErrorString(err));                       \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    }

int main(void)
{
    const int N = 10;                // Number of elements
    const int bytes = N * sizeof(int);

    // Allocate host memory
    int *h_A = (int *)malloc(bytes);
    int *h_B = (int *)malloc(bytes);
    int *h_C = (int *)malloc(bytes);

    // Initialize input arrays with sample data
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = i;                 // 0,1,2,...,9
        h_B[i] = (i % 2 == 0) ? i : i + 1; // even indices equal, odd indices differ
    }

    // Allocate device memory
    int *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Determine execution configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    compareArrays<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Print results
    printf("Index\tA\tB\tC\n");
    for (int i = 0; i < N; ++i)
    {
        printf("%d\t%d\t%d\t%d\n", i, h_A[i], h_B[i], h_C[i]);
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
