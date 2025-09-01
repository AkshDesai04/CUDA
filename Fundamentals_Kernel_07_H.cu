/*
Define the block dimensions using a `dim3` variable: `dim3 block_dim(8, 8);`

Thinking:
- The task is to write a CUDA C program that demonstrates defining block dimensions with a dim3 variable. 
- I will create a simple vector addition kernel that uses the defined block dimensions.
- The host code will allocate device memory, copy data, launch the kernel with the specified block dimensions, and copy results back.
- Grid dimensions will be calculated to cover the entire vector size.
- I will include error checking for clarity.
- The program will be a complete .cu file with a main function, kernel, and necessary includes.
- The output is only the .cu file code, no additional text.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Simple kernel that adds two vectors
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Define block dimensions
    dim3 block_dim(8, 8); // 64 threads per block

    // Compute number of blocks needed
    int threadsPerBlock = block_dim.x * block_dim.y;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    dim3 grid_dim(blocksPerGrid, 1);

    // Launch kernel
    vectorAdd<<<grid_dim, block_dim>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Simple validation
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            ++errors;
            if (errors < 10) {
                printf("Mismatch at index %d: %f + %f != %f\n",
                       i, h_A[i], h_B[i], h_C[i]);
            }
        }
    }
    if (errors == 0) {
        printf("All results are correct.\n");
    } else {
        printf("Total mismatches: %d\n", errors);
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
