```cu
/*
Implement the canonical first parallel program: element-wise vector addition. Create two input vectors (A and B) and one output vector (C) of size 256. Initialize A and B on the host. Launch a kernel with one block of 256 threads. Each thread i should compute C[i] = A[i] + B[i].

Thinking process:
1. Use a single block of 256 threads, which matches the array size exactly. This simplifies indexing: each thread can use its blockIdx.x * blockDim.x + threadIdx.x to compute the global index.
2. The kernel will be very simple: read A[idx] and B[idx] and write to C[idx].
3. Host side: allocate three arrays of size 256, initialize A and B with some values (e.g., A[i] = i, B[i] = 2*i).
4. Allocate device memory for A, B, C using cudaMalloc. Copy host A and B to device using cudaMemcpy.
5. Launch kernel with <<<1, 256>>>.
6. After kernel completion, copy result C back to host using cudaMemcpy.
7. Optionally print first few results to verify correctness.
8. Add basic error checking for CUDA calls (cudaGetLastError, cudaMemcpy, etc.) for robustness.
9. Free device memory and host arrays before exiting.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 256

__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    float *h_A = (float *)malloc(N * sizeof(float));
    float *h_B = (float *)malloc(N * sizeof(float));
    float *h_C = (float *)malloc(N * sizeof(float));

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(2 * i);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, N * sizeof(float));
    cudaMalloc((void **)&d_B, N * sizeof(float));
    cudaMalloc((void **)&d_C, N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel: one block of 256 threads
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(1);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy result back to host
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Optional: print first 10 results
    printf("First 10 results of A + B:\n");
    for (int i = 0; i < 10; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```