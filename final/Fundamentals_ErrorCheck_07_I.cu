/*
Why does CUDA use an asynchronous model? (Performance, allows CPU and GPU to work in parallel).

Thoughts on the program:
- The aim is to illustrate CUDA's asynchronous execution model in a very simple way.
- By launching a kernel and then immediately performing another CPU operation (e.g., printing a message),
  we can observe that the CPU does not wait for the GPU kernel to finish before continuing.
- The kernel itself does a trivial vector addition to keep the example short and self‑contained.
- After the kernel launch, we use cudaDeviceSynchronize() to ensure all GPU work is finished before we read back
  the result and exit, making the program correct while still demonstrating asynchrony.
- The code uses only standard CUDA C/C++ constructs, making it easy to compile with nvcc.
- Comments are added to explain each step, particularly where asynchrony is relevant.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 256

// Simple kernel that adds two vectors
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main(void) {
    size_t size = N * sizeof(float);

    // Host allocations
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize input vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Device allocations
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy inputs to device (asynchronous by default)
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel asynchronously
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Immediately after kernel launch, CPU can perform other work
    printf("Kernel launched asynchronously. CPU can continue execution.\n");

    // Example of other CPU work: simple loop
    int sum = 0;
    for (int i = 0; i < 1000000; ++i) sum += i;
    printf("Dummy CPU work completed. Sum = %d\n", sum);

    // Synchronize to make sure kernel is finished before we copy results back
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify a few results
    printf("Verification of first 5 results:\n");
    for (int i = 0; i < 5; ++i) {
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
