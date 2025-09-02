```cuda
/* 
Aim: Implement C[i] = A[i] * B[i] * D[i] (three-vector multiplication).

Thinking: 
To perform an element‑wise product of three input vectors A, B, and D on the GPU, we need to:

1. Allocate host memory for the vectors and initialise them with test data.
2. Allocate device memory for the same vectors and the output vector C.
3. Copy the input vectors from host to device.
4. Define a CUDA kernel that computes C[idx] = A[idx] * B[idx] * D[idx] for each index.
5. Launch the kernel with an appropriate grid and block configuration.
6. Synchronise, copy the result back to the host, and optionally validate it.
7. Clean up all allocated resources.

The kernel uses a simple one‑dimensional thread mapping; each thread handles a single index if that index is within bounds. Error checking is performed after each CUDA API call to make debugging easier.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Kernel to compute element-wise product of three vectors
__global__ void multiplyThreeVectors(const float *A, const float *B, const float *D, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] * B[idx] * D[idx];
    }
}

int main(void) {
    const int N = 1 << 20;            // Number of elements (about 1 million)
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_D = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    if (!h_A || !h_B || !h_D || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host arrays with sample data
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i) * 0.001f;
        h_B[i] = static_cast<float>(i) * 0.002f;
        h_D[i] = static_cast<float>(i) * 0.003f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_D, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_D, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

    // Copy inputs from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_D, h_D, bytes, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    multiplyThreeVectors<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_D, d_C, N);
    CHECK_CUDA(cudaGetLastError());               // Check for kernel launch errors
    CHECK_CUDA(cudaDeviceSynchronize());          // Ensure kernel has finished

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Optional: Verify a few results
    for (int i = 0; i < 5; ++i) {
        float expected = h_A[i] * h_B[i] * h_D[i];
        printf("C[%d] = %f (expected %f)\n", i, h_C[i], expected);
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_D);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_D));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
```