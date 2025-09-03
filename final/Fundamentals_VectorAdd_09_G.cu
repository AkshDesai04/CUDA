/*
Aim of the program:
Implement `D[i] = (A[i] * s1) + (B[i] * s2)`.

Thought process:
The task is a classic element‑wise vector operation that can be parallelised on the GPU.
I will create a CUDA kernel that receives pointers to device arrays A, B and D,
as well as the two scalars s1 and s2, and the number of elements N.
Each thread will compute one element of the result array: D[i] = A[i] * s1 + B[i] * s2.
To keep the example simple yet complete, the host code will:
  - Allocate and initialise host arrays with dummy data.
  - Allocate device memory, copy the data from host to device.
  - Launch the kernel with a sufficient number of blocks and threads.
  - Copy the result back to the host.
  - Verify a few elements to ensure correctness.
  - Clean up all allocated memory.
Error checking for CUDA API calls is included to aid debugging.
The program uses float precision and a fixed array size, but the kernel
is written generically to handle any N that fits in an integer index range.
The goal is to demonstrate the core idea: performing a linear combination
of two input vectors on the GPU efficiently.

*/

#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel that computes D[i] = A[i] * s1 + B[i] * s2
__global__ void vectorAddScale(const float *A, const float *B, float *D,
                               float s1, float s2, size_t N) {
    // Calculate global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        D[idx] = A[idx] * s1 + B[idx] * s2;
    }
}

// Utility macro for CUDA error checking
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                   \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

int main(void) {
    const size_t N = 1 << 20; // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_D = (float*)malloc(bytes);

    if (!h_A || !h_B || !h_D) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialise host arrays with some values
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = (float)i * 0.001f;        // Example pattern
        h_B[i] = (float)(N - i) * 0.002f;  // Example pattern
    }

    // Device pointers
    float *d_A = NULL, *d_B = NULL, *d_D = NULL;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_D, bytes));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Scalars
    float s1 = 2.5f;
    float s2 = -1.2f;

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vectorAddScale<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_D, s1, s2, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_D, d_D, bytes, cudaMemcpyDeviceToHost));

    // Verify a few results
    bool ok = true;
    for (size_t i = 0; i < 10; ++i) {
        float expected = h_A[i] * s1 + h_B[i] * s2;
        if (fabs(h_D[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %zu: GPU=%f, Expected=%f\n",
                    i, h_D[i], expected);
            ok = false;
            break;
        }
    }
    if (ok) {
        printf("Success! First 10 results verified.\n");
    } else {
        printf("Verification failed.\n");
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_D);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_D));

    return EXIT_SUCCESS;
}
