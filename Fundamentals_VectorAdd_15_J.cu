```cpp
/*
Aim of the program: 
A kernel that computes `C[i] = a*A[i] + b*B[i]` is a fused operation.

Thought process and design considerations:
- The goal is to demonstrate a simple CUDA kernel that performs a fused multiply-add operation across
  two input arrays A and B, scaling them by scalar coefficients a and b, and storing the result in C.
- I chose a straightforward example with float data type, but the kernel could be templated for other
  numeric types if needed.
- The program will:
  1. Allocate host arrays A, B, C (and a reference array C_ref for verification).
  2. Initialize A and B with some deterministic values so that the result can be verified.
  3. Allocate device memory for the arrays.
  4. Copy A and B from host to device.
  5. Launch the kernel with a sufficient number of threads/blocks to cover all elements.
  6. Copy the computed C back to the host.
  7. Compute a reference result on the CPU and compare with the GPU result to validate correctness.
  8. Print a success message or highlight mismatches.
- Error checking is performed after each CUDA API call and kernel launch to catch issues early.
- The kernel itself is a simple one-liner that performs the fused operation. No shared memory or
  advanced optimizations are needed for this demonstration.
- To keep the code self-contained and compile-ready, I include all necessary headers and a minimal
  Makefile-like comment at the top that reminds how to compile: `nvcc -arch=sm_52 fused.cu -o fused`
- The array size is set to 1024 elements, which is small enough for demonstration but large enough to
  illustrate the use of GPU parallelism. The block size is chosen as 256 threads, which is a common
  choice for many GPUs.
- Since the focus is on the fused operation, I avoid any additional device-side work or shared
  memory usage; this keeps the kernel straightforward and educational.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 1024          // Number of elements
#define BLOCK_SIZE 256  // Threads per block

// Error checking macro
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Kernel that performs C[i] = a*A[i] + b*B[i]
__global__ void fused_add_mul(const float *A, const float *B, float *C, float a, float b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = a * A[idx] + b * B[idx];
    }
}

int main(void) {
    // Host memory allocation
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    float *h_C = (float*)malloc(N * sizeof(float));
    float *h_C_ref = (float*)malloc(N * sizeof(float));

    if (!h_A || !h_B || !h_C || !h_C_ref) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i) * 0.5f;          // Example values
        h_B[i] = static_cast<float>(i) * 0.25f;
    }

    float a = 2.0f;
    float b = 3.0f;

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, N * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fused_add_mul<<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, d_C, a, b, N);
    CUDA_CHECK(cudaGetLastError()); // Check kernel launch

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute reference result on CPU
    for (int i = 0; i < N; ++i) {
        h_C_ref[i] = a * h_A[i] + b * h_B[i];
    }

    // Verify results
    bool success = true;
    for (int i = 0; i < N; ++i) {
        float diff = fabs(h_C[i] - h_C_ref[i]);
        if (diff > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: GPU %f, CPU %f, diff %f\n",
                    i, h_C[i], h_C_ref[i], diff);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Success: GPU results match CPU reference.\n");
    } else {
        printf("Failure: GPU results do not match CPU reference.\n");
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
```