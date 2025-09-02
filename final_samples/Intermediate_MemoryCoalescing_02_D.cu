```cpp
/* 
This pattern ensures both global reads and global writes are fully coalesced.

Thinking process:
- The goal is to create a simple CUDA program that demonstrates coalesced global memory accesses.
- Coalesced accesses happen when threads in a warp access consecutive memory addresses. 
  By using a linear index per thread and operating on contiguous arrays, each warp will
  read and write contiguous elements.
- A vector addition kernel is a classic example: each thread reads A[i] and B[i] (global reads)
  and writes C[i] (global write). With proper alignment and contiguous memory, the accesses
  are fully coalesced.
- We'll allocate three arrays (A, B, C) on the host, copy A and B to the device,
  launch the kernel, copy C back, and verify the result.
- We add simple error checking via a macro to keep the code clean.
- The code uses the CUDA runtime API and is written in standard C++ (CUDA C).
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Kernel: Element-wise addition of two vectors
__global__ void vectorAdd(const float *A, const float *B, float *C, size_t N) {
    // Compute global linear index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we do not read/write out of bounds
    if (idx < N) {
        // Coalesced global reads from A[idx] and B[idx]
        float a_val = A[idx];
        float b_val = B[idx];

        // Coalesced global write to C[idx]
        C[idx] = a_val + b_val;
    }
}

int main() {
    const size_t N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Host allocations
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize host arrays
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Device allocations
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify result
    bool success = true;
    for (size_t i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %zu: got %f, expected %f\n",
                    i, h_C[i], expected);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Vector addition succeeded! All results are correct.\n");
    } else {
        printf("Vector addition failed.\n");
    }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
```