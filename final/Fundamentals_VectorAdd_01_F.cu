```cuda
/*
Implement element-wise vector multiplication: C[i] = A[i] * B[i].

Goal:
The goal of this program is to demonstrate a simple element-wise multiplication
of two vectors using CUDA.  We allocate two input vectors A and B on the host,
copy them to the device, launch a kernel that computes C[i] = A[i] * B[i] in
parallel, and then copy the result back to the host for verification.

Key design choices:
1. Use float vectors for simplicity; the element type can be changed easily.
2. Size of the vectors is defined by a macro (N).  Here we choose 1<<20 (1 million)
   elements, which is large enough to benefit from GPU parallelism but still
   small enough to run quickly on most GPUs.
3. Kernel launch configuration:
   - Block size (threads per block) is set to 256, a common choice for
     maximizing occupancy on modern GPUs.
   - Grid size is calculated as (N + blockSize - 1) / blockSize to cover all
     elements.
4. Memory management:
   - Host memory is allocated with new[] and freed with delete[].
   - Device memory is allocated with cudaMalloc and freed with cudaFree.
   - All cuda* calls are wrapped in a CUDA_CHECK macro that prints errors and
     aborts on failure.
5. Validation:
   After copying the result back, we print the first ten elements of A, B,
   and C to verify correctness.  For production code one would compare all
   elements or use a checksum.

Edge cases handled:
- The kernel checks if the thread index is within bounds (idx < n) to avoid
  out-of-range accesses when N is not an exact multiple of the block size.
- The program checks for CUDA errors after each API call.

This self-contained .cu file can be compiled with:
    nvcc -o vecmul vecmul.cu
and executed:
    ./vecmul
*/

#include <iostream>
#include <cstdlib>

#define N (1 << 20)          // Number of elements (1,048,576)
#define BLOCK_SIZE 256       // Threads per block

// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " - " << cudaGetErrorString(err) << std::endl;        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Kernel for element-wise multiplication
__global__ void vecMul(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] * B[idx];
    }
}

int main() {
    // Host memory allocation
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i) * 0.001f;      // e.g., 0.0, 0.001, 0.002, ...
        h_B[i] = static_cast<float>(N - i) * 0.002f;  // e.g., 2097.998, 2097.996, ...
    }

    // Device memory allocation
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, N * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vecMul<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());          // Check for kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize());     // Wait for kernel to finish

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Simple verification: print first 10 elements
    std::cout << "Index\tA\tB\tC=A*B\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << i << "\t"
                  << h_A[i] << "\t"
                  << h_B[i] << "\t"
                  << h_C[i] << "\n";
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    std::cout << "Vector multiplication completed successfully.\n";
    return 0;
}
```