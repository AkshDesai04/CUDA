```cpp
/*
Aim of the program:
Modify your large vector addition program to handle input sizes that are not an exact multiple of the block size (e.g., 1,000,000 elements). Add a boundary check inside the kernel (`if (index < N) { ... }`) to prevent threads from accessing memory out of bounds.

Thinking:
1. Need a CUDA kernel that performs element‑wise addition of two vectors.
2. The vector size N may not be a multiple of the chosen block size. Therefore each thread must check that its computed global index is less than N before accessing array elements.
3. Host code should:
   - Allocate host memory for vectors A, B, C.
   - Initialize A and B with some values (e.g., random or sequential).
   - Allocate device memory and copy A, B to device.
   - Determine block size (e.g., 256) and grid size as ceil(N / blockSize).
   - Launch the kernel with boundary check.
   - Copy result back to host and optionally verify correctness.
4. Include error checking for CUDA API calls and kernel launches.
5. Use standard CUDA headers and functions.
6. Ensure the code compiles with nvcc and runs correctly for arbitrary N, including 1,000,000.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <ctime>

// Macro to check CUDA errors
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",\
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Kernel for vector addition with boundary check
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Helper function to fill vector with random floats
void fillRandom(float *data, int N)
{
    for (int i = 0; i < N; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main()
{
    const int N = 1000000;            // Example size, not necessarily multiple of block size
    const int blockSize = 256;        // Typical block size
    const int gridSize = (N + blockSize - 1) / blockSize;

    // Allocate host memory
    float *h_A = (float *)malloc(N * sizeof(float));
    float *h_B = (float *)malloc(N * sizeof(float));
    float *h_C = (float *)malloc(N * sizeof(float));

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize input vectors
    srand(static_cast<unsigned>(time(nullptr)));
    fillRandom(h_A, N);
    fillRandom(h_B, N);

    // Allocate device memory
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_C, N * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Optional: verify correctness for a few elements
    bool correct = true;
    for (int i = 0; i < 10; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            fprintf(stderr, "Mismatch at index %d: host=%f, device=%f\n", i, expected, h_C[i]);
            correct = false;
            break;
        }
    }
    if (correct) {
        printf("Vector addition successful for %d elements.\n", N);
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
```