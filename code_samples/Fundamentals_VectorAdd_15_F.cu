```cuda
/* 
Aim of the program: 
Write a kernel that computes C[i] = A[i] + B[i] and writes the result to C, and also writes A[i] to another buffer D if A[i] is greater than some threshold. 

Thinking: 
The requirement is a simple element‑wise vector addition that also conditionally copies values from array A to array D when a threshold condition is satisfied. We can achieve this with a single CUDA kernel that takes device pointers to A, B, C, and D, as well as the array length and a threshold value. Each thread processes one element: 
  1. Compute the index from blockIdx, blockDim, and threadIdx. 
  2. Check bounds (index < N). 
  3. Read A[i] and B[i]. 
  4. Write C[i] = A[i] + B[i]. 
  5. If A[i] > threshold, write D[i] = A[i]; otherwise, optionally leave D[i] unchanged or set to 0. 
For simplicity we’ll set D[i] = A[i] only when the condition holds; otherwise, we’ll set D[i] to 0. 

The host code will:
  - Allocate host arrays A, B, C, D. 
  - Initialize A and B with sample data. 
  - Allocate device memory for A, B, C, D. 
  - Copy A and B to device. 
  - Define a threshold (e.g., 5.0). 
  - Launch the kernel with a grid size large enough to cover all elements. 
  - Copy back C and D from device to host. 
  - Print a few results to verify correctness. 
All operations include basic error checking. The program is self‑contained and compiles with nvcc as a .cu file. 
*/ 

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",              \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Kernel that adds A and B into C, and conditionally copies A to D
__global__ void add_and_conditional_copy(const float *A, const float *B, float *C, float *D, float threshold, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float a = A[idx];
    float b = B[idx];
    C[idx] = a + b;
    if (a > threshold) {
        D[idx] = a;
    } else {
        D[idx] = 0.0f; // optional: set to zero if condition not met
    }
}

int main(void)
{
    const size_t N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    float *h_D = (float *)malloc(size);

    // Initialize input arrays
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i % 10);   // values 0-9
        h_B[i] = static_cast<float>((i + 5) % 10); // shift
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_D;
    CHECK_CUDA(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size));
    CHECK_CUDA(cudaMalloc((void **)&d_D, size));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Define threshold
    float threshold = 5.0f;

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add_and_conditional_copy<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_D, threshold, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_D, d_D, size, cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("Index\tA\tB\tC=A+B\tD (A if A>%.1f else 0)\n", threshold);
    for (size_t i = 0; i < 10; ++i) {
        printf("%zu\t%.1f\t%.1f\t%.1f\t%.1f\n", i, h_A[i], h_B[i], h_C[i], h_D[i]);
    }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_D));

    return 0;
}
```