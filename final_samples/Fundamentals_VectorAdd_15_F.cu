```cuda
/*
Aim of the program: 
Write a kernel that computes `C[i] = A[i] + B[i]` and writes the result to `C`, 
and also writes `A[i]` to another buffer `D` if `A[i]` is greater than some threshold.

Thinking:
- We need a simple CUDA kernel that takes four integer pointers: A, B, C, and D.
- Each thread processes one index `i`. The kernel computes the sum of A[i] and B[i] and stores it in C[i].
- If A[i] exceeds a user‑supplied threshold, we copy A[i] to D[i]; otherwise we store a sentinel value (e.g., -1) to indicate that the condition was not met.
- The kernel bounds‑checks the index to avoid out‑of‑bounds memory accesses.
- In `main`, we allocate host arrays, initialize them with example data, allocate device memory, copy data to device, set a threshold, launch the kernel, copy results back, and print them for verification.
- Basic error checking is performed after each CUDA API call.
- The program uses integer types for simplicity, but the same pattern works for floats or other types.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Kernel that performs element‑wise addition and conditional copy
__global__ void addAndConditional(const int *A, const int *B, int *C, int *D, int N, int threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int sum = A[i] + B[i];
        C[i] = sum;
        if (A[i] > threshold) {
            D[i] = A[i];
        } else {
            D[i] = -1;   // sentinel value when condition is not met
        }
    }
}

int main(void)
{
    const int N = 10;
    const int threshold = 5;   // example threshold

    // Allocate host memory
    int h_A[N], h_B[N], h_C[N], h_D[N];

    // Initialize input arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = i;          // 0,1,2,...
        h_B[i] = 10 - i;     // 10,9,8,...
    }

    // Allocate device memory
    int *d_A, *d_B, *d_C, *d_D;
    CHECK_CUDA(cudaMalloc((void**)&d_A, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_B, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_C, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_D, N * sizeof(int)));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice));

    // Determine grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    addAndConditional<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_D, N, threshold);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_D, d_D, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    printf("Index |  A[i] |  B[i] |  C[i] = A+B |  D[i] (if A>threshold)\n");
    printf("---------------------------------------------------------------\n");
    for (int i = 0; i < N; ++i) {
        printf("%5d | %5d | %5d | %9d | %9d\n", i, h_A[i], h_B[i], h_C[i], h_D[i]);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_D));

    return 0;
}
```