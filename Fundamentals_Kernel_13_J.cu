/*
Chain kernel calls. The output of the first kernel (`b`) becomes the input to a second kernel that, for example, squares every element.

Thinking:
- The program demonstrates how to launch two kernels sequentially in CUDA.
- The first kernel will perform a simple operation on an input array `a`, e.g., add 1 to each element, producing array `b`.
- The second kernel will take array `b` as input and square each element, producing array `c`.
- Host code will allocate device memory for `a`, `b`, and `c`, initialize `a`, launch the first kernel, then launch the second kernel.
- After both kernels finish, the host will copy back `c` and print the results to verify correctness.
- Proper CUDA error checking macros are used for clarity.
- Simple block and grid dimensions are chosen for demonstration purposes.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                      \
    {                                                          \
        cudaError_t err = call;                                \
        if (err != cudaSuccess) {                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    }

__global__ void firstKernel(const float* a, float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        b[idx] = a[idx] + 1.0f;  // Example operation: add 1
    }
}

__global__ void secondKernel(const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = b[idx] * b[idx];  // Square each element
    }
}

int main(void) {
    const int N = 10;
    const int size = N * sizeof(float);

    // Host memory allocation
    float h_a[N], h_c[N];
    for (int i = 0; i < N; ++i) h_a[i] = (float)i;  // Initialize input array

    // Device memory allocation
    float *d_a = NULL, *d_b = NULL, *d_c = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_a, size));
    CHECK_CUDA(cudaMalloc((void**)&d_b, size));
    CHECK_CUDA(cudaMalloc((void**)&d_c, size));

    // Copy input from host to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch first kernel
    firstKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, N);
    CHECK_CUDA(cudaGetLastError());

    // Launch second kernel
    secondKernel<<<blocksPerGrid, threadsPerBlock>>>(d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Print results
    printf("Input a:\n");
    for (int i = 0; i < N; ++i) printf("%6.1f ", h_a[i]);
    printf("\n");

    printf("Output c (squared values of a+1):\n");
    for (int i = 0; i < N; ++i) printf("%6.1f ", h_c[i]);
    printf("\n");

    // Clean up
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return 0;
}
