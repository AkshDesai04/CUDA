/*
Aim: Write a kernel that takes two pointers, `int* in` and `int* out`. It should read the value from `in`, multiply it by 2, and write the result to `out`.

Thinking:
- The kernel will be very simple: each thread will process one element of the arrays.
- We compute a global index from blockIdx, blockDim, and threadIdx.
- If the index is within bounds, we read `in[idx]`, multiply by 2, and write to `out[idx]`.
- In the host code we allocate small test arrays, copy them to the device, launch the kernel, copy back, and print the results.
- We include basic error checking via a CUDA_CHECK macro to make debugging easier.
- The program is self‑contained in a single .cu file and can be compiled with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro for simple CUDA error checking
#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",            \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// Kernel that multiplies each input element by 2 and stores the result
__global__ void multiply_by_two(const int* in, int* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2;
    }
}

int main(void) {
    const size_t N = 10;
    int h_in[N];
    int h_out[N];

    // Initialize host input array
    for (size_t i = 0; i < N; ++i) {
        h_in[i] = static_cast<int>(i + 1); // values 1..10
    }

    // Allocate device memory
    int *d_in = nullptr;
    int *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, N * sizeof(int)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel: one block per 256 threads (common practice)
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    multiply_by_two<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify and print results
    printf("Input  : ");
    for (size_t i = 0; i < N; ++i) printf("%d ", h_in[i]);
    printf("\nOutput : ");
    for (size_t i = 0; i < N; ++i) printf("%d ", h_out[i]);
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
