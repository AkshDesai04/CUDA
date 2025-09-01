/*
Aim of the program: Add error checking to the `cudaMemset` call.

Thinking:
- The goal is to demonstrate how to add robust error handling to a CUDA API call, specifically `cudaMemset`. 
- We will allocate device memory using `cudaMalloc`, then use `cudaMemset` to initialize it. 
- Immediately after the call, we will check the returned `cudaError_t` and, if it's not `cudaSuccess`, print an error message and exit.
- For completeness and to show that the code works, we will also launch a trivial kernel that increments each element, copy the data back, and print the results.
- A helper macro `CHECK_CUDA_ERR` will be defined to reduce boilerplate for checking CUDA errors throughout the code.
- The code will include standard headers, CUDA runtime header, and use `printf` for output.
- The program will be self‑contained, compile with `nvcc`, and produce output verifying the operation.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/* Macro for checking CUDA errors */
#define CHECK_CUDA_ERR(call)                                                     \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                    \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

/* Simple kernel that increments each element by one */
__global__ void increment_kernel(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] += 1;
    }
}

int main(void) {
    const int N = 10;
    const size_t size = N * sizeof(int);

    /* Allocate device memory */
    int *d_data = nullptr;
    CHECK_CUDA_ERR(cudaMalloc((void**)&d_data, size));

    /* Initialize device memory with 42 using cudaMemset */
    /* Note: cudaMemset operates byte-wise. To set all bytes to 42, 
       we use a value that fits in a byte (0x2A). For multi‑byte types,
       this will set each byte to 0x2A, resulting in the integer value 0x2A2A2A2A. */
    CHECK_CUDA_ERR(cudaMemset(d_data, 42, size));

    /* Launch kernel to increment each element by one */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    increment_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    /* Check for kernel launch errors */
    CHECK_CUDA_ERR(cudaGetLastError());
    /* Wait for GPU to finish before accessing on host */
    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    /* Copy data back to host */
    int h_data[N];
    CHECK_CUDA_ERR(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    /* Print results */
    printf("Device data after cudaMemset(42) and increment kernel:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    /* Clean up */
    CHECK_CUDA_ERR(cudaFree(d_data));

    return 0;
}
