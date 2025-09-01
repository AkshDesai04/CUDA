/*
 * Aim of the program: Refactor your vector addition program from the previous section to use the `CHECK()` macro for every single CUDA API call (`cudaMalloc`, `cudaMemcpy`, `cudaFree`, etc.).
 *
 * Thinking:
 * The goal is to replace all raw CUDA API calls with a wrapper macro that checks for errors immediately.
 * The macro `CHECK` will call the CUDA API, capture the returned `cudaError_t`, and if it's not `cudaSuccess` it will print an informative message and terminate the program.
 * This approach guarantees that each API call is verified, making debugging easier and preventing silent failures.
 *
 * I will define the macro at the top, then write a simple vector addition kernel and host code.
 * Every allocation (`cudaMalloc`), memory copy (`cudaMemcpy`), and deallocation (`cudaFree`) will be wrapped by `CHECK`.
 * After launching the kernel, I will also call `CHECK(cudaGetLastError())` and `CHECK(cudaDeviceSynchronize())` to catch launch errors.
 *
 * The program will allocate two input vectors, fill them with sample data, copy them to the GPU,
 * perform addition on the GPU, copy the result back, and finally free all resources.
 * The host result will be validated against a simple CPU-computed reference to ensure correctness.
 *
 * All code is written in C with CUDA extensions, suitable for compiling with `nvcc` into a `.cu` file.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call)                                            \
    do {                                                       \
        cudaError_t err = call;                                \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

__global__ void vecAdd(const float *A, const float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f; // arbitrary data
        h_B[i] = 2.0f; // arbitrary data
    }

    // Allocate device memory
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    CHECK(cudaMalloc((void**)&d_A, size));
    CHECK(cudaMalloc((void**)&d_B, size));
    CHECK(cudaMalloc((void**)&d_C, size));

    // Copy host to device
    CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify result
    int correct = 1;
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            fprintf(stderr, "Result mismatch at index %d: GPU %f != CPU %f\n",
                    i, h_C[i], h_A[i] + h_B[i]);
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("Vector addition successful, all results correct.\n");
    else
        printf("Vector addition failed.\n");

    // Free device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
