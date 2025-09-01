/*
Pass two arrays, `in_a` and `in_b`, and one output array `out_c`. Have each thread with global ID `id` compute `out_c[id] = in_a[id] + in_b[id]`.

This program demonstrates a simple element‑wise vector addition on the GPU using CUDA. The key idea is to map each element of the input arrays to a unique CUDA thread identified by its global thread ID. The kernel performs the addition only when the ID is within the bounds of the array length. The host code allocates host and device memory, initializes the input arrays, copies them to the device, launches the kernel, copies the result back, and finally cleans up resources. Basic CUDA error checking is included for robustness. The program uses single‑precision floating point (`float`) for the vectors, but it could be adapted to other data types easily. The kernel launch configuration uses a typical block size of 256 threads, adjusting the grid size accordingly. This example serves as a minimal yet complete template for CUDA kernel development and memory management.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple macro for CUDA error checking
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",     \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),         \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Kernel: each thread adds elements of in_a and in_b into out_c
__global__ void add_arrays(const float* in_a, const float* in_b, float* out_c, int N)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N)
        out_c[id] = in_a[id] + in_b[id];
}

int main(void)
{
    const int N = 1024;               // Number of elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // Initialize input arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    add_arrays<<<blocks, threads_per_block>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // (Optional) Verify results
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5) {
            if (errors < 10) { // Print only first few errors
                fprintf(stderr, "Mismatch at index %d: host=%f, device=%f, expected=%f\n",
                        i, h_c[i], expected, h_c[i]);
            }
            ++errors;
        }
    }
    if (errors == 0)
        printf("All results are correct.\n");
    else
        printf("Number of mismatches: %d\n", errors);

    // Clean up
    free(h_a);
    free(h_b);
    free(h_c);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
