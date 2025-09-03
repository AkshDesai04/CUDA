/*
Call `cudaDeviceReset()` at the very beginning of your `main` function to ensure a clean state.

My thinking:

- The user requested a CUDA C program (.cu) that explicitly calls `cudaDeviceReset()` at the very start of `main`. 
- The program should compile on any standard CUDA-capable compiler (nvcc).
- I will create a minimal, but functional example: a simple vector addition kernel. 
- The code will:
  1. Call `cudaDeviceReset()` immediately after the start of `main`.
  2. Allocate host and device memory for two input vectors and an output vector.
  3. Initialize the input vectors on the host.
  4. Copy the inputs to the device.
  5. Launch a kernel that adds the vectors element-wise.
  6. Copy the result back to the host.
  7. Verify and print a few output values.
  8. Free all allocated memory and call `cudaDeviceReset()` again (optional but good practice).
- I will also include error checking for CUDA API calls to make debugging easier. 
- The code will use the standard `cuda_runtime.h` header and the `printf` function for output.
- The program will be self-contained and will not rely on any external files or libraries beyond the CUDA runtime.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Simple kernel to add two vectors
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    // Ensure a clean state before anything else
    CHECK_CUDA(cudaDeviceReset());

    const int N = 1 << 20;  // 1M elements
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize input vectors
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc((void**)&d_a, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_b, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_c, bytes));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError()); // Check kernel launch

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify a few elements
    int errors = 0;
    for (int i = 0; i < 10; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: got %f, expected %f\n",
                    i, h_c[i], expected);
            errors++;
        }
    }
    if (errors == 0) {
        printf("Vector addition succeeded for first 10 elements.\n");
    }

    // Free device memory
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    // Reset device at end for cleanliness (optional)
    CHECK_CUDA(cudaDeviceReset());

    return EXIT_SUCCESS;
}
