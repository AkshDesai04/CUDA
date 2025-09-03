/*
Aim of the program: Wrap the grid/block calculation and kernel launch into a helper function.

Thinking process:

1. The goal is to demonstrate how to encapsulate the CUDA kernel launch configuration (block size, grid size) inside a reusable helper function. 
2. I chose a simple vector addition kernel (`addKernel`) as the example because it is a common pattern and easy to understand.
3. The helper function (`launchVectorAdd`) will:
   - Take device pointers `d_a`, `d_b`, `d_c`, and the number of elements `N`.
   - Compute the block dimension (`threadsPerBlock`) and grid dimension (`blocksPerGrid`) based on a fixed block size (e.g., 256 threads).
   - Launch the kernel with those dimensions.
   - Perform a simple error check after the launch.
4. The host code (`main`) will:
   - Allocate and initialize host arrays.
   - Allocate device memory.
   - Copy data to the device.
   - Call the helper function to launch the kernel.
   - Copy the result back to host.
   - Verify the result.
   - Clean up device memory.
5. I also included a macro `CUDA_CHECK` for error handling to make the code robust.
6. All code is within a single `.cu` file, ready to compile with `nvcc`.

The final program is a self‑contained CUDA example illustrating how to wrap launch configuration into a helper function. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Simple vector addition kernel
__global__ void addKernel(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Helper function that wraps grid/block calculation and kernel launch
void launchVectorAdd(const float *d_a, const float *d_b, float *d_c, int n) {
    const int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Check for launch errors
    CUDA_CHECK(cudaGetLastError());
    // Wait for GPU to finish before returning
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main(void) {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host arrays.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_a, size));
    CUDA_CHECK(cudaMalloc((void **)&d_b, size));
    CUDA_CHECK(cudaMalloc((void **)&d_c, size));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch kernel via helper function
    launchVectorAdd(d_a, d_b, d_c, N);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify result
    bool success = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: got %f, expected %f\n",
                    i, h_c[i], expected);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Vector addition successful for %d elements.\n", N);
    }

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
