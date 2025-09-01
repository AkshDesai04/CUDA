```cuda
/*
Aim of the program: Implement for `float` data.

Thinking and Design Notes:
1. Since the prompt only specifies “implement for `float` data”, I chose a classic GPU compute
   example that demonstrates how to work with single‑precision floating‑point values on
   NVIDIA hardware: element‑wise vector addition.
2. The program will:
   - Allocate two input vectors `a` and `b` on the host (CPU) and fill them with
     deterministic values so that the expected result is known.
   - Allocate corresponding device memory, copy the data to the GPU.
   - Launch a CUDA kernel that adds the two vectors element‑wise, storing the result
     in vector `c` on the device.
   - Copy the result back to the host and verify correctness by comparing with the
     CPU reference result.
3. I’ll use a simple error‑checking macro `CUDA_CHECK` to simplify error handling.
4. The kernel will be straightforward: each thread handles one element.
5. The code will compile with `nvcc` and will run on any CUDA‑enabled GPU that supports
   compute capability >= 2.0. The vector size will be large enough to show the benefit
   of GPU parallelism but small enough to keep the example clear.
6. All code is contained in a single `.cu` file with no external dependencies.

*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Kernel: element‑wise addition of two float vectors
__global__ void vecAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void) {
    const int N = 1 << 20;          // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays with deterministic values
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i * 0.5f;
        h_b[i] = (float)i * 1.5f;
    }

    // Allocate device memory
    float *d_a = NULL, *d_b = NULL, *d_c = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_a, size));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size));
    CUDA_CHECK(cudaMalloc((void**)&d_c, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());  // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify results
    bool success = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: GPU %f != CPU %f\n",
                    i, h_c[i], expected);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Vector addition successful. All %d elements match.\n", N);
    } else {
        printf("Vector addition failed.\n");
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
```