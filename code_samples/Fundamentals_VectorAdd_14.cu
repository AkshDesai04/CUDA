/*
Implement vector addition for `double` precision floating-point numbers instead of `float`s.

Thinking:
- Use a simple CUDA kernel `vectorAdd` that operates on `double` arrays.
- Allocate host arrays A, B, C of type double and initialize A and B with some values.
- Allocate device arrays and copy data from host to device.
- Launch the kernel with enough blocks/threads to cover N elements.
- Copy result C back to host and verify correctness.
- Include basic error checking via a macro that prints CUDA errors.
- Query device properties to ensure double precision support (compute capability >= 1.3).
- Keep the code self-contained, compiling with `nvcc`.
- Use standard headers `<cstdio>` and `<cstdlib>`; avoid extra libraries.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Kernel for vector addition of double precision floats
__global__ void vectorAdd(const double *A, const double *B, double *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    const int N = 1 << 20;  // 1M elements
    const size_t size = N * sizeof(double);

    // Allocate host memory
    double *h_A = (double *)malloc(size);
    double *h_B = (double *)malloc(size);
    double *h_C = (double *)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<double>(i);
        h_B[i] = static_cast<double>(2 * i);
    }

    // Allocate device memory
    double *d_A = nullptr;
    double *d_B = nullptr;
    double *d_C = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size));

    // Copy host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify results
    bool success = true;
    for (int i = 0; i < N; ++i) {
        double expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-9) {  // tolerance for double
            fprintf(stderr, "Mismatch at index %d: host %f, device %f\n",
                    i, expected, h_C[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Vector addition successful.\n");
    }

    // Free resources
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return EXIT_SUCCESS;
}
