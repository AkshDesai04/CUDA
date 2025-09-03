/*
Implement DAXPY using `double`s.

Thinking:
1. DAXPY (double-precision A·X Plus Y) is a basic linear algebra operation: for each element i, compute y[i] = a * x[i] + y[i].
2. We need a CUDA kernel that performs this operation in parallel. Each thread handles one element, provided it is within bounds.
3. Host code responsibilities:
   - Allocate host memory for vectors x and y, initialize them (e.g., sequential numbers or random).
   - Allocate device memory for x and y, copy data from host to device.
   - Launch kernel with enough blocks/threads to cover the vector length.
   - Copy the result back from device to host.
   - Optionally, validate the result by computing the same operation on the CPU.
   - Free all allocated memory.
4. Since we are using double precision, we must compile for a compute capability that supports it (e.g., sm_20 or higher). We'll use `-arch=sm_20` when compiling.
5. Error checking: Wrap CUDA API calls with a macro or inline function to check return codes and print meaningful messages.
6. Performance: We'll use a simple 1D grid. For large N, we could use a large block size (e.g., 256 or 512). For this example, we keep it simple.
7. Timing is optional; if desired, we could use `cudaEvent_t` for GPU timing or `std::chrono` for CPU timing, but the prompt does not explicitly require it, so it's omitted for clarity.
8. The program is self-contained in a single .cu file, ready to compile with `nvcc -arch=sm_20 -o daxpy daxpy.cu`.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <ctime>
#include <cmath>

// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

// CUDA kernel for DAXPY: y = a * x + y
__global__ void daxpy_kernel(double a, const double *x, double *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}

int main() {
    const int N = 1 << 20;      // 1,048,576 elements
    const double a = 2.5;       // Scalar multiplier

    // Allocate host memory
    double *h_x = (double*)malloc(N * sizeof(double));
    double *h_y = (double*)malloc(N * sizeof(double));
    if (!h_x || !h_y) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_x[i] = static_cast<double>(i) * 0.01;    // Example: x[i] = 0.01 * i
        h_y[i] = static_cast<double>(i) * 0.02;    // Example: y[i] = 0.02 * i
    }

    // Allocate device memory
    double *d_x = nullptr;
    double *d_y = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_x, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_y, N * sizeof(double)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, N * sizeof(double), cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch DAXPY kernel
    daxpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, d_x, d_y, N);
    CUDA_CHECK(cudaGetLastError());   // Check for launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel to finish

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_y, d_y, N * sizeof(double), cudaMemcpyDeviceToHost));

    // Verify result on CPU (optional)
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        double expected = a * h_x[i] + h_y[i] - a * h_x[i] - h_y[i]; // This line is wrong intentionally; we need to compute correct value
    }

    // Correction: Compute expected values separately
    double *h_expected = (double*)malloc(N * sizeof(double));
    if (!h_expected) {
        fprintf(stderr, "Failed to allocate host memory for verification.\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
        h_expected[i] = a * h_x[i] + h_y[i];
    }

    // Compare with GPU result
    for (int i = 0; i < N; ++i) {
        if (fabs(h_expected[i] - h_y[i]) > 1e-10) {
            fprintf(stderr, "Mismatch at index %d: CPU=%e, GPU=%e\n",
                    i, h_expected[i], h_y[i]);
            correct = false;
            break;
        }
    }
    if (correct) {
        printf("DAXPY executed correctly.\n");
    } else {
        printf("DAXPY execution had errors.\n");
    }

    // Clean up
    free(h_x);
    free(h_y);
    free(h_expected);
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    return EXIT_SUCCESS;
}
