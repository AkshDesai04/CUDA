```cpp
/* 
Aim of the program: Verify the results of subtraction on the CPU.

Thinking:
The task is to write a CUDA program that performs subtraction of two arrays
on the GPU, copies the result back to the host, and then verifies those
results by performing the same subtraction on the CPU.  This allows us to
confirm that the GPU kernel produces correct results by comparing it to a
trusted CPU implementation.

Key design decisions:
1. Use float arrays for simplicity; the same approach works for other types.
2. Define a kernel `subtract_kernel` that performs element‑wise subtraction
   of arrays `a` and `b` into array `c`.
3. Allocate host arrays `h_a`, `h_b`, `h_c_gpu`, and `h_c_cpu`.
4. Initialize `h_a` and `h_b` with deterministic values so that the result
   is predictable and easy to verify.
5. Copy `h_a` and `h_b` to device memory `d_a` and `d_b`, launch the kernel,
   and copy the resulting `d_c` back to `h_c_gpu`.
6. Perform the same subtraction on the host to obtain `h_c_cpu`.
7. Compare `h_c_gpu` and `h_c_cpu` element‑wise; report mismatches if any.
8. Include error checking for CUDA API calls and kernel launch.
9. Use a modest array size (e.g., 1024) to keep the example concise, but
   the code scales to larger sizes.
10. Ensure that all allocated resources are freed before program exit.

The program prints a success message if all elements match, otherwise
reports the first mismatch encountered. This demonstrates verification of
GPU subtraction results against CPU computations.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",      \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

// Kernel to perform element-wise subtraction: c = a - b
__global__ void subtract_kernel(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

int main(void) {
    const int N = 1024;                  // Number of elements
    const int size = N * sizeof(float);  // Size in bytes

    // Allocate host memory
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c_gpu = (float *)malloc(size);
    float *h_c_cpu = (float *)malloc(size);

    if (!h_a || !h_b || !h_c_gpu || !h_c_cpu) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host arrays with deterministic values
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);          // 0.0, 1.0, 2.0, ...
        h_b[i] = static_cast<float>(N - i - 1);  // N-1, N-2, ..., 0
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, size));
    CUDA_CHECK(cudaMalloc((void **)&d_b, size));
    CUDA_CHECK(cudaMalloc((void **)&d_c, size));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Determine block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    subtract_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());  // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost));

    // Perform subtraction on CPU
    for (int i = 0; i < N; ++i) {
        h_c_cpu[i] = h_a[i] - h_b[i];
    }

    // Verify results
    bool match = true;
    const float epsilon = 1e-5f;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_c_gpu[i] - h_c_cpu[i]) > epsilon) {
            fprintf(stderr, "Mismatch at index %d: GPU=%f, CPU=%f\n",
                    i, h_c_gpu[i], h_c_cpu[i]);
            match = false;
            break;
        }
    }

    if (match) {
        printf("All results match. Subtraction verified successfully.\n");
    } else {
        printf("Verification failed.\n");
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c_gpu);
    free(h_c_cpu);

    return 0;
}
```