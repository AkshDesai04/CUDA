```cpp
/* 
Aim: Implement a fused kernel that computes SAXPY and also the dot product of two vectors (using atomic adds for the reduction part).

Thinking:
- The fused kernel should perform the SAXPY operation: z[i] = a * x[i] + y[i] for all i in the vector.
- Simultaneously, it should compute the dot product of the two input vectors x and y: dot = Σ_i x[i] * y[i].
- To avoid race conditions when accumulating the dot product, atomicAdd will be used on a single float variable in global memory.
- The kernel will be launched with a 1D grid of blocks; each thread processes one element.
- Each thread will:
  1. Load x[i] and y[i] from global memory.
  2. Compute z[i] = a * x[i] + y[i] and store it back to global memory.
  3. Compute the product x[i] * y[i] and atomically add it to the global accumulator.
- Host code will:
  1. Allocate device memory for x, y, z, and the accumulator.
  2. Initialize x and y with test values.
  3. Copy data to device.
  4. Launch the kernel.
  5. Copy back the result vector z and the scalar dot product.
  6. Print the results for verification.
- Error checking macros will be used to simplify CUDA API error handling.
- The code is written in CUDA C++ and should compile with a GPU supporting atomicAdd for floats (compute capability 2.0+).
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Fused SAXPY and dot product kernel
__global__ void fused_saxpy_dot(const float *x, const float *y, float *z,
                                float a, int n, float *dot_accum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float xi = x[idx];
        float yi = y[idx];
        // SAXPY
        z[idx] = a * xi + yi;
        // Atomic dot product
        float prod = xi * yi;
        atomicAdd(dot_accum, prod);
    }
}

int main()
{
    const int N = 1 << 20; // 1M elements
    const float a = 2.5f;

    // Allocate host memory
    float *h_x = (float*)malloc(N * sizeof(float));
    float *h_y = (float*)malloc(N * sizeof(float));
    float *h_z = (float*)malloc(N * sizeof(float));
    float h_dot = 0.0f;

    // Initialize input vectors
    for (int i = 0; i < N; ++i) {
        h_x[i] = static_cast<float>(i) * 0.001f;
        h_y[i] = static_cast<float>(i) * 0.002f;
    }

    // Allocate device memory
    float *d_x, *d_y, *d_z, *d_dot;
    CUDA_CHECK(cudaMalloc((void**)&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_y, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_z, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_dot, sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dot, 0, sizeof(float))); // initialize accumulator to zero

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    fused_saxpy_dot<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, a, N, d_dot);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_z, d_z, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_dot, d_dot, sizeof(float), cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("Sample outputs (index: x, y, z, dot component):\n");
    for (int i = 0; i < 5; ++i) {
        printf("%d: x=%.4f, y=%.4f, z=%.4f, x*y=%.4f\n",
               i, h_x[i], h_y[i], h_z[i], h_x[i] * h_y[i]);
    }
    printf("Computed dot product: %.6f\n", h_dot);

    // Clean up
    free(h_x);
    free(h_y);
    free(h_z);
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_dot));

    return 0;
}
```