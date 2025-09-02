```cuda
/* 
Aim of the program: Implement a fused kernel that computes SAXPY and also the dot product of two vectors (using atomic adds for the reduction part).

Thinking:

1. **Fusion concept**: We want to compute `y[i] = a[i] + alpha * b[i]` (SAXPY) and simultaneously accumulate the dot product `sum += a[i] * b[i]`. Doing this in one kernel reduces memory traffic and kernel launch overhead.

2. **Atomic reduction**: Each thread will compute its local product `a[i] * b[i]` and atomically add it to a global accumulator. This avoids the need for shared memory reduction but can become a contention hotspot for very large arrays. For the purpose of demonstrating fusion, atomic adds are acceptable.

3. **Kernel design**:
   - Each thread handles one element `i`.
   - Compute `y[i]`.
   - Compute `prod = a[i] * b[i]`.
   - `atomicAdd(&dot, prod)`.

4. **Memory layout**: Vectors `a`, `b`, and output `y` are contiguous float arrays. The dot product accumulator is a single float stored on the device.

5. **Host side**:
   - Allocate host vectors, initialize with random values.
   - Allocate device memory, copy inputs.
   - Launch kernel with a reasonable block size (e.g., 256) and grid size covering all elements.
   - Copy back results for verification.
   - Optionally compute dot product on host for correctness check.

6. **Error checking**: Macro `CUDA_CHECK` to wrap CUDA API calls.

7. **Testing**: Print first and last elements of `y`, the computed dot product, and the reference dot product from host.

8. **Compilation**: The file is a standard `.cu` source and can be compiled with `nvcc`.

*/

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Fused kernel: SAXPY + dot product (using atomic add)
__global__ void fused_saxpy_dot(const float *a, const float *b, float *y,
                                float alpha, float *dot, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ai = a[idx];
        float bi = b[idx];
        // SAXPY
        y[idx] = ai + alpha * bi;
        // Dot product contribution
        float prod = ai * bi;
        atomicAdd(dot, prod);
    }
}

int main(int argc, char *argv[]) {
    size_t N = 1 << 20; // default size: 1,048,576 elements
    if (argc > 1) {
        N = strtoull(argv[1], nullptr, 10);
    }
    const float alpha = 2.0f;

    // Allocate host memory
    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_y = (float*)malloc(N * sizeof(float));
    if (!h_a || !h_b || !h_y) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize random seed
    std::srand((unsigned)std::time(nullptr));

    // Fill host vectors with random data
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(std::rand()) / RAND_MAX; // [0,1)
        h_b[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_y, *d_dot;
    CUDA_CHECK(cudaMalloc((void**)&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_y, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_dot, sizeof(float)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dot, 0, sizeof(float)));

    // Launch kernel
    const int threadsPerBlock = 256;
    int blocksPerGrid = (int)((N + threadsPerBlock - 1) / threadsPerBlock);
    fused_saxpy_dot<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_y, alpha, d_dot, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));
    float device_dot;
    CUDA_CHECK(cudaMemcpy(&device_dot, d_dot, sizeof(float), cudaMemcpyDeviceToHost));

    // Compute reference results on host
    float host_dot = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        host_dot += h_a[i] * h_b[i];
    }

    // Verify results
    printf("First element y[0] = %f (expected %f)\n", h_y[0], h_a[0] + alpha * h_b[0]);
    printf("Last element y[%zu] = %f (expected %f)\n", N-1, h_y[N-1], h_a[N-1] + alpha * h_b[N-1]);
    printf("Device dot product: %f\n", device_dot);
    printf("Host dot product:   %f\n", host_dot);
    printf("Dot product difference: %f\n", fabsf(device_dot - host_dot));

    // Clean up
    free(h_a);
    free(h_b);
    free(h_y);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_dot));

    return 0;
}
```