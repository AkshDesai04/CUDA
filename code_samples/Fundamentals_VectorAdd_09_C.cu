/*
Implement `D[i] = (A[i] > B[i]) ? A[i] * s1 : B[i] * s2`, where `s1` and `s2` are scalars.

My thinking:
- I will create a CUDA kernel that receives pointers to the input arrays `A` and `B`, an output array `D`, the scalars `s1` and `s2`, and the number of elements `N`.  
- Each thread will compute one element `i = blockIdx.x * blockDim.x + threadIdx.x`.  
- Bounds checking will be performed inside the kernel: if `i < N`, the computation is performed; otherwise the thread does nothing.  
- The host code will:
  1. Allocate device memory for `A`, `B`, and `D` using `cudaMalloc`.  
  2. Allocate host memory and initialize `A` and `B` with some test data (e.g., random floats).  
  3. Copy the host arrays to the device with `cudaMemcpy`.  
  4. Define scalars `s1` and `s2` and set the grid and block dimensions (a common choice is 256 threads per block).  
  5. Launch the kernel.  
  6. Copy the result back to the host.  
  7. Optionally print a few elements to verify correctness.  
- I will also include a simple macro `CUDA_CHECK` for error handling after each CUDA API call to make debugging easier.  
- The code will be self‑contained and compile with `nvcc`.  
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",             \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

__global__ void compute_kernel(const float* A, const float* B, float* D,
                               float s1, float s2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        D[idx] = (A[idx] > B[idx]) ? A[idx] * s1 : B[idx] * s2;
    }
}

int main() {
    const int N = 1 << 20; // 1,048,576 elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_D = (float*)malloc(size);

    // Initialize host arrays with random values
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Scalars
    float s1 = 2.0f;
    float s2 = 3.0f;

    // Allocate device memory
    float *d_A, *d_B, *d_D;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_D, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    compute_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_D, s1, s2, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_D, d_D, size, cudaMemcpyDeviceToHost));

    // Optional: print first 10 results
    printf("First 10 results:\n");
    for (int i = 0; i < 10; ++i) {
        printf("D[%d] = %f\n", i, h_D[i]);
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_D);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_D));

    return 0;
}
