```cpp
/*
 * Aim of the program:
 * Write a kernel that takes three pointers, `a`, `b`, and `c`, and computes `c[i] = a[i] + b[i]`.
 *
 * My thinking while writing this program:
 * 1. The kernel must be a __global__ function so it can be launched from the host.
 * 2. It will accept three pointers to memory on the device: `a`, `b`, and `c`.
 * 3. The kernel needs to know how many elements it should process; thus an `int n` parameter is added.
 * 4. Each thread computes a unique index using the typical `blockIdx`, `blockDim`, and `threadIdx`.
 * 5. Boundary check (`if (idx < n)`) ensures no out‑of‑bounds memory accesses.
 * 6. The kernel performs the addition: `c[idx] = a[idx] + b[idx];`.
 *
 * To make the file self‑contained and compilable, a simple `main` function is added:
 *   - Allocate host arrays, initialize them.
 *   - Allocate device arrays and copy data from host to device.
 *   - Launch the kernel with a sensible grid/block configuration.
 *   - Copy the result back to host and print a few elements to verify.
 *   - Free all allocated memory.
 *
 * Basic error checking is performed after CUDA API calls to aid debugging.
 * The program uses `float` data type for simplicity, but the kernel is templated
 * on the pointer type, so it can be reused with other numeric types by modifying
 * the prototype.
 */

#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel: adds two arrays element-wise
__global__ void addArrays(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__     \
                      << " : " << cudaGetErrorString(err) << std::endl;      \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

int main()
{
    const int N = 1 << 20;  // 1M elements
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    // Initialize host arrays
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(N - i);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    addArrays<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());  // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify a few results
    std::cout << "c[0]   = " << h_c[0]   << " (expected: " << h_a[0] + h_b[0]   << ")" << std::endl;
    std::cout << "c[N/2] = " << h_c[N/2] << " (expected: " << h_a[N/2] + h_b[N/2] << ")" << std::endl;
    std::cout << "c[N-1] = " << h_c[N-1] << " (expected: " << h_a[N-1] + h_b[N-1] << ")" << std::endl;

    // Clean up
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
```