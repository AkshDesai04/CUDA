```cpp
/*
Aim of the program:
Create a `bool` flag `is_correct` and set it to `false` if any element mismatches.
Print a final 'Verification successful' or 'Verification FAILED' message.

Thinking process:
1. We want a complete CUDA C program that demonstrates a simple computation on the GPU
   and verifies the results on the host.
2. Allocate two input arrays `a` and `b` and one output array `c` on the host.
3. Allocate corresponding device arrays.
4. Initialize `a` and `b` with known values (e.g., a[i] = i, b[i] = 2*i).
5. Launch a kernel that computes `c[i] = a[i] + b[i]`.
6. Copy the result back to the host.
7. Perform verification on the host:
   - Initialize `bool is_correct = true;`
   - Loop over all elements, compare the GPU result with the expected value
     (`expected = a[i] + b[i]`).
   - If a mismatch is found, set `is_correct = false;` and break the loop.
8. After the loop, print "Verification successful" if `is_correct` is true,
   otherwise print "Verification FAILED".
9. Include basic CUDA error checking for all API calls.
10. Use `nvcc` to compile the program.
*/

#include <iostream>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__        \
                      << " : " << cudaGetErrorString(err) << std::endl;         \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Simple kernel to add two arrays
__global__ void add_arrays(const float *a, const float *b, float *c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1024; // Number of elements

    // Host arrays
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // Device arrays
    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_a, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_b, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_c, N * sizeof(float)));

    // Copy input data from host to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add_arrays<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verification
    bool is_correct = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5f) { // tolerance for floating point
            is_correct = false;
            std::cerr << "Mismatch at index " << i
                      << ": GPU result = " << h_c[i]
                      << ", Expected = " << expected << std::endl;
            break;
        }
    }

    if (is_correct) {
        std::cout << "Verification successful" << std::endl;
    } else {
        std::cout << "Verification FAILED" << std::endl;
    }

    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return 0;
}
```