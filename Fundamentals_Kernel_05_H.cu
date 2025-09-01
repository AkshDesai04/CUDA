/*
Aim: Pass two arrays, in_a and in_b, and one output array out_c. Have each thread with global ID id compute out_c[id] = in_a[id] + in_b[id].

Thinking:
- The program demonstrates a simple vector addition on a GPU using CUDA.
- Two input arrays (in_a and in_b) and an output array (out_c) are allocated on the host.
- Memory is allocated on the device, data is copied from host to device.
- A kernel is launched where each thread calculates a single element of out_c by adding the corresponding elements of in_a and in_b.
- The global thread index is computed as `id = blockIdx.x * blockDim.x + threadIdx.x`.
- After kernel execution, the result is copied back to the host, verified, and all memory is freed.
- Error checking is added after each CUDA call for robustness.
- The program uses float arrays and a size of 1<<20 elements (about one million) for demonstration purposes.
- Launch configuration uses 256 threads per block, which is a common choice for maximizing occupancy.
- The host prints the first 10 results to confirm correctness.
*/

#include <iostream>
#include <cuda_runtime.h>

// Kernel: element-wise addition
__global__ void vecAdd(const float *in_a, const float *in_b, float *out_c, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        out_c[id] = in_a[id] + in_b[id];
    }
}

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in file '" << __FILE__ << "' in line " \
                      << __LINE__ << ": " << cudaGetErrorString(err)        \
                      << std::endl;                                         \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

int main() {
    const int N = 1 << 20; // 1,048,576 elements
    const int size = N * sizeof(float);

    // Allocate host memory
    float *h_in_a = new float[N];
    float *h_in_b = new float[N];
    float *h_out_c = new float[N];

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_in_a[i] = static_cast<float>(i);
        h_in_b[i] = static_cast<float>(2 * i);
    }

    // Allocate device memory
    float *d_in_a = nullptr;
    float *d_in_b = nullptr;
    float *d_out_c = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_in_a, size));
    CHECK_CUDA(cudaMalloc((void**)&d_in_b, size));
    CHECK_CUDA(cudaMalloc((void**)&d_out_c, size));

    // Copy input data from host to device
    CHECK_CUDA(cudaMemcpy(d_in_a, h_in_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_in_b, h_in_b, size, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_in_a, d_in_b, d_out_c, N);
    CHECK_CUDA(cudaGetLastError()); // Check for launch errors

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_out_c, d_out_c, size, cudaMemcpyDeviceToHost));

    // Verify a few results
    bool success = true;
    for (int i = 0; i < 10; ++i) {
        float expected = h_in_a[i] + h_in_b[i];
        if (fabs(h_out_c[i] - expected) > 1e-5) {
            std::cerr << "Mismatch at index " << i << ": "
                      << h_out_c[i] << " vs " << expected << std::endl;
            success = false;
            break;
        }
    }
    if (success) {
        std::cout << "First 10 results verified successfully." << std::endl;
    }

    // Free device memory
    CHECK_CUDA(cudaFree(d_in_a));
    CHECK_CUDA(cudaFree(d_in_b));
    CHECK_CUDA(cudaFree(d_out_c));

    // Free host memory
    delete[] h_in_a;
    delete[] h_in_b;
    delete[] h_out_c;

    return 0;
}
