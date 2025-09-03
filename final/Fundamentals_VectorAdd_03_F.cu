```cpp
/*
Apply the same boundary check logic to a vector squaring kernel: `if (i < N) out[i] = in[i] * in[i]`.

Thinking:
- The goal is to create a minimal yet complete CUDA program that demonstrates how to safely
  launch a kernel which operates on a vector, ensuring that threads that map outside the
  valid range do not access memory. The kernel will compute the square of each element.
- We'll use a standard CUDA error checking macro to keep the code clean.
- The host code will:
  1. Allocate and initialize an input array on the host.
  2. Allocate corresponding device memory.
  3. Copy the data to the device.
  4. Launch the kernel with a configurable block and grid size.
  5. Copy the result back to the host.
  6. Verify a few elements for correctness.
  7. Clean up resources.
- Boundary checking is performed in the kernel by testing if the global thread index
  `i` is less than `N`. Only those threads that satisfy this condition perform the
  computation.
- This program can be compiled with `nvcc` and run on any device that supports the
  required compute capability.
*/

#include <iostream>
#include <cuda_runtime.h>

// Error checking macro
#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            std::cerr << "CUDA error in " << __FILE__ << ":"      \
                      << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Device kernel: square each element with boundary check
__global__ void squareKernel(const float* __restrict__ in,
                             float* __restrict__ out,
                             int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = in[i] * in[i];
    }
}

int main()
{
    const int N = 1 << 20;  // Example vector size: 1,048,576
    const size_t bytes = N * sizeof(float);

    // Host allocations
    float* h_in  = new float[N];
    float* h_out = new float[N];

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_in[i] = static_cast<float>(i);
    }

    // Device allocations
    float *d_in = nullptr, *d_out = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_in,  bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_out, bytes));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Launch configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    squareKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);
    CHECK_CUDA(cudaGetLastError()); // Check kernel launch

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // Simple verification
    bool correct = true;
    for (int i = 0; i < 10; ++i) { // check first 10 elements
        float expected = h_in[i] * h_in[i];
        if (fabs(h_out[i] - expected) > 1e-5) {
            std::cerr << "Mismatch at index " << i
                      << ": got " << h_out[i]
                      << ", expected " << expected << std::endl;
            correct = false;
            break;
        }
    }
    if (correct) {
        std::cout << "Verification passed for first 10 elements." << std::endl;
    }

    // Clean up
    delete[] h_in;
    delete[] h_out;
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}
```