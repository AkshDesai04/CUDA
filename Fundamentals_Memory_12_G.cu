/*
Aim: Add error checking to all CUDA calls within the class methods.

Thinking:
- Use a macro `CUDA_CHECK` to wrap every CUDA API call, including memory allocation,
  memory copy, kernel launch, and synchronization.  
- The macro checks the return status; if an error occurs it prints a detailed
  message with file, line, and error string, then exits.
- Define a simple `GPUVector` class that manages a device array of floats.
  All methods (`allocate`, `copyToDevice`, `copyFromDevice`, `add`) will
  use `CUDA_CHECK` on every CUDA call.
- The `add` method launches a kernel that performs element‑wise addition.
  After the launch we check for launch errors (`cudaGetLastError`) and
  then synchronize the device, checking the status again.
- Provide a minimal `main` that demonstrates creating two vectors, copying
  data to the device, adding them, copying back, and verifying the result.
- The code is self‑contained, compiles as a single .cu file, and contains
  only the required error checking.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                \
            exit(err);                                                        \
        }                                                                     \
    } while (0)

// Kernel for element‑wise addition
__global__ void addKernel(float* a, const float* b, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        a[idx] += b[idx];
}

// Class that encapsulates a device float array
class GPUVector {
private:
    float* d_data;   // device pointer
    size_t size;     // number of elements

public:
    // Constructor: allocate device memory
    GPUVector(size_t n) : size(n), d_data(nullptr) {
        CUDA_CHECK(cudaMalloc((void**)&d_data, size * sizeof(float)));
    }

    // Destructor: free device memory
    ~GPUVector() {
        if (d_data) {
            CUDA_CHECK(cudaFree(d_data));
        }
    }

    // Copy data from host to device
    void copyToDevice(const float* h_data) {
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // Copy data from device to host
    void copyFromDevice(float* h_data) const {
        CUDA_CHECK(cudaMemcpy(h_data, d_data, size * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    // Element‑wise addition with another GPUVector
    void add(const GPUVector& other) {
        if (other.size != size) {
            fprintf(stderr, "Size mismatch in GPUVector::add\n");
            exit(EXIT_FAILURE);
        }
        const int threadsPerBlock = 256;
        int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
        addKernel<<<blocks, threadsPerBlock>>>(d_data, other.d_data, size);
        CUDA_CHECK(cudaGetLastError());          // Check kernel launch
        CUDA_CHECK(cudaDeviceSynchronize());     // Ensure completion
    }

    // Expose size (read‑only)
    size_t getSize() const { return size; }
};

int main()
{
    const size_t N = 1 << 20; // 1M elements
    size_t sizeBytes = N * sizeof(float);

    // Host data
    float* h_a = (float*)malloc(sizeBytes);
    float* h_b = (float*)malloc(sizeBytes);
    float* h_c = (float*)malloc(sizeBytes);

    // Initialize host arrays
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // Create GPU vectors
    GPUVector d_a(N);
    GPUVector d_b(N);

    // Copy data to device
    d_a.copyToDevice(h_a);
    d_b.copyToDevice(h_b);

    // Perform addition on GPU: d_a = d_a + d_b
    d_a.add(d_b);

    // Copy result back to host
    d_a.copyFromDevice(h_c);

    // Verify result
    bool success = true;
    for (size_t i = 0; i < N; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            fprintf(stderr, "Mismatch at index %zu: %f != %f + %f\n",
                    i, h_c[i], h_a[i], h_b[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Result verification succeeded.\n");
    }

    // Clean up host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
