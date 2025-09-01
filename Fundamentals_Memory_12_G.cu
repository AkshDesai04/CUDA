/*
Add error checking to all CUDA calls within the class methods.

The user requested that all CUDA calls inside a class be wrapped with error checking.  
I considered that a typical example of a class performing CUDA work is a simple vector
addition class.  The class will allocate device memory, copy data to the device,
launch a kernel, copy the result back, and free resources.  Every CUDA API call
(`cudaMalloc`, `cudaMemcpy`, kernel launch, `cudaGetLastError`, `cudaDeviceSynchronize`,
`cudaFree`) is wrapped with a macro that checks the return status.  If an error occurs
the program prints a helpful message including file and line information and exits
with `EXIT_FAILURE`.  

The macro is defined once and used throughout the class methods.  
The kernel is trivial but illustrates how to launch it and then check for errors
immediately after the launch.  The destructor also checks `cudaFree`.  

The resulting .cu file is self‑contained and demonstrates comprehensive error
handling for all CUDA calls within a class.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Macro to check CUDA API calls
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Simple vector addition kernel
__global__ void vecAddKernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

class VectorAdd {
public:
    VectorAdd(int n) : n_(n) {
        size_in_bytes_ = n_ * sizeof(float);

        // Allocate device memory with error checking
        CUDA_CHECK(cudaMalloc((void**)&d_a_, size_in_bytes_));
        CUDA_CHECK(cudaMalloc((void**)&d_b_, size_in_bytes_));
        CUDA_CHECK(cudaMalloc((void**)&d_c_, size_in_bytes_));
    }

    void addVectors(const float* h_a, const float* h_b, float* h_c) {
        // Copy input vectors to device with error checking
        CUDA_CHECK(cudaMemcpy(d_a_, h_a, size_in_bytes_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b_, h_b, size_in_bytes_, cudaMemcpyHostToDevice));

        // Determine grid and block dimensions
        int threadsPerBlock = 256;
        int blocksPerGrid = (n_ + threadsPerBlock - 1) / threadsPerBlock;

        // Launch kernel and check for errors
        vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a_, d_b_, d_c_, n_);
        CUDA_CHECK(cudaGetLastError());           // Check for launch errors
        CUDA_CHECK(cudaDeviceSynchronize());      // Wait for completion and catch errors

        // Copy result back to host with error checking
        CUDA_CHECK(cudaMemcpy(h_c, d_c_, size_in_bytes_, cudaMemcpyDeviceToHost));
    }

    ~VectorAdd() {
        // Free device memory with error checking
        if (d_a_) CUDA_CHECK(cudaFree(d_a_));
        if (d_b_) CUDA_CHECK(cudaFree(d_b_));
        if (d_c_) CUDA_CHECK(cudaFree(d_c_));
    }

private:
    int n_;
    size_t size_in_bytes_;
    float *d_a_ = nullptr;
    float *d_b_ = nullptr;
    float *d_c_ = nullptr;
};

int main() {
    const int N = 1 << 20; // 1M elements
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // Create VectorAdd object
    VectorAdd va(N);

    // Perform vector addition
    va.addVectors(h_a, h_b, h_c);

    // Verify results (optional)
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            success = false;
            fprintf(stderr, "Verification failed at index %d: %f != %f + %f\n",
                    i, h_c[i], h_a[i], h_b[i]);
            break;
        }
    }
    if (success) {
        printf("Vector addition succeeded.\n");
    }

    // Clean up host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
