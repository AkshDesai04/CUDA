```cuda
/* 
Add error checking to all CUDA calls within the class methods.

Thinking:
- We need a class that performs CUDA operations.
- Each CUDA API call (cudaMalloc, cudaMemcpy, kernel launch, cudaDeviceSynchronize, etc.) will be wrapped with error checking.
- We'll define a macro CHECK_CUDA that checks the return status and aborts if error.
- For kernel launch, we'll also call cudaGetLastError and cudaDeviceSynchronize with error checking.
- The class will have methods: init(), run(), cleanup().
- init() will allocate device memory; run() will copy data and launch a simple kernel; cleanup() will free memory.
- In main, we instantiate the class, call its methods, and print success.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Macro to check CUDA errors
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Simple vector addition kernel
__global__ void vecAdd(const float* a, const float* b, float* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

class VectorAddKernel
{
public:
    VectorAddKernel(int n)
        : size(n), d_a(nullptr), d_b(nullptr), d_c(nullptr),
          h_a(nullptr), h_b(nullptr), h_c(nullptr) {}

    ~VectorAddKernel() { cleanup(); }

    // Allocate host and device memory, initialize data
    void init()
    {
        // Allocate host memory
        h_a = (float*)malloc(sizeof(float) * size);
        h_b = (float*)malloc(sizeof(float) * size);
        h_c = (float*)malloc(sizeof(float) * size);

        if (!h_a || !h_b || !h_c) {
            fprintf(stderr, "Host memory allocation failed.\n");
            exit(EXIT_FAILURE);
        }

        // Initialize host data
        for (int i = 0; i < size; ++i) {
            h_a[i] = static_cast<float>(i);
            h_b[i] = static_cast<float>(2 * i);
        }

        // Allocate device memory
        CHECK_CUDA(cudaMalloc((void**)&d_a, sizeof(float) * size));
        CHECK_CUDA(cudaMalloc((void**)&d_b, sizeof(float) * size));
        CHECK_CUDA(cudaMalloc((void**)&d_c, sizeof(float) * size));
    }

    // Copy data to device, launch kernel, copy result back
    void run()
    {
        // Copy inputs to device
        CHECK_CUDA(cudaMemcpy(d_a, h_a, sizeof(float) * size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b, h_b, sizeof(float) * size, cudaMemcpyHostToDevice));

        // Determine grid and block dimensions
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

        // Launch kernel
        vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);

        // Check for kernel launch errors
        CHECK_CUDA(cudaGetLastError());

        // Synchronize to ensure kernel completion
        CHECK_CUDA(cudaDeviceSynchronize());

        // Copy result back to host
        CHECK_CUDA(cudaMemcpy(h_c, d_c, sizeof(float) * size, cudaMemcpyDeviceToHost));
    }

    // Free device and host memory
    void cleanup()
    {
        if (d_a) CHECK_CUDA(cudaFree(d_a));
        if (d_b) CHECK_CUDA(cudaFree(d_b));
        if (d_c) CHECK_CUDA(cudaFree(d_c));

        if (h_a) free(h_a);
        if (h_b) free(h_b);
        if (h_c) free(h_c);

        d_a = d_b = d_c = nullptr;
        h_a = h_b = h_c = nullptr;
    }

    // Verify results
    void verify()
    {
        for (int i = 0; i < size; ++i) {
            float expected = h_a[i] + h_b[i];
            if (fabs(h_c[i] - expected) > 1e-5) {
                fprintf(stderr, "Verification failed at index %d: %f != %f\n",
                        i, h_c[i], expected);
                exit(EXIT_FAILURE);
            }
        }
        printf("Verification succeeded.\n");
    }

private:
    int size;
    float *d_a, *d_b, *d_c;
    float *h_a, *h_b, *h_c;
};

int main()
{
    int N = 1 << 20; // 1M elements

    VectorAddKernel vecAddKernel(N);

    vecAddKernel.init();
    vecAddKernel.run();
    vecAddKernel.verify();
    vecAddKernel.cleanup();

    printf("Vector addition completed successfully.\n");
    return 0;
}
```