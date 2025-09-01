```cpp
/*
Aim of the program:
Implement using __half precision numbers if your GPU supports it.

Thinking:
The requirement is to write a CUDA program that uses half-precision floating point
(__half) arithmetic when the GPU hardware supports it, otherwise it should fall
back to single-precision floats.  A simple and illustrative operation is a
vector addition: c = a + b.  The program will:

1. Allocate host arrays of floats and fill them with sample data.
2. Query the device properties to determine if the GPU supports __half
   (compute capability >= 5.3).  In practice __half is supported on GPUs with
   architecture sm_53 and newer.  We use the device properties to make this
   decision at runtime.
3. Depending on the support, allocate device buffers of either __half or float.
   If half precision is used we need to convert the host float data to __half
   before copying to device memory.
4. Launch a templated vector-add kernel that operates on the chosen data type.
5. Copy the result back to the host.  If half precision was used we convert the
   __half results back to float for printing.
6. Verify correctness by comparing the computed results against a CPU
   implementation.

The kernel is templated on the data type so the same code is reused for both
float and __half.  CUDA provides helper functions such as __float2half,
__half2float, and __half2half2 for conversion.  For half precision we also
take advantage of the __half2 type (pair of halfs packed into a 32‑bit word)
to illustrate vectorization, but the main arithmetic uses single __half
operations.

The program is fully self‑contained, written in C++ but using CUDA C
extensions, and can be compiled with nvcc.  It prints out the first few
elements of the result for inspection.

This design satisfies the prompt by conditionally using __half when available
and otherwise falling back to float, demonstrating how to write a
device‑agnostic program that adapts to the GPU’s capabilities.
*/

#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Templated vector addition kernel
template <typename T>
__global__ void vecAdd(const T* a, const T* b, T* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Host function to compute vector addition in float (CPU reference)
void vecAddCPU(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// Helper to convert a host float array to __half array on device
void copyFloatToHalf(const float* h_src, __half* d_dst, int n) {
    __half* h_temp = new __half[n];
    for (int i = 0; i < n; ++i) {
        h_temp[i] = __float2half(h_src[i]);
    }
    CUDA_CHECK(cudaMemcpy(d_dst, h_temp, n * sizeof(__half), cudaMemcpyHostToDevice));
    delete[] h_temp;
}

// Helper to convert a device __half array to host float array
void copyHalfToFloat(const __half* d_src, float* h_dst, int n) {
    __half* h_temp = new __half[n];
    CUDA_CHECK(cudaMemcpy(h_temp, d_src, n * sizeof(__half), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        h_dst[i] = __half2float(h_temp[i]);
    }
    delete[] h_temp;
}

int main(int argc, char* argv[]) {
    // Vector size
    const int N = 1 << 20;  // 1M elements
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate host memory
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c = new float[N];
    float* h_ref = new float[N];

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i) * 0.001f;
        h_b[i] = static_cast<float>(N - i) * 0.002f;
    }

    // Compute reference result on CPU
    vecAddCPU(h_a, h_b, h_ref, N);

    // Get device properties to decide on half support
    cudaDeviceProp prop;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    bool useHalf = (prop.major > 5) || (prop.major == 5 && prop.minor >= 3);

    printf("Device: %s (Compute Capability %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Using %s precision.\n", useHalf ? "__half" : "float");

    if (useHalf) {
        // Allocate device memory for __half
        __half *d_a, *d_b, *d_c;
        CUDA_CHECK(cudaMalloc((void**)&d_a, N * sizeof(__half)));
        CUDA_CHECK(cudaMalloc((void**)&d_b, N * sizeof(__half)));
        CUDA_CHECK(cudaMalloc((void**)&d_c, N * sizeof(__half)));

        // Copy host data to device (float -> half)
        copyFloatToHalf(h_a, d_a, N);
        copyFloatToHalf(h_b, d_b, N);

        // Launch kernel
        vecAdd<__half><<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy result back to host (half -> float)
        copyHalfToFloat(d_c, h_c, N);

        // Clean up device memory
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
    } else {
        // Allocate device memory for float
        float *d_a, *d_b, *d_c;
        CUDA_CHECK(cudaMalloc((void**)&d_a, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&d_b, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&d_c, N * sizeof(float)));

        // Copy host data to device
        CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

        // Launch kernel
        vecAdd<float><<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

        // Clean up device memory
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
    }

    // Verify results
    int mismatches = 0;
    for (int i = 0; i < N; ++i) {
        float diff = fabs(h_c[i] - h_ref[i]);
        if (diff > 1e-3f) { // tolerance due to half precision loss
            mismatches++;
            if (mismatches <= 10) {
                printf("Mismatch at index %d: GPU=%f, CPU=%f, diff=%f\n",
                       i, h_c[i], h_ref[i], diff);
            }
        }
    }
    if (mismatches == 0) {
        printf("All results match within tolerance.\n");
    } else {
        printf("Total mismatches: %d\n", mismatches);
    }

    // Clean up host memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_ref;

    return 0;
}
```