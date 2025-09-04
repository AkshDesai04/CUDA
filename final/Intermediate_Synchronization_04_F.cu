```cu
/*
Atomics on floating point numbers were not always supported. Check your device's compute capability. Modern GPUs support them for `float` and `double`.

This program demonstrates the use of atomicAdd on both float and double types in CUDA. 
The goal is to show that modern GPUs (compute capability >= 6.0) provide native support for atomic operations on floating point numbers, which was not always the case in older GPU architectures. 
To that end, the program performs the following steps:

1. Query the first CUDA device and retrieve its compute capability (major and minor version). This information is printed so the user can confirm whether the device supports atomic operations on floating point values.
2. Allocate device memory for a single float and a single double variable, both initialized to zero.
3. Launch a kernel where many threads concurrently perform atomicAdd operations on these variables. Each thread adds a small constant (0.5f for float, 0.25 for double) to demonstrate the accumulation across threads.
4. Copy the results back to host memory and print the final accumulated values. If the GPU supports the required atomics, the results should be the expected multiples of the thread count.
5. Clean up device memory and exit.

The program also includes error checking after each CUDA API call to ensure that any failure is reported. 
Note that the kernel uses the built-in atomicAdd functions for float and double, which are only available on devices with compute capability 6.0 or higher. If a lower compute capability is detected, the program will still run but the atomicAdd on double may degrade to a slower implementation or produce incorrect results, depending on the CUDA version. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",       \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),          \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel that performs atomicAdd on float and double
__global__ void atomicAddKernel(float *f_val, double *d_val, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        // Each thread adds a small constant
        atomicAdd(f_val, 0.5f);
        atomicAdd(d_val, 0.25);
    }
}

int main(void) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }

    // Use the first device
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("Using device %d: %s\n", device, prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Supports atomicAdd for float: %s\n", (prop.major > 2 || (prop.major == 2 && prop.minor >= 0)) ? "Yes" : "No");
    printf("Supports atomicAdd for double: %s\n", (prop.major > 3 || (prop.major == 3 && prop.minor >= 5)) ? "Yes" : "No");

    // Allocate device memory for float and double
    float *d_f = NULL;
    double *d_d = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_f, sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_d, sizeof(double)));

    // Initialize to zero
    CUDA_CHECK(cudaMemset(d_f, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_d, 0, sizeof(double)));

    const int numThreads = 1024;
    const int threadsPerBlock = 256;
    const int numBlocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    atomicAddKernel<<<numBlocks, threadsPerBlock>>>(d_f, d_d, numThreads);
    CUDA_CHECK(cudaGetLastError());

    // Copy results back to host
    float h_f = 0.0f;
    double h_d = 0.0;
    CUDA_CHECK(cudaMemcpy(&h_f, d_f, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_d, d_d, sizeof(double), cudaMemcpyDeviceToHost));

    printf("Final accumulated float value: %f (expected %f)\n", h_f, 0.5f * numThreads);
    printf("Final accumulated double value: %lf (expected %lf)\n", h_d, 0.25 * numThreads);

    // Clean up
    CUDA_CHECK(cudaFree(d_f));
    CUDA_CHECK(cudaFree(d_d));

    return EXIT_SUCCESS;
}
```