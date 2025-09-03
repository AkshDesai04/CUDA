```cuda
/* 
Use a grid-stride loop to initialize a very large array to a specific value. 

Thinking process:
- The requirement is to write a CUDA program that initializes a large array on the device using a grid-stride loop.
- The array will be allocated on the device using cudaMalloc, and the kernel will be launched with a configuration that can cover the entire array even if the array size exceeds the total number of threads.
- Inside the kernel, each thread will loop over its assigned indices using the grid-stride pattern: 
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x)
- A specific value (e.g., 42.0f) will be written into each element.
- After the kernel completes, the host will copy the data back with cudaMemcpy, and optionally verify a few elements.
- Error checking macros are included to catch CUDA runtime errors.
- The program is self-contained and can be compiled with nvcc.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Error checking macro
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel: grid-stride loop to initialize array
__global__ void init_array(float* arr, size_t N, float value)
{
    size_t stride = blockDim.x * gridDim.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < N; i += stride)
    {
        arr[i] = value;
    }
}

int main()
{
    const size_t N = 100'000'000;          // 100 million elements
    const float init_value = 42.0f;

    // Allocate device memory
    float* d_arr = nullptr;
    CHECK_CUDA(cudaMalloc(&d_arr, N * sizeof(float)));

    // Launch kernel with a suitable grid and block size
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 65535) blocksPerGrid = 65535; // Limit to max blocks per grid

    init_array<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N, init_value);
    CHECK_CUDA(cudaGetLastError()); // Check kernel launch errors

    // Allocate host memory to verify
    float* h_arr = (float*)malloc(N * sizeof(float));
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Copy back to host
    CHECK_CUDA(cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Simple verification: check first, middle, last elements
    bool ok = true;
    if (h_arr[0] != init_value) ok = false;
    if (h_arr[N/2] != init_value) ok = false;
    if (h_arr[N-1] != init_value) ok = false;

    printf("Verification %s\n", ok ? "PASSED" : "FAILED");

    // Clean up
    free(h_arr);
    CHECK_CUDA(cudaFree(d_arr));

    return 0;
}
```