```cpp
/*
 Aim of the program:
 Declare an array using __device__, e.g., __device__ float lookup_table[256];
 Initialize it from the host and use it in a kernel.

 Thinking process:
 1. In CUDA, __device__ variables are allocated in the device's global memory and
    can be accessed from kernels. However, they cannot be directly copied with
    cudaMemcpy. Instead, we use cudaMemcpyToSymbol to copy data from a host array
    to the device __device__ array.

 2. We'll create a host array `h_lookup` of size 256 and fill it with some values
    (e.g., the index as a float). Then we copy it to the device array
    `lookup_table` using cudaMemcpyToSymbol.

 3. A simple kernel `use_lookup_table` will read values from `lookup_table` and
    write them to an output array `d_out`. Each thread will process one element
    based on its global index. This demonstrates that the device array is
    accessible inside the kernel.

 4. After kernel execution, we copy the result back to the host and print the
    first few elements to verify correctness.

 5. Error checking macros (`CUDA_CHECK`) are used to ensure that any CUDA API
    errors are reported.

 6. The program is fully self‑contained in a single .cu file and can be compiled
    with `nvcc -o lookup_table lookup_table.cu`.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define N 256

// Device array declared with __device__
__device__ float lookup_table[N];

// Kernel that uses the device array
__global__ void use_lookup_table(float *d_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Read from the device array and write to output
        d_out[idx] = lookup_table[idx];
    }
}

// Simple CUDA error checking macro
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

int main(void) {
    // Host array to initialize the device lookup_table
    float h_lookup[N];
    for (int i = 0; i < N; ++i) {
        h_lookup[i] = static_cast<float>(i) * 0.5f;  // Example values
    }

    // Copy host array to device __device__ array
    CUDA_CHECK(cudaMemcpyToSymbol(lookup_table, h_lookup, sizeof(float) * N));

    // Allocate device output array
    float *d_out;
    CUDA_CHECK(cudaMalloc((void**)&d_out, sizeof(float) * N));

    // Launch kernel: one thread per element
    int threadsPerBlock = 64;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    use_lookup_table<<<blocksPerGrid, threadsPerBlock>>>(d_out);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    float h_out[N];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost));

    // Print first 10 results to verify
    printf("First 10 elements from lookup_table via kernel:\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_out[%d] = %f\n", i, h_out[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_out));

    // Reset device
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
```