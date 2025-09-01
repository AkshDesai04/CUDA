```cpp
/*
Aim of the program:
Declare an array using `__device__`, e.g., `__device__ float lookup_table[256];`. Initialize it from the host and use it in a kernel.

Thinking:
To demonstrate host initialization of a __device__ array, we declare a global array of 256 floats on the device side.  
We create a host array with known values (here, simply index * 0.1f) and copy it to the device using cudaMemcpyToSymbol.  
A simple kernel will read indices from a device array and write the corresponding lookup table values to an output array.  
After kernel execution, we copy the output back to host and print it to verify correctness.  
The code includes basic error checking via a macro and uses standard CUDA runtime API functions. This is a self‑contained .cu file that can be compiled with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Device lookup table of 256 floats
__device__ float lookup_table[256];

// Kernel that uses the device lookup table
__global__ void use_lookup(const int *indices, float *results, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int lookupIdx = indices[idx];
        // Ensure lookupIdx is within bounds (0-255)
        if (lookupIdx >= 0 && lookupIdx < 256) {
            results[idx] = lookup_table[lookupIdx];
        } else {
            results[idx] = -1.0f; // sentinel for out-of-range
        }
    }
}

// Macro for error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main() {
    const int TABLE_SIZE = 256;
    const int N = 10; // number of indices to process

    // Host lookup table initialization
    float host_table[TABLE_SIZE];
    for (int i = 0; i < TABLE_SIZE; ++i) {
        host_table[i] = i * 0.1f; // example values
    }

    // Copy host table to device symbol
    CUDA_CHECK(cudaMemcpyToSymbol(lookup_table, host_table, TABLE_SIZE * sizeof(float)));

    // Host arrays for indices and results
    int h_indices[N];
    for (int i = 0; i < N; ++i) {
        h_indices[i] = i * 10; // indices: 0, 10, 20, ..., 90
    }

    float h_results[N] = {0};

    // Device memory allocation
    int *d_indices = nullptr;
    float *d_results = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_indices, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_results, N * sizeof(float)));

    // Copy indices to device
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel: one thread per element
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    use_lookup<<<blocksPerGrid, threadsPerBlock>>>(d_indices, d_results, N);
    CUDA_CHECK(cudaGetLastError()); // check for launch errors

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_results, d_results, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print results
    printf("Lookup results:\n");
    for (int i = 0; i < N; ++i) {
        printf("Index %d -> %f\n", h_indices[i], h_results[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_results));

    return 0;
}
```