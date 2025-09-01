```cuda
/*
Aim of the program: Try allocating pinned memory with the `cudaHostAllocMapped` flag.

Thinking:
1. The goal is to demonstrate how to allocate host memory that is both pinned (page-locked)
   and mapped into the device address space, using the flag `cudaHostAllocMapped`.  
2. Once we have the mapped host pointer, we can obtain a device pointer that points to
   the same physical memory using `cudaHostGetDevicePointer`.  
3. The program will:
   - Allocate a small array on the host with `cudaHostAlloc` using `cudaHostAllocMapped`.
   - Fill the host array with some data.
   - Get the corresponding device pointer.
   - Perform a simple kernel that doubles each element in place, operating directly on
     the mapped memory via the device pointer. Because the memory is shared, the host
     will see the changes immediately without an explicit copy.
   - Finally, print the results from the host to verify that the kernel ran correctly.
4. We'll include error checking macros for clarity.
5. The code will be a minimal, self‑contained CUDA C example that compiles with `nvcc`
   and runs on a device that supports unified virtual addressing (most recent GPUs).
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple macro for CUDA error checking
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    __FUNCTION__, __FILE__, __LINE__,               \
                    cudaGetErrorString(err));                       \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Kernel that doubles each element in the array
__global__ void double_elements(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] *= 2.0f;
    }
}

int main() {
    const int N = 10;
    const size_t bytes = N * sizeof(float);

    // Allocate pinned, mapped host memory
    float *h_ptr = nullptr;
    CUDA_CHECK(cudaHostAlloc((void**)&h_ptr, bytes, cudaHostAllocMapped));

    // Fill host array
    for (int i = 0; i < N; ++i) {
        h_ptr[i] = static_cast<float>(i);
    }

    // Obtain device pointer that refers to the same memory
    float *d_ptr = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_ptr, (void*)h_ptr, 0));

    // Launch kernel to double the elements
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    double_elements<<<blocks, threadsPerBlock>>>(d_ptr, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // After kernel execution, the host memory should be updated
    printf("Result after kernel execution:\n");
    for (int i = 0; i < N; ++i) {
        printf("%f ", h_ptr[i]);
    }
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaFreeHost(h_ptr));
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
```