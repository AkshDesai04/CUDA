```cpp
/*
Allocate and copy a much larger array, e.g., 1 million integers.

Thought process:
- The requirement is to demonstrate allocating a large array (1 million integers) on both host and device, copying it from host to device, optionally performing a simple kernel operation, then copying it back.
- For CUDA C/C++, we need to include <cuda_runtime.h> and standard headers.
- Use cudaMalloc for device memory, and malloc (or new) for host memory.
- Perform error checking after each CUDA call to catch issues.
- A simple kernel (e.g., multiply each element by 2) will show that data is indeed on the device and modified.
- After kernel execution, copy the array back to host with cudaMemcpy and verify a few elements.
- Finally, free device and host memory and exit.

The code below follows this plan and is self-contained. It compiles as a .cu file and can be built with nvcc. */

#include <cstdio>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call or kernel launch
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                     \
    } while (0)

// Kernel that multiplies each element by 2
__global__ void multiplyByTwo(int *d_array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_array[idx] *= 2;
    }
}

int main(void) {
    const int N = 1'000'000;            // 1 million integers
    const size_t size = N * sizeof(int);

    // Allocate host memory
    int *h_array = (int *)malloc(size);
    if (h_array == nullptr) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        h_array[i] = i;
    }

    // Allocate device memory
    int *d_array = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&d_array, size));

    // Copy from host to device
    CHECK_CUDA(cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice));

    // Launch kernel to multiply each element by 2
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    multiplyByTwo<<<blocksPerGrid, threadsPerBlock>>>(d_array, N);
    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost));

    // Verify a few elements
    printf("Sample results after kernel execution:\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_array[%d] = %d\n", i, h_array[i]); // Should be 0, 2, 4, ..., 18
    }
    printf("...\n");
    printf("h_array[%d] = %d\n", N-1, h_array[N-1]); // Should be (N-1)*2

    // Clean up
    CHECK_CUDA(cudaFree(d_array));
    free(h_array);

    // Reset device (optional)
    CHECK_CUDA(cudaDeviceReset());

    return EXIT_SUCCESS;
}
```