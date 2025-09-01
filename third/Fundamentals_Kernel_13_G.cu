```cpp
/*
Aim of the program:
Implement a kernel that copies one device array to another: `b[i] = a[i]`.

Thinking:
1. We need a CUDA kernel that takes two pointers, `a` and `b`, and an integer `N` (size of the arrays).
2. Each thread will compute its global index and, if the index is within bounds, copy the element from `a` to `b`.
3. For simplicity and clarity, we'll use a 1D grid of 1D blocks, where each block has a fixed number of threads (e.g., 256).
4. The host code will:
   - Allocate and initialize an array `h_a` on the host.
   - Allocate device memory for `d_a` and `d_b`.
   - Copy `h_a` to `d_a`.
   - Launch the kernel to copy `d_a` to `d_b`.
   - Copy `d_b` back to a host array `h_b`.
   - Verify that `h_b` matches `h_a`.
5. Error checking will be performed after each CUDA API call using a helper macro `CUDA_CHECK`.
6. The program will print a success message if all elements match, otherwise it will indicate a mismatch.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",        \
                    __FILE__, __LINE__, static_cast<int>(err),                 \
                    cudaGetErrorName(err), cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Kernel that copies array a to array b
__global__ void copyKernel(const float *a, float *b, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        b[idx] = a[idx];
    }
}

int main()
{
    const int N = 1 << 20;          // Number of elements (1M)
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    if (!h_a || !h_b) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_a = nullptr;
    float *d_b = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_a, size));
    CUDA_CHECK(cudaMalloc((void **)&d_b, size));

    // Copy host array to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    copyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, N);
    CUDA_CHECK(cudaGetLastError());  // Check for kernel launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost));

    // Verify results
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_a[i] != h_b[i]) {
            fprintf(stderr, "Mismatch at index %d: host=%f, device=%f\n",
                    i, h_a[i], h_b[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Success: All elements match.\n");
    } else {
        printf("Failure: Elements do not match.\n");
    }

    // Clean up
    free(h_a);
    free(h_b);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
```