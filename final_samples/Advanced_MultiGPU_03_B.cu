```cpp
/*
Write a kernel running on GPU 0 that takes a pointer to memory allocated on GPU 1 and accesses it directly (this requires Unified Virtual Addressing).

Thinking:
1. Unified Virtual Addressing (UVA) allows a single virtual address space to be shared across multiple GPUs. If UVA is enabled, a pointer returned by cudaMalloc on one GPU can be used on another GPU without explicit data transfer.
2. To demonstrate this, the program will:
   - Detect at least two GPUs.
   - Allocate an array on GPU 1.
   - Initialize the array (e.g., with zeros).
   - Switch context to GPU 0 and launch a kernel that increments each element of the array.
   - Switch back to GPU 1, copy the data back to host memory, and verify the result.
3. CUDA runtime API functions `cudaSetDevice`, `cudaMalloc`, `cudaMemcpy`, and kernel launch syntax are used.
4. Error checking is performed after each CUDA call to catch and report failures.
5. The kernel uses a grid-stride loop to cover all elements, ensuring it works regardless of the number of threads.
6. The program is self-contained; compile with `nvcc -arch=sm_30` (or higher) to target a GPU that supports UVA.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Error checking macro
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                   \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Kernel that increments each element of an array
__global__ void incrementKernel(int* arr, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < N; i += stride) {
        arr[i] += 1;
    }
}

int main()
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount < 2) {
        fprintf(stderr, "Error: At least two GPUs are required.\n");
        return EXIT_FAILURE;
    }

    const int N = 1024;               // Number of integers
    int *d_arr_gpu1 = nullptr;        // Pointer to memory allocated on GPU 1

    // Allocate memory on GPU 1
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMalloc(&d_arr_gpu1, N * sizeof(int)));

    // Initialize the array on GPU 1 (set all to 0)
    CUDA_CHECK(cudaMemset(d_arr_gpu1, 0, N * sizeof(int)));

    // Launch kernel on GPU 0 that accesses memory on GPU 1
    CUDA_CHECK(cudaSetDevice(0));
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    incrementKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr_gpu1, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host from GPU 1 for verification
    int *h_arr = (int*)malloc(N * sizeof(int));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr_gpu1, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify that each element was incremented by 1
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_arr[i] != 1) {
            fprintf(stderr, "Error at index %d: expected 1, got %d\n", i, h_arr[i]);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Success: all elements incremented correctly.\n");
    }

    // Clean up
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaFree(d_arr_gpu1));
    free(h_arr);

    return EXIT_SUCCESS;
}
```