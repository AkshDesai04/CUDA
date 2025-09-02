/*
Launch your grid-stride kernel with a number of blocks equal to the number of SMs on your GPU, and a reasonable block size (e.g. 128 or 256). This is a common performance heuristic.

Thinking:
- The goal is to demonstrate how to obtain the number of streaming multiprocessors (SMs) on the device using cudaDeviceGetAttribute.
- We then launch a simple CUDA kernel that uses the grid‑stride loop pattern. The kernel will operate on an array, for example incrementing each element.
- The launch configuration uses the SM count as the grid dimension and a fixed block size (128 or 256 threads per block). This aligns with common performance heuristics.
- The program allocates host memory, copies data to the device, launches the kernel, copies the result back, and verifies correctness.
- Error checking is included for CUDA API calls and kernel launches.
- All code is contained within a single .cu file, ready to compile with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple error checking macro
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Kernel: increment each element by 1 using grid-stride loop
__global__ void increment_kernel(int *data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < N; i += stride)
    {
        data[i] += 1;
    }
}

int main()
{
    // Problem size
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(int);

    // Allocate host memory
    int *h_data = (int *)malloc(size);
    if (!h_data)
    {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host data
    for (int i = 0; i < N; ++i)
        h_data[i] = i;

    // Allocate device memory
    int *d_data = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_data, size));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    // Determine number of SMs
    int device = 0;
    int smCount = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
    printf("Device %d has %d SMs\n", device, smCount);

    // Set block size and grid size
    const int blockSize = 128; // reasonable block size
    int gridSize = smCount;    // one block per SM

    // Launch kernel
    increment_kernel<<<gridSize, blockSize>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // Verify correctness
    bool success = true;
    for (int i = 0; i < N; ++i)
    {
        if (h_data[i] != i + 1)
        {
            fprintf(stderr, "Verification failed at index %d: expected %d, got %d\n",
                    i, i + 1, h_data[i]);
            success = false;
            break;
        }
    }
    if (success)
        printf("All values verified successfully.\n");

    // Clean up
    free(h_data);
    CUDA_CHECK(cudaFree(d_data));

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
