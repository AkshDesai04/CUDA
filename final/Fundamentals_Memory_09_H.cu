```cpp
/*
Aim of the program:
Try allocating pinned memory with the `cudaHostAllocMapped` flag.

Thinking:
- The goal is to demonstrate how to allocate host memory that is pinned (page-locked) and also mapped into the device address space using the cudaHostAllocMapped flag.
- When memory is allocated with this flag, CUDA provides a pointer that can be accessed from both the host (CPU) and the device (GPU). The host can read/write directly, while the device can obtain a device pointer via cudaHostGetDevicePointer.
- This example will:
  1. Allocate an array of integers on the host with cudaHostAllocMapped.
  2. Fill the array on the host.
  3. Obtain the corresponding device pointer.
  4. Launch a simple kernel that increments each element.
  5. Copy the results back to the host (directly possible due to mapping).
  6. Print the results to verify correctness.
  7. Clean up resources.
- Error checking will be performed after each CUDA call.
- The program is self-contained and compiles as a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 10

// Simple kernel that adds 1 to each element
__global__ void addOne(int *d_data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        d_data[idx] += 1;
    }
}

// Helper macro for error checking
#define CUDA_CHECK(call)                                                   \
    {                                                                      \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess)                                            \
        {                                                                  \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    }

int main()
{
    int *h_data = NULL;
    int *d_data = NULL;
    size_t bytes = N * sizeof(int);

    // Allocate pinned, mapped host memory
    CUDA_CHECK(cudaHostAlloc((void**)&h_data, bytes, cudaHostAllocMapped));

    // Initialize host data
    for (int i = 0; i < N; ++i)
    {
        h_data[i] = i;
    }

    // Obtain device pointer corresponding to the mapped host memory
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_data, (void*)h_data, 0));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addOne<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Since the memory is mapped, h_data now contains the updated values
    printf("Result after kernel execution:\n");
    for (int i = 0; i < N; ++i)
    {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaFreeHost(h_data));
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
```