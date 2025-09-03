```cpp
/*
Allocate memory with `cudaMallocManaged`. Now try to access this pointer from the host. Does it work? Why?

The aim of this program is to demonstrate the behavior of CUDA unified memory (managed memory) when allocated with `cudaMallocManaged`. Unified memory allows a single pointer to be accessed from both the host (CPU) and the device (GPU) without explicit data transfers. This means that after allocating memory with `cudaMallocManaged`, the host can read from and write to that memory directly, and the GPU can also read/write it within kernels.

To test this, the program performs the following steps:
1. Allocate an array of integers using `cudaMallocManaged`.
2. Initialize the array on the host side.
3. Launch a simple CUDA kernel that doubles each element of the array.
4. After the kernel execution, the host reads back the modified data and prints it.
5. The program also prints the device ordinal to show that the memory is accessible on that device.

The key points to understand:
- Unified memory allocation (`cudaMallocManaged`) gives the host a valid pointer to a memory region that the GPU can also access. The driver handles page migrations automatically when the memory is accessed from one side or the other.
- Accessing the pointer from the host immediately after allocation works because the allocation is synchronized; the pointer refers to memory that the host can use. The actual memory resides in host memory initially, but the driver may migrate pages to device memory upon device access. This migration is transparent.
- The program includes error checks after CUDA API calls to ensure proper behavior.

The result should show that the host can write to and read from the unified memory, that the GPU kernel successfully updates the data, and that the host sees the updated values after the kernel finishes. This confirms that `cudaMallocManaged` indeed creates memory accessible from both host and device.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Kernel that doubles each element of the array
__global__ void doubleElements(int* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        data[idx] *= 2;
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

int main()
{
    const int N = 10;
    int *managedPtr = nullptr;

    // Allocate unified memory
    CUDA_CHECK(cudaMallocManaged(&managedPtr, N * sizeof(int)));

    // Initialize from host
    for (int i = 0; i < N; ++i)
    {
        managedPtr[i] = i + 1;  // Fill with 1,2,3,...
    }

    // Print initial data
    printf("Initial data (host view):\n");
    for (int i = 0; i < N; ++i)
    {
        printf("%d ", managedPtr[i]);
    }
    printf("\n");

    // Launch kernel to double each element
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    doubleElements<<<blocks, threadsPerBlock>>>(managedPtr, N);
    CUDA_CHECK(cudaGetLastError());  // Check for launch errors

    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Print data after kernel execution
    printf("Data after kernel (host view):\n");
    for (int i = 0; i < N; ++i)
    {
        printf("%d ", managedPtr[i]);
    }
    printf("\n");

    // Free unified memory
    CUDA_CHECK(cudaFree(managedPtr));

    return 0;
}
```