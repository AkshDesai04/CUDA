```cpp
/*
Aim of the program:
Try allocating pinned memory with the `cudaHostAllocMapped` flag.

Thinking:
The goal is to demonstrate how to allocate host memory that is both pinned (page-locked) and mapped into the device's address space using the CUDA runtime API. 
Pinned memory allows faster DMA transfers between host and device, and the mapped flag gives the device a direct pointer to the host memory, eliminating the need for explicit cudaMemcpy calls for that region.

To illustrate this:
1. Allocate an array of integers on the host using `cudaHostAlloc` with the `cudaHostAllocMapped` flag. 
2. Obtain the corresponding device pointer using `cudaHostGetDevicePointer`.
3. Fill the host array with sample data.
4. Launch a simple CUDA kernel that increments each element in the array. Because the memory is mapped, the kernel can directly read/write the host array.
5. After kernel execution, the host array will contain the updated values.
6. Print a few elements to verify the operation.
7. Clean up by freeing the pinned memory with `cudaFreeHost`.

Error handling is performed via a macro that checks CUDA return codes and prints a message if an error occurs. 
The program is self‑contained and can be compiled with `nvcc` and run on any CUDA‑capable device. 
*/
#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Simple kernel that increments each element in the array
__global__ void increment_kernel(int *d_arr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] += 1;
    }
}

int main(void)
{
    const int N = 1024;                    // Number of integers
    const size_t size = N * sizeof(int);   // Size in bytes

    int *h_ptr = NULL;   // Host pointer to pinned memory
    int *d_ptr = NULL;   // Device pointer mapped to the same memory

    // Allocate pinned, mapped host memory
    CHECK_CUDA(cudaHostAlloc((void**)&h_ptr, size, cudaHostAllocMapped));

    // Obtain the device pointer corresponding to the mapped host memory
    CHECK_CUDA(cudaHostGetDevicePointer((void**)&d_ptr, (void*)h_ptr, 0));

    // Initialize host memory
    for (int i = 0; i < N; ++i) {
        h_ptr[i] = i;
    }

    // Launch kernel to increment each element
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    increment_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_ptr, N);
    CHECK_CUDA(cudaGetLastError());  // Check kernel launch error
    CHECK_CUDA(cudaDeviceSynchronize()); // Wait for kernel to finish

    // Verify results on host
    printf("First 10 elements after kernel execution:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_ptr[i]);
    }
    printf("\n");

    // Free the pinned memory
    CHECK_CUDA(cudaFreeHost(h_ptr));

    return 0;
}
```