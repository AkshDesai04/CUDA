```cuda
/*
Aim: Allocate memory, launch a kernel that uses it, synchronize, and then free the memory.

Thinking:
1. Use CUDA C language (.cu file) with standard headers: <stdio.h> and <cuda_runtime.h>.
2. Define a simple kernel that writes sequential indices into a device array:
   __global__ void initArray(int *d_arr, int N) { int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx < N) d_arr[idx] = idx; }
3. In main:
   a. Set a small array size N (e.g., 10).
   b. Allocate host array for later verification (optional).
   c. Allocate device memory with cudaMalloc.
   d. Launch kernel with enough blocks/threads to cover N.
   e. Synchronize with cudaDeviceSynchronize to ensure kernel completion.
   f. Optionally copy data back to host using cudaMemcpy to verify results.
   g. Free device memory with cudaFree.
   h. Release host memory if allocated.
4. Include error-checking macro to wrap CUDA API calls for clarity.
5. Keep code concise yet functional, compile with nvcc.

The program demonstrates the basic memory lifecycle in CUDA: allocation, usage in a kernel, synchronization, and deallocation.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Simple kernel that writes sequential values into the array
__global__ void initArray(int *d_arr, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        d_arr[idx] = idx;
}

int main(void)
{
    const int N = 10;                         // Size of the array
    const size_t size = N * sizeof(int);      // Size in bytes

    // Host array for verification
    int h_arr[N];

    // Device pointer
    int *d_arr = NULL;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_arr, size));

    // Launch kernel: 2 blocks of 5 threads each (covers N=10)
    int threadsPerBlock = 5;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    initArray<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    // Synchronize to ensure kernel completion
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host for verification
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost));

    // Print the result
    printf("Device array contents after kernel execution:\n");
    for (int i = 0; i < N; ++i)
        printf("%d ", h_arr[i]);
    printf("\n");

    // Free device memory
    CUDA_CHECK(cudaFree(d_arr));

    // Reset the device (optional but good practice)
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
```