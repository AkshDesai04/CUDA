/*
Aim of the program: In a loop, allocate memory and overwrite the pointer variable in the next iteration. Explain why this is a leak.

Thinking:
1. The program is written in CUDA C (.cu file) to illustrate the memory leak scenario.
2. Inside the host code, a pointer variable (d_ptr) is used to hold a device memory allocation.
3. In a for-loop, cudaMalloc() is called each iteration to allocate a new chunk of device memory and the resulting device pointer is stored in d_ptr.
4. After each allocation, the pointer is immediately overwritten by the next cudaMalloc() call. This discards the previous pointer value.
5. Since cudaFree() is never called on the previously allocated memory (except possibly the last allocation if freed at the end), the memory that was allocated in earlier iterations remains allocated but no longer accessible – that is a memory leak.
6. The code demonstrates that freeing only the last allocation is insufficient; every allocation must be freed.
7. The kernel in this example does nothing, it's just there to keep the file complete.
8. The comments in the code and this explanatory comment explain the cause of the leak.
*/

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void dummyKernel(int *data, int size)
{
    // This kernel does nothing but could be used to verify memory access.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        data[idx] = idx;
}

int main()
{
    int *d_ptr = NULL;
    const int allocSize = 1024 * sizeof(int); // 4 KB per allocation

    // Allocate memory in a loop, overwriting the pointer each time.
    for (int i = 0; i < 5; ++i)
    {
        cudaError_t err = cudaMalloc((void**)&d_ptr, allocSize);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc failed at iteration %d: %s\n", i, cudaGetErrorString(err));
            // If allocation fails, exit immediately.
            return -1;
        }

        // Optionally launch a kernel (not necessary for the leak demonstration).
        // Launch one thread block with one thread to keep it simple.
        dummyKernel<<<1, 1>>>(d_ptr, 1024);
        cudaDeviceSynchronize();

        // At the end of this iteration, d_ptr will be overwritten
        // by the next cudaMalloc() call. The previously allocated
        // memory is no longer referenced.
    }

    // Only the last allocation is freed here; the previous four
    // allocations remain allocated and are inaccessible – a memory leak.
    cudaFree(d_ptr);

    // Note: In a real application, you would free all allocated memory
    // or use a memory pool / tracking mechanism to avoid leaks.

    printf("Completed allocations and freed the last one.\n");
    return 0;
}