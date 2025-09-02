/*
Aim of the program:
After allocating with `cudaMalloc`, use `cudaMemset` to set all the bytes of the device array to 0 before copying the host data over.

Thinking:
The goal is to demonstrate the proper use of `cudaMalloc` followed by `cudaMemset` to initialize device memory to zero, and then copying host data onto that memory using `cudaMemcpy`.  
1. Allocate a small host array and fill it with known values.  
2. Allocate a device array of the same size with `cudaMalloc`.  
3. Immediately call `cudaMemset` on the device pointer to set every byte to zero.  
   - `cudaMemset` operates on raw bytes, so for an array of floats the number of bytes to set is `N * sizeof(float)`.  
   - Setting to zero ensures the device memory is clean before any host data is written.  
4. Use `cudaMemcpy` to copy the host data into the device array.  
   - The data will overwrite the zeroed memory.  
5. Copy the data back to another host array to verify that the copy succeeded.  
6. Print a few elements to confirm that the values match the original host array.  
7. Clean up all allocated memory.  

This simple program shows the sequence: `cudaMalloc` → `cudaMemset` → `cudaMemcpy`. It includes basic error checking after each CUDA call.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",     \
                    #call, __FILE__, __LINE__,                   \
                    cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

int main(void)
{
    const int N = 1024;
    size_t size = N * sizeof(float);

    // Host allocation and initialization
    float *h_src = (float*)malloc(size);
    float *h_dst = (float*)malloc(size);
    if (!h_src || !h_dst) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
        h_src[i] = (float)i; // simple pattern
    }

    // Device allocation
    float *d_arr = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_arr, size));

    // Set device memory to zero
    CHECK_CUDA(cudaMemset(d_arr, 0, size));

    // Copy host data to device
    CHECK_CUDA(cudaMemcpy(d_arr, h_src, size, cudaMemcpyHostToDevice));

    // Copy device data back to host for verification
    CHECK_CUDA(cudaMemcpy(h_dst, d_arr, size, cudaMemcpyDeviceToHost));

    // Verify a few elements
    printf("First 10 elements after copy back:\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_dst[%d] = %f\n", i, h_dst[i]);
    }

    // Cleanup
    free(h_src);
    free(h_dst);
    CHECK_CUDA(cudaFree(d_arr));

    // Reset device
    CHECK_CUDA(cudaDeviceReset());

    return EXIT_SUCCESS;
}
