```cuda
/*
Aim of the program:
Allocate two separate device arrays. Try to use an index that is valid for the second array on a pointer to the first array.

Thinking:
- We want to illustrate an out-of-bounds memory access on the GPU.
- Allocate two device arrays of different sizes: d_arr1 of size N1 and d_arr2 of size N2 (N2 > N1).
- Launch a kernel where each thread writes to d_arr1 at an index that ranges over 0..N2-1.
  This will cause some threads to write past the end of d_arr1, which is undefined behaviour.
- After kernel launch, we use cudaGetLastError and cudaDeviceSynchronize to capture any launch or execution errors.
- Finally, copy back the first array to host and print a few elements to see the effect.
- This demonstrates that accessing memory outside the bounds of a device array can lead to memory corruption or runtime errors.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Kernel that writes to d_arr1 using an index that may be out-of-bounds
__global__ void write_out_of_bounds(float *d_arr1, int size1, int size2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size2) {
        // Intentionally using idx as if it belongs to d_arr1
        d_arr1[idx] = (float)idx * 1.1f; // Write some value
    }
}

int main(void)
{
    const int N1 = 50;   // Size of first array
    const int N2 = 100;  // Size of second array (larger)

    float *d_arr1 = NULL;
    float *d_arr2 = NULL;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_arr1, N1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_arr2, N2 * sizeof(float)));

    // Optionally initialize d_arr2 (not strictly needed)
    CHECK_CUDA(cudaMemset(d_arr2, 0, N2 * sizeof(float)));

    // Launch kernel: 1 block of 256 threads (enough for N2)
    int threadsPerBlock = 256;
    int blocks = (N2 + threadsPerBlock - 1) / threadsPerBlock;
    write_out_of_bounds<<<blocks, threadsPerBlock>>>(d_arr1, N1, N2);

    // Check for launch errors
    CHECK_CUDA(cudaGetLastError());

    // Synchronize to catch any runtime errors
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel execution error: %s\n", cudaGetErrorString(sync_err));
    } else {
        printf("Kernel executed without reported errors (note: may have written out-of-bounds).\n");
    }

    // Copy back d_arr1 to host to inspect
    float h_arr1[N1];
    CHECK_CUDA(cudaMemcpy(h_arr1, d_arr1, N1 * sizeof(float), cudaMemcpyDeviceToHost));

    printf("First 10 elements of d_arr1 after kernel:\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_arr1[%d] = %f\n", i, h_arr1[i]);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_arr1));
    CHECK_CUDA(cudaFree(d_arr2));
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
```