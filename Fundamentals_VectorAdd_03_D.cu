```cuda
/*
Aim of the program:
Remove the `if (index < N)` check and run the program for a non-multiple size with `cuda-memcheck`. Observe the out-of-bounds access errors.

Thinking:
1. We want to demonstrate how omitting a bounds check in a CUDA kernel can cause out-of-bounds memory accesses, which will be detected by `cuda-memcheck`.
2. Choose a size `N` that is not a multiple of the block size to ensure that the grid has an incomplete block that will launch threads with indices beyond `N-1`. For example, `N = 500` with a block size of `256` will create `2` blocks: the second block will contain threads with indices `256` to `511`, of which indices `500` to `511` are out-of-bounds.
3. In the kernel, we simply write the thread index to the output array without any bounds checking. This will write to memory beyond the allocated array for the out-of-bounds indices.
4. In the host code, allocate an array of size `N` on the device, launch the kernel, copy the results back, and print a few values to show that the program runs but `cuda-memcheck` will flag the out-of-bounds writes.
5. Compile with `nvcc -arch=sm_70 -o out_of_bounds out_of_bounds.cu` and run with `cuda-memcheck ./out_of_bounds` to see the reported errors.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void writeIndices(int *arr, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Intentionally no bounds check
    arr[idx] = idx;
}

int main()
{
    const int N = 500;               // Non-multiple of block size
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize; // 2

    int *d_arr = NULL;
    cudaError_t err = cudaMalloc((void**)&d_arr, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch kernel without bounds check
    writeIndices<<<gridSize, blockSize>>>(d_arr, N);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    // Copy result back to host
    int *h_arr = (int*)malloc(N * sizeof(int));
    if (!h_arr) {
        fprintf(stderr, "Host allocation failed\n");
        cudaFree(d_arr);
        return 1;
    }

    err = cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        free(h_arr);
        cudaFree(d_arr);
        return 1;
    }

    // Print first 10 values
    printf("First 10 values:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Clean up
    free(h_arr);
    cudaFree(d_arr);

    return 0;
}
```