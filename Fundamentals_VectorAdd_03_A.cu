```cuda
/*
Test your program with `N = 1` and a block size of 256. Does it work correctly?

This program demonstrates a simple CUDA kernel that doubles each element of an
array. The aim is to verify that the kernel behaves correctly even when the
block size (number of threads per block) exceeds the problem size (N).
In this test, N is set to 1 and the block size is 256, so most threads will
execute the bounds check and simply return without performing any computation.
The kernel uses an explicit boundary guard (`if (idx < N)`) to prevent out-of-bounds
memory accesses, which is crucial when the number of launched threads is larger
than the number of data elements. After the kernel completes, the host copies the
result back to CPU memory and prints it. If the program outputs the correct
result (the single element multiplied by 2), we can conclude that it works
correctly under the given conditions.
*/

#include <stdio.h>
#include <cuda.h>

__global__ void doubleArray(float *d_arr, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        d_arr[idx] *= 2.0f;
    }
}

int main()
{
    const int N = 1;           // Problem size
    const int blockSize = 256; // Threads per block
    const int gridSize = (N + blockSize - 1) / blockSize;

    float h_arr[N];
    h_arr[0] = 1.5f; // Example value

    float *d_arr = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_arr, N * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(d_arr, h_arr, N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    doubleArray<<<gridSize, blockSize>>>(d_arr, N);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    err = cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    printf("Result after doubling: %f\n", h_arr[0]);

    cudaFree(d_arr);
    return 0;
}
```