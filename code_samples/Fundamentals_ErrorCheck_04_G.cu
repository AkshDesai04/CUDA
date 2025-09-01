```cpp
/*
Aim: To provide a minimal CUDA C program that demonstrates kernel launch and error handling.
The user asked: "Does this error occur synchronously or asynchronously?"
Answer: Many CUDA runtime errors, especially kernel launch failures, are reported asynchronously.
They are not immediately visible after the launch call. To detect such errors, one must
invoke cudaGetLastError() or perform a cudaDeviceSynchronize() which forces the device
to finish execution and propagates any pending errors. Some errors, such as invalid API
parameter checks, are reported synchronously.
Below is a simple program that allocates an array, launches a kernel to increment its
elements, and checks for errors synchronously after calling cudaGetLastError().
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that increments each element by 1
__global__ void increment(int *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        data[idx] += 1;
    }
}

int main(void)
{
    const int N = 1024;
    const int SIZE = N * sizeof(int);

    int *h_data = (int *)malloc(SIZE);
    int *d_data = NULL;

    // Initialize host data
    for (int i = 0; i < N; ++i)
    {
        h_data[i] = i;
    }

    // Allocate device memory
    cudaError_t err = cudaMalloc((void **)&d_data, SIZE);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy data to device
    err = cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    increment<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);

    // Check for launch errors (synchronous after cudaGetLastError)
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Synchronize to catch any asynchronous errors that may have occurred during execution
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Copy result back to host
    err = cudaMemcpy(h_data, d_data, SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Verify results
    for (int i = 0; i < N; ++i)
    {
        if (h_data[i] != i + 1)
        {
            fprintf(stderr, "Data mismatch at index %d: expected %d, got %d\n", i, i + 1, h_data[i]);
            cudaFree(d_data);
            return 1;
        }
    }

    printf("All data verified successfully.\n");

    // Clean up
    cudaFree(d_data);
    free(h_data);

    return 0;
}
```