/*
Store the grid and block dimensions in `dim3` variables on the host before launching the kernel. Launch the kernel using these variables, e.g., `myKernel<<<gridDim, blockDim>>>();`. This is the standard way to configure a launch.

The program below demonstrates this by allocating an array on the device, launching a kernel that increments each element, and then copying the results back to the host. The key point is that the grid and block dimensions are stored in `dim3` variables (`gridDim` and `blockDim`) on the host side before the kernel launch, following the standard CUDA launch syntax.

The kernel performs a simple addition to each element. We compute the total number of threads needed to cover the array and derive the grid size accordingly. After the kernel runs, we check the first few elements to confirm the operation succeeded. Error checking and resource cleanup are performed in a straightforward manner.

This example is minimal yet complete, illustrating how to use `dim3` for launch configuration and ensuring that the program can be compiled and executed with a standard CUDA compiler (`nvcc`).
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void myKernel(int *data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        data[idx] += 1;  // Simple operation: increment each element
    }
}

int main()
{
    const int N = 1 << 20;            // 1,048,576 elements
    const size_t size = N * sizeof(int);

    // Allocate and initialize host memory
    int *h_data = (int *)malloc(size);
    if (!h_data)
    {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i)
    {
        h_data[i] = i;
    }

    // Allocate device memory
    int *d_data;
    cudaError_t err = cudaMalloc((void **)&d_data, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_data);
        return EXIT_FAILURE;
    }

    // Copy data from host to device
    err = cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        free(h_data);
        return EXIT_FAILURE;
    }

    // Define block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    dim3 blockDim(threadsPerBlock);  // dim3 variable for block dimensions
    dim3 gridDim(blocksPerGrid);     // dim3 variable for grid dimensions

    // Launch the kernel using the dim3 variables
    myKernel<<<gridDim, blockDim>>>(d_data, N);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        free(h_data);
        return EXIT_FAILURE;
    }

    // Wait for GPU to finish before accessing on host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        free(h_data);
        return EXIT_FAILURE;
    }

    // Copy the result back to host
    err = cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        free(h_data);
        return EXIT_FAILURE;
    }

    // Verify a few results
    printf("First 10 results after increment:\n");
    for (int i = 0; i < 10; ++i)
    {
        printf("h_data[%d] = %d\n", i, h_data[i]);
    }

    // Clean up
    cudaFree(d_data);
    free(h_data);

    return EXIT_SUCCESS;
}
