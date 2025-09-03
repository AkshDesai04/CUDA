```cpp
/*
Wrap the `cudaMalloc` calls.

The aim of this program is to demonstrate how to wrap the `cudaMalloc` CUDA runtime API call with a custom wrapper that centralises error handling and simplifies the allocation process. By encapsulating `cudaMalloc` inside a helper function (or template) we can:

1. Reduce boilerplate error-checking code throughout the program.
2. Provide a single point to modify allocation behaviour (e.g., logging, debugging, or custom allocation strategies).
3. Make the main code more readable by hiding the verbose CUDA error handling logic.

In this example, a simple function `cudaMallocWrap` is implemented. It takes a pointer to a device pointer and the size to allocate. It calls `cudaMalloc`, checks the returned `cudaError_t`, and prints an informative error message if the allocation fails. The function returns the error code so callers can react accordingly.

The program allocates an array of 10 floats, initializes them on the device via a tiny kernel, copies the data back to the host, prints the results, and finally frees the device memory using `cudaFree`. This demonstrates both the wrapper and typical allocation/deallocation flow.

The code is fully self‑contained and compiles as a CUDA C++ source file (`.cu`). It uses the CUDA runtime API and standard C++ I/O for demonstration purposes.

Important points covered:
- Wrapper function with error checking.
- Example usage in `main`.
- Simple kernel to confirm allocation works.
- Proper cleanup with `cudaFree`.
*/

#include <cuda_runtime.h>
#include <stdio.h>

// Wrapper around cudaMalloc that handles errors and prints a message.
cudaError_t cudaMallocWrap(void** devPtr, size_t size)
{
    cudaError_t err = cudaMalloc(devPtr, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s (size=%zu)\n", cudaGetErrorString(err), size);
    }
    return err;
}

// Simple kernel to initialize a float array on the device.
__global__ void initKernel(float* arr, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        arr[idx] = (float)idx;
}

int main()
{
    const int N = 10;
    float* d_array = nullptr;
    size_t bytes = N * sizeof(float);

    // Allocate device memory using the wrapper.
    if (cudaMallocWrap((void**)&d_array, bytes) != cudaSuccess)
    {
        return EXIT_FAILURE;
    }

    // Launch kernel to initialize array.
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    initKernel<<<blocksPerGrid, threadsPerBlock>>>(d_array, N);

    // Check for kernel launch errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_array);
        return EXIT_FAILURE;
    }

    // Copy results back to host.
    float h_array[N];
    err = cudaMemcpy(h_array, d_array, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_array);
        return EXIT_FAILURE;
    }

    // Print the array.
    printf("Device array contents:\n");
    for (int i = 0; i < N; ++i)
        printf("%f ", h_array[i]);
    printf("\n");

    // Clean up.
    cudaFree(d_array);
    return EXIT_SUCCESS;
}
```