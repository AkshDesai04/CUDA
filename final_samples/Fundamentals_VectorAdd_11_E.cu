```c
/*
Aim of the program:
Implement a vector squaring kernel using a grid-stride loop.

Thinking:
1. The program is a self-contained CUDA C (C++-style) application that demonstrates how to square each element of a vector on the GPU.
2. It uses a grid-stride loop in the kernel so that the code works correctly regardless of the number of threads per block or the size of the vector.
3. The workflow is:
   - Allocate and initialize a host vector with some test data.
   - Allocate a device vector of the same size.
   - Copy the data from host to device.
   - Launch the kernel with an appropriate number of blocks and threads per block.
   - The kernel processes the vector using the pattern:
        for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < N;
             idx += gridDim.x * blockDim.x)
          d_arr[idx] = d_arr[idx] * d_arr[idx];
     This ensures every element is processed and works for arbitrary grid dimensions.
   - Copy the result back to the host.
   - Verify the result by comparing a few elements or computing a checksum.
   - Measure execution time using cudaEvent_t to show the GPU performance.
4. Error checking is performed after each CUDA API call using a macro that prints an error message and exits if the call fails.
5. The program compiles with `nvcc` and can be run on any machine with an NVIDIA GPU that supports the required compute capability.

The final code includes:
- Necessary headers (`cuda_runtime.h`, `stdio.h`, `stdlib.h`, `assert.h`).
- An error-checking macro `CUDA_CHECK`.
- The kernel `squareKernel`.
- The `main` function that orchestrates memory allocation, data transfer, kernel launch, result verification, and cleanup.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel that squares each element of the array using a grid-stride loop
__global__ void squareKernel(float* d_arr, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride)
    {
        float val = d_arr[i];
        d_arr[i] = val * val;
    }
}

int main(void)
{
    const size_t N = 1 << 20;          // 1,048,576 elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float* h_arr = (float*)malloc(bytes);
    if (h_arr == NULL)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array with values 1.0f, 2.0f, ..., N
    for (size_t i = 0; i < N; ++i)
    {
        h_arr[i] = (float)(i + 1);
    }

    // Allocate device memory
    float* d_arr = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_arr, bytes));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start event
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch kernel
    squareKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop, 0));

    // Wait for the kernel to finish
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Compute elapsed time
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    printf("Kernel execution time: %f ms\n", ms);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, bytes, cudaMemcpyDeviceToHost));

    // Verify a few elements
    bool ok = true;
    for (size_t i = 0; i < 10; ++i)
    {
        float expected = (float)(i + 1) * (float)(i + 1);
        if (fabs(h_arr[i] - expected) > 1e-5)
        {
            printf("Mismatch at index %zu: got %f, expected %f\n", i, h_arr[i], expected);
            ok = false;
            break;
        }
    }

    if (ok)
        printf("Result verification PASSED.\n");
    else
        printf("Result verification FAILED.\n");

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_arr));
    free(h_arr);

    return EXIT_SUCCESS;
}
```