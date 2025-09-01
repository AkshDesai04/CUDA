```cpp
/* 
Wrap the `cudaMalloc` calls.

My thinking:

- The goal is to provide a safe wrapper around `cudaMalloc` so that error handling is centralized and repeated error checks are avoided throughout the code.
- I want a function that attempts to allocate memory on the device, checks the returned `cudaError_t`, prints a descriptive error message if allocation fails, and exits the program. 
- To keep it generic, I will use a `void*` return type and cast the pointer in the caller. Additionally, I will provide a templated helper that returns a typed pointer (e.g., `int*` or `float*`) for convenience.
- In the example program I will:
  1. Allocate an array of integers on the device using the wrapper.
  2. Allocate a corresponding array on the host.
  3. Initialize the host array, copy it to the device, launch a simple kernel that multiplies each element by 2, copy the result back, and print it.
  4. Free all allocated memory.
- The kernel and memory copy calls will be straightforward; the focus is on the wrapper.
- I will include necessary headers (`<cstdio>`, `<cuda_runtime.h>`) and use `cudaDeviceSynchronize()` for simplicity.
- The wrapper should be self-contained so that it can be reused in larger projects.
- Finally, I will put everything in a single `.cu` file, as requested, and ensure it compiles with `nvcc`.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/* Wrapper for cudaMalloc that checks for errors and exits on failure. */
void* safeCudaMalloc(size_t size)
{
    void* devPtr = nullptr;
    cudaError_t err = cudaMalloc(&devPtr, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed (%zu bytes): %s\n",
                size, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return devPtr;
}

/* Templated helper to allocate typed arrays on the device. */
template <typename T>
T* cudaMallocTyped(size_t n)
{
    size_t bytes = n * sizeof(T);
    return static_cast<T*>(safeCudaMalloc(bytes));
}

/* Simple kernel that multiplies each element by a factor. */
__global__ void multiplyKernel(int* d_arr, int n, int factor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        d_arr[idx] *= factor;
    }
}

int main()
{
    const int N = 10;
    const int factor = 2;

    /* Allocate host array and initialize. */
    int h_arr[N];
    for (int i = 0; i < N; ++i)
        h_arr[i] = i;

    /* Allocate device array using the wrapper. */
    int* d_arr = cudaMallocTyped<int>(N);

    /* Copy data from host to device. */
    cudaError_t err = cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        exit(EXIT_FAILURE);
    }

    /* Launch kernel: one block with enough threads. */
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    multiplyKernel<<<blocks, threadsPerBlock>>>(d_arr, N, factor);

    /* Check for kernel launch errors. */
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        exit(EXIT_FAILURE);
    }

    /* Synchronize to ensure kernel completion. */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        exit(EXIT_FAILURE);
    }

    /* Copy result back to host. */
    err = cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        exit(EXIT_FAILURE);
    }

    /* Print the results. */
    printf("Result after multiplying by %d:\n", factor);
    for (int i = 0; i < N; ++i)
        printf("%d ", h_arr[i]);
    printf("\n");

    /* Free device memory. */
    cudaFree(d_arr);

    return 0;
}
```