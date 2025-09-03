```cpp
/*
 * Aim of the program (as requested in the prompt):
 * Can tools like Valgrind detect CUDA device memory leaks? (Not directly, but `cuda-memcheck --leak-check` can).
 *
 * ------------------------------------------------------------------
 * My entire thinking:
 * 
 * 1. The prompt asks for a CUDA program that illustrates the point that Valgrind cannot detect
 *    device memory leaks, while CUDA's own memcheck tool (`cuda-memcheck --leak-check`)
 *    can. 
 * 
 * 2. The simplest way to demonstrate a device memory leak is to allocate memory on the GPU
 *    (using `cudaMalloc`) and never free it.  When we run the program under `cuda-memcheck
 *    --leak-check`, it should report the allocation as leaked.  Valgrind, on the other hand,
 *    will only report host memory leaks; it has no visibility into device memory.
 * 
 * 3. For the program itself, I will:
 *    - Include the necessary headers.
 *    - Write a trivial kernel that performs a simple operation (e.g., add two arrays).
 *    - In `main`, allocate two device arrays, launch the kernel, and *intentionally
 *      omit the `cudaFree` calls*.
 *    - Print a message that the kernel has finished.
 * 
 * 4. Users can compile with `nvcc` and then run under `cuda-memcheck --leak-check=full ./a.out`
 *    to see the leak report.  Running the same binary under Valgrind will produce no
 *    device‑memory leak information.
 * 
 * 5. The code below follows best practices: it checks the return status of CUDA calls,
 *    prints error messages if something goes wrong, and keeps the example short
 *    and self‑contained.
 * 
 * 6. Note: The program intentionally leaks device memory; do not run it in a
 *    production environment.  It's purely for educational demonstration.
 * 
 * 7. The output of `cuda-memcheck` will look something like:
 *    ```
 *    ----------- CUDA-MEMCHECK: Device memory leak check -----------
 *    All allocations were freed
 *    Unrecovered device memory leaks:
 *    1 allocation(s) not freed; 16 bytes leaked.
 *    ```
 *    while Valgrind will not report these device memory leaks.
 * ------------------------------------------------------------------
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Helper macro for checking CUDA errors
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// A simple kernel that adds two arrays
__global__ void add_arrays(const float *a, const float *b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void)
{
    const int N = 1 << 20;  // 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(N - i);
    }

    // Device pointers
    float *d_a, *d_b, *d_c;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_a, size));
    CUDA_CHECK(cudaMalloc((void **)&d_b, size));
    CUDA_CHECK(cudaMalloc((void **)&d_c, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add_arrays<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify a few elements
    printf("h_c[0] = %f, h_c[%d] = %f\n", h_c[0], N-1, h_c[N-1]);

    // Intentionally omit cudaFree for d_a, d_b, d_c to create a device memory leak
    // For demonstration purposes only.  This is where Valgrind would not see the leak,
    // but cuda-memcheck would report it.

    // Clean up host memory
    free(h_a);
    free(h_b);
    free(h_c);

    // Explicitly synchronize to ensure all device work is finished before exit
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Program finished.\n");
    return 0;
}
```