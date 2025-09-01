```cuda
/*
Aim of the program:
Generalize the kernel to work on arrays. It should take `int* a`, `int* b`, and `int N`. Each thread `i` should compute `b[i] = a[i] + 10;`.

Thinking process:
1. The task requires a CUDA kernel that operates on whole arrays, not just a single element.
2. We will pass pointers to device memory `int* a` and `int* b` along with the length `N` to the kernel.
3. Inside the kernel, each thread will calculate its global index `i = blockIdx.x * blockDim.x + threadIdx.x`.
4. If `i < N`, the thread will read `a[i]`, add 10, and write the result to `b[i]`.
5. In the host code:
   - Allocate host arrays for input and output.
   - Initialize input with sample data.
   - Allocate device memory for `a` and `b`.
   - Copy input array from host to device.
   - Determine grid and block dimensions; a common choice is 256 threads per block.
   - Launch the kernel.
   - Copy the result back to the host.
   - Print the results to verify correctness.
   - Clean up device and host memory.
6. Add basic error checking after CUDA API calls using a macro to keep the code readable.
7. Include necessary headers: `<cstdio>` for I/O and `<cuda_runtime.h>` for CUDA runtime.
8. Compile with `nvcc -o array_add array_add.cu` and run the executable.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Kernel that adds 10 to each element of array a and stores in array b
__global__ void add_ten_kernel(const int* a, int* b, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        b[i] = a[i] + 10;
    }
}

int main(void) {
    const int N = 1024;               // Size of arrays
    size_t size = N * sizeof(int);

    // Allocate host memory
    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);

    // Initialize input array
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;  // Example data: 0,1,2,...
    }

    // Allocate device memory
    int *d_a = nullptr;
    int *d_b = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_a, size));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    add_ten_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, N);
    CUDA_CHECK(cudaGetLastError());  // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost));

    // Verify results (print first 10 elements)
    printf("First 10 results (a[i] + 10):\n");
    for (int i = 0; i < 10; ++i) {
        printf("b[%d] = %d\n", i, h_b[i]);
    }

    // Clean up
    free(h_a);
    free(h_b);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    return 0;
}
```