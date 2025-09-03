```cpp
/*
Aim of the program:
Create a `__device__` function that is recursive, like a factorial function. Try calling it from a kernel. Be aware of stack limitations.

Thinking:
- CUDA supports device-side recursion starting with compute capability 2.0.  
- Each thread has a private stack; the default size is 1024 bytes, which can be increased with cudaDeviceSetLimit(cudaLimitStackSize, size).  
- Recursive factorial is simple but each recursive call consumes a small amount of stack.  
- To keep stack usage low, we limit the factorial to small arguments (e.g., <=12, since 13! > 32-bit).  
- The kernel will compute factorial for each element in an input array and write the result to an output array.  
- The host code will set an adequate stack size, allocate memory, launch the kernel, copy back, and print results.  
- A helper macro `CUDA_CHECK` is used for error handling.  
- For demonstration, we compute factorials for numbers 0-10.  
- The code is self‑contained and ready to compile with nvcc.  
*/

#include <cstdio>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

// Recursive device factorial function
__device__ unsigned long long fact(unsigned int n)
{
    // Base case
    if (n == 0 || n == 1)
        return 1ULL;
    // Recursive call
    return n * fact(n - 1);
}

// Kernel that calls the recursive device function
__global__ void factorial_kernel(const unsigned int *input, unsigned long long *output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        // Compute factorial for the input value
        unsigned int val = input[idx];
        output[idx] = fact(val);
    }
}

int main(void)
{
    // Number of elements
    const int N = 11; // 0 through 10
    size_t input_bytes = N * sizeof(unsigned int);
    size_t output_bytes = N * sizeof(unsigned long long);

    // Host arrays
    unsigned int h_input[N];
    unsigned long long h_output[N];

    // Initialize input data
    for (int i = 0; i < N; ++i)
        h_input[i] = i;

    // Device pointers
    unsigned int *d_input = nullptr;
    unsigned long long *d_output = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_output, output_bytes));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));

    // Set device stack size to accommodate recursion depth
    // Default is 1024 bytes; increase to 4096 bytes for safety
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 4096));

    // Launch kernel
    const int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    factorial_kernel<<<blocks, threads_per_block>>>(d_input, d_output, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost));

    // Print results
    for (int i = 0; i < N; ++i)
        printf("%2u! = %llu\n", h_input[i], h_output[i]);

    // Clean up
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    // Reset device (optional)
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
```